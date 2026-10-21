// CAGRA build-time benchmark: IVF-PQ vs NN-Descent kNN-graph construction.
//
// Mirrors raft::neighbors::cagra::detail::build step by step so each stage can
// be timed separately:
//   knn      : build_knn_graph (IVF-PQ build+search+refine, or NN-Descent)
//   optimize : 2-hop detour pruning + reverse edges -> final graph_degree
//   index    : cagra::index construction (dataset + graph copied to device)
// "build" = knn + optimize + index, i.e. what raft-ann-bench reports as build time.
// Dataset loading and graph saving are NOT included.
//
// Usage:
//   build_bench --algo ivf_pq|nn_descent --graph-degree 32 [options]
// Options:
//   --data PATH                 .u8bin dataset (default: sift1M base)
//   --intermediate-degree N     default 2 * graph_degree (ann-bench default)
//   --nnd-niter N               NN-Descent max iterations (default 20)
//   --refine-rate F             IVF-PQ refine rate (default 2.0, raft default)
//   --reps N                    timed repetitions (default 3)
//   --warmup N                  untimed warm-up builds (default 1)
//   --csv PATH                  append one CSV row per timed rep
//   --save PATH                 save the last built graph as .npy

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/detail/cagra/cagra_build.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/neighbors/nn_descent_types.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "utils.hpp"

namespace cagra = raft::neighbors::cagra;

struct Args {
  std::string data  = "/home/lanlu/microBenchmark/dataset/sift1M/base.1M.u8bin";
  std::string algo  = "nn_descent";
  uint32_t graph_degree        = 32;
  uint32_t intermediate_degree = 0;  // 0 -> 2 * graph_degree
  size_t nnd_niter             = 20;
  float refine_rate            = 2.0f;
  int reps                     = 3;
  int warmup                   = 1;
  std::string csv;
  std::string save;
};

static Args parse_args(int argc, char** argv)
{
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << k << "\n";
        std::exit(1);
      }
      return argv[++i];
    };
    if (k == "--data") a.data = next();
    else if (k == "--algo") a.algo = next();
    else if (k == "--graph-degree") a.graph_degree = std::stoul(next());
    else if (k == "--intermediate-degree") a.intermediate_degree = std::stoul(next());
    else if (k == "--nnd-niter") a.nnd_niter = std::stoul(next());
    else if (k == "--refine-rate") a.refine_rate = std::stof(next());
    else if (k == "--reps") a.reps = std::stoi(next());
    else if (k == "--warmup") a.warmup = std::stoi(next());
    else if (k == "--csv") a.csv = next();
    else if (k == "--save") a.save = next();
    else {
      std::cerr << "Unknown argument: " << k << "\n";
      std::exit(1);
    }
  }
  if (a.algo != "ivf_pq" && a.algo != "nn_descent") {
    std::cerr << "--algo must be ivf_pq or nn_descent\n";
    std::exit(1);
  }
  if (a.intermediate_degree == 0) a.intermediate_degree = 2 * a.graph_degree;
  return a;
}

struct StageTimes {
  double knn = 0, optimize = 0, index = 0;
  double total() const { return knn + optimize + index; }
};

int main(int argc, char** argv)
{
  using Clock   = std::chrono::high_resolution_clock;
  const Args a  = parse_args(argc, argv);
  auto seconds  = [](Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
  };

  std::cout << "=== Loading " << a.data << " ===\n";
  int64_t N = 0, D = 0;
  std::vector<uint8_t> h_data = read_u8bin_raw(a.data, N, D);
  auto dataset = raft::make_host_matrix_view<const uint8_t, int64_t>(h_data.data(), N, D);
  std::cout << "  N=" << N << " D=" << D << "\n";

  // Same runtime setup as raft-ann-bench (RMM pool + managed large workspace).
  using pool_mr_t = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  auto* orig_mr   = rmm::mr::get_current_device_resource();
  pool_mr_t pool_mr(orig_mr, 1024ull * 1024ull * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);
  auto large_mr = std::make_shared<rmm::mr::managed_memory_resource>();

  raft::resources res;
  raft::resource::set_large_workspace_resource(res, large_mr);
  raft::resource::set_cuda_stream_pool(res, std::make_shared<rmm::cuda_stream_pool>(8));

  cudaDeviceProp prop;
  int dev = 0;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);

  std::cout << "=== CAGRA build: algo=" << a.algo << " graph_degree=" << a.graph_degree
            << " intermediate_degree=" << a.intermediate_degree;
  if (a.algo == "nn_descent") std::cout << " nnd_niter=" << a.nnd_niter;
  else std::cout << " refine_rate=" << a.refine_rate;
  std::cout << " GPU=" << prop.name << " ===\n";

  auto ivf_pq_params = raft::neighbors::ivf_pq::index_params::from_dataset(dataset);
  if (a.algo == "ivf_pq") {
    std::cout << "  IVF-PQ: n_lists=" << ivf_pq_params.n_lists
              << " pq_dim=" << ivf_pq_params.pq_dim << " pq_bits=" << ivf_pq_params.pq_bits
              << " kmeans_n_iters=" << ivf_pq_params.kmeans_n_iters << "\n";
  }

  raft::neighbors::experimental::nn_descent::index_params nnd_params;
  nnd_params.graph_degree              = a.intermediate_degree;
  nnd_params.intermediate_graph_degree = 1.5 * a.intermediate_degree;
  nnd_params.max_iterations            = a.nnd_niter;

  auto final_graph = raft::make_host_matrix<uint32_t, int64_t>(N, a.graph_degree);

  auto run_once = [&]() {
    StageTimes t;
    auto sync = [&]() {
      raft::resource::sync_stream(res);
      cudaDeviceSynchronize();
    };

    auto t0       = Clock::now();
    auto knn      = raft::make_host_matrix<uint32_t, int64_t>(N, a.intermediate_degree);
    if (a.algo == "ivf_pq") {
      cagra::detail::build_knn_graph(res, dataset, knn.view(), a.refine_rate, ivf_pq_params);
    } else {
      cagra::detail::build_knn_graph<uint8_t, uint32_t>(res, dataset, knn.view(), nnd_params);
    }
    sync();
    t.knn = seconds(t0);

    t0 = Clock::now();
    cagra::detail::optimize<uint32_t>(res, knn.view(), final_graph.view());
    sync();
    t.optimize = seconds(t0);

    t0 = Clock::now();
    {
      cagra::index<uint8_t, uint32_t> idx(
        res, raft::distance::DistanceType::L2Expanded, dataset,
        raft::make_const_mdspan(final_graph.view()));
      sync();
    }
    t.index = seconds(t0);
    return t;
  };

  for (int i = 0; i < a.warmup; ++i) {
    auto t = run_once();
    std::cout << "  [warmup " << i << "] build=" << std::fixed << std::setprecision(3)
              << t.total() << "s\n";
  }

  std::ofstream csv;
  if (!a.csv.empty()) {
    bool need_header = !std::filesystem::exists(a.csv) || std::filesystem::file_size(a.csv) == 0;
    csv.open(a.csv, std::ios::app);
    if (need_header)
      csv << "dataset,gpu,algo,graph_degree,intermediate_degree,nnd_niter,refine_rate,rep,"
             "knn_s,optimize_s,index_s,build_s\n";
  }

  std::vector<StageTimes> times;
  for (int r = 0; r < a.reps; ++r) {
    auto t = run_once();
    times.push_back(t);
    std::cout << "  [rep " << r << "] knn=" << std::fixed << std::setprecision(3) << t.knn
              << "s optimize=" << t.optimize << "s index=" << t.index
              << "s build=" << t.total() << "s\n";
    if (csv.is_open()) {
      csv << std::filesystem::path(a.data).parent_path().filename().string() << ","
          << prop.name << "," << a.algo << "," << a.graph_degree << ","
          << a.intermediate_degree << "," << a.nnd_niter << "," << a.refine_rate << "," << r
          << "," << t.knn << "," << t.optimize << "," << t.index << "," << t.total() << "\n";
    }
  }

  if (!times.empty()) {
    StageTimes mean;
    double best = times[0].total();
    for (auto& t : times) {
      mean.knn += t.knn / times.size();
      mean.optimize += t.optimize / times.size();
      mean.index += t.index / times.size();
      best = std::min(best, t.total());
    }
    std::cout << "\n=== Summary (" << a.algo << ", degree " << a.graph_degree << "/"
              << a.intermediate_degree << ", " << times.size() << " reps) ===\n"
              << std::fixed << std::setprecision(3)
              << "  knn graph (mean):   " << mean.knn << "s\n"
              << "  optimize  (mean):   " << mean.optimize << "s\n"
              << "  index     (mean):   " << mean.index << "s\n"
              << "  build     (mean):   " << mean.total() << "s\n"
              << "  build     (best):   " << best << "s\n";
  }

  if (!a.save.empty()) {
    write_npy_2d<uint32_t>(a.save, final_graph.data_handle(), N, a.graph_degree);
    std::cout << "  Saved graph to " << a.save << "\n";
  }

  rmm::mr::set_current_device_resource(orig_mr);
  return 0;
}
