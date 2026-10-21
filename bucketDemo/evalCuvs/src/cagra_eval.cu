// cagra_eval.cu
// cuvs 26.10 reference harness for the bucket2 optimization loop.
//
//   build : CAGRA native build (NN-Descent, intermediate 64 -> graph 32),
//           times `reps` builds, writes the graph in graph_io's forward-graph
//           format (int64 N, int32 K, uint32 graph[N*K]).
//   search: loads a dataset + forward graph (original id order), wraps them
//           in a cuvs::neighbors::cagra::index and sweeps itopk with the same
//           CAGRA searcher for every graph, so our graph and CAGRA's native
//           graph are compared on equal terms. Prints one JSON object.
//
// Usage:
//   cagra_eval build  --data base.u8bin --out cagra.bin [--reps 3]
//                     [--graph-degree 32 --intermediate-degree 64 --nnd-iters 20]
//   cagra_eval search --data base.u8bin --graph g.bin --query q.u8bin --gt gt.ibin
//                     [--k 10] [--itopk 32,64,128,256,512] [--reps 5]

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

static std::vector<uint8_t> read_u8bin(const std::string& p, int64_t& n, int64_t& d)
{
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + p);
    int32_t h[2];
    in.read(reinterpret_cast<char*>(h), 8);
    n = h[0]; d = h[1];
    std::vector<uint8_t> v(static_cast<size_t>(n) * d);
    in.read(reinterpret_cast<char*>(v.data()), v.size());
    if (!in) throw std::runtime_error("short read " + p);
    return v;
}

static std::vector<uint32_t> read_graph(const std::string& p, int64_t& n, int32_t& k)
{
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + p);
    in.read(reinterpret_cast<char*>(&n), 8);
    in.read(reinterpret_cast<char*>(&k), 4);
    std::vector<uint32_t> g(static_cast<size_t>(n) * k);
    in.read(reinterpret_cast<char*>(g.data()), g.size() * 4);
    if (!in) throw std::runtime_error("short read " + p);
    return g;
}

static void write_graph(const std::string& p, const uint32_t* g, int64_t n, int32_t k)
{
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&n), 8);
    out.write(reinterpret_cast<const char*>(&k), 4);
    out.write(reinterpret_cast<const char*>(g), static_cast<size_t>(n) * k * 4);
}

// big-ann GT: int32 nq, int32 k, int32 ids[nq*k], (float dists follow, unused)
static std::vector<int32_t> read_gt(const std::string& p, int64_t& nq, int64_t& k)
{
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + p);
    int32_t h[2];
    in.read(reinterpret_cast<char*>(h), 8);
    nq = h[0]; k = h[1];
    std::vector<int32_t> v(static_cast<size_t>(nq) * k);
    in.read(reinterpret_cast<char*>(v.data()), v.size() * 4);
    return v;
}

static std::map<std::string, std::string> parse(int argc, char** argv)
{
    std::map<std::string, std::string> m;
    for (int i = 2; i + 1 < argc; i += 2) m[argv[i]] = argv[i + 1];
    return m;
}
static std::string get(std::map<std::string, std::string>& m, const char* k, const char* def = nullptr)
{
    auto it = m.find(k);
    if (it != m.end()) return it->second;
    if (!def) throw std::runtime_error(std::string("missing ") + k);
    return def;
}

static int do_build(std::map<std::string, std::string>& a)
{
    int64_t n, d;
    auto host = read_u8bin(get(a, "--data"), n, d);
    int reps = std::stoi(get(a, "--reps", "3"));
    raft::resources res;
    auto data = cuvs::neighbors::make_host_standard_dataset_view(
        raft::make_host_matrix_view<const uint8_t, int64_t>(host.data(), n, d));

    cuvs::neighbors::cagra::index_params ip;
    ip.graph_degree              = std::stoi(get(a, "--graph-degree", "32"));
    ip.intermediate_graph_degree = std::stoi(get(a, "--intermediate-degree", "64"));
    auto nnd = cuvs::neighbors::cagra::graph_build_params::nn_descent_params(ip.intermediate_graph_degree);
    nnd.max_iterations = std::stoi(get(a, "--nnd-iters", "20"));
    ip.graph_build_params = nnd;

    std::vector<double> ts;
    for (int r = -1; r < reps; r++) {  // r=-1: warmup
        auto t0  = Clock::now();
        auto idx = cuvs::neighbors::cagra::build(res, ip, data);
        raft::resource::sync_stream(res);
        double t = secs(t0);
        fprintf(stderr, "[cagra build] %s %.3fs\n", r < 0 ? "warmup" : "rep", t);
        if (r >= 0) ts.push_back(t);
        if (r == reps - 1) {
            auto g = raft::make_host_matrix<uint32_t, int64_t>(idx.graph().extent(0), idx.graph().extent(1));
            raft::copy(g.data_handle(), idx.graph().data_handle(), g.size(), raft::resource::get_cuda_stream(res));
            raft::resource::sync_stream(res);
            write_graph(get(a, "--out"), g.data_handle(), g.extent(0), g.extent(1));
        }
    }
    printf("{\"build_s\": [");
    for (size_t i = 0; i < ts.size(); i++) printf("%s%.4f", i ? ", " : "", ts[i]);
    printf("]}\n");
    return 0;
}

static int do_search(std::map<std::string, std::string>& a)
{
    int64_t n, d, nq, dq, gq, gk;
    int32_t deg;
    auto base  = read_u8bin(get(a, "--data"), n, d);
    auto graph = read_graph(get(a, "--graph"), gq, deg);
    auto qry   = read_u8bin(get(a, "--query"), nq, dq);
    auto gt    = read_gt(get(a, "--gt"), gq, gk);
    if (graph.size() / deg != static_cast<size_t>(n)) throw std::runtime_error("graph N != data N");
    const int k    = std::stoi(get(a, "--k", "10"));
    const int reps = std::stoi(get(a, "--reps", "5"));
    std::vector<int> itopks;
    { std::stringstream ss(get(a, "--itopk", "32,64,128,256,512")); std::string t;
      while (std::getline(ss, t, ',')) itopks.push_back(std::stoi(t)); }

    // structural check: ids in range, no self loops, no duplicates per row
    size_t bad = 0, self = 0, dup = 0;
    for (int64_t i = 0; i < n; i++) {
        std::set<uint32_t> s;
        for (int j = 0; j < deg; j++) {
            uint32_t v = graph[i * deg + j];
            if (v >= n) bad++;
            else if (v == i) self++;
            else if (!s.insert(v).second) dup++;
        }
    }

    raft::resources res;
    auto stream = raft::resource::get_cuda_stream(res);
    auto d_data  = raft::make_device_matrix<uint8_t, int64_t>(res, n, d);
    auto d_graph = raft::make_device_matrix<uint32_t, int64_t>(res, n, deg);
    auto d_q     = raft::make_device_matrix<uint8_t, int64_t>(res, nq, d);
    auto d_nb    = raft::make_device_matrix<uint32_t, int64_t>(res, nq, k);
    auto d_dist  = raft::make_device_matrix<float, int64_t>(res, nq, k);
    raft::copy(d_data.data_handle(), base.data(), base.size(), stream);
    raft::copy(d_graph.data_handle(), graph.data(), graph.size(), stream);
    raft::copy(d_q.data_handle(), qry.data(), qry.size(), stream);
    cuvs::neighbors::cagra::device_padded_index<uint8_t, uint32_t> idx(
        res, cuvs::distance::DistanceType::L2Expanded,
        cuvs::neighbors::make_device_padded_dataset_view(res, raft::make_const_mdspan(d_data.view())),
        raft::make_const_mdspan(d_graph.view()));

    std::vector<uint32_t> nb(static_cast<size_t>(nq) * k);
    printf("{\"n\": %ld, \"degree\": %d, \"bad_ids\": %zu, \"self_loops\": %zu, \"dup_edges\": %zu, \"points\": [",
           n, deg, bad, self, dup);
    for (size_t pi = 0; pi < itopks.size(); pi++) {
        cuvs::neighbors::cagra::search_params sp;
        sp.itopk_size = itopks[pi];
        sp.algo       = cuvs::neighbors::cagra::search_algo::AUTO;
        auto run = [&] {
            cuvs::neighbors::cagra::search(res, sp, idx, raft::make_const_mdspan(d_q.view()),
                                           d_nb.view(), d_dist.view());
        };
        run(); raft::resource::sync_stream(res);  // warmup
        auto t0 = Clock::now();
        for (int r = 0; r < reps; r++) run();
        raft::resource::sync_stream(res);
        double qps = nq * reps / secs(t0);
        raft::copy(nb.data(), d_nb.data_handle(), nb.size(), stream);
        raft::resource::sync_stream(res);
        size_t hit = 0;
        for (int64_t q = 0; q < nq; q++) {
            std::set<int32_t> s(gt.begin() + q * gk, gt.begin() + q * gk + k);
            for (int j = 0; j < k; j++) hit += s.count(static_cast<int32_t>(nb[q * k + j]));
        }
        printf("%s{\"itopk\": %d, \"recall\": %.5f, \"qps\": %.1f}", pi ? ", " : "", itopks[pi],
               double(hit) / (nq * k), qps);
        fflush(stdout);
    }
    printf("]}\n");
    return 0;
}

int main(int argc, char** argv)
{
    try {
        if (argc < 2) { fprintf(stderr, "usage: cagra_eval build|search ...\n"); return 1; }
        auto a = parse(argc, argv);
        std::string mode = argv[1];
        if (mode == "build") return do_build(a);
        if (mode == "search") return do_search(a);
        throw std::runtime_error("unknown mode " + mode);
    } catch (const std::exception& e) {
        fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}
