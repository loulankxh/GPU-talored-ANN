// translate_ids.cpp
// Standalone CLI wrapper around translate_graph_ids.hpp: translates a final
// graph/index (in the reordered "new" id space bucket2/optimize_chunked/
// prune_windowed all work in) back to the original dataset's id order, so it
// can be compared against ground truth files or handed to anything else that
// expects original ids. See translate_graph_ids.hpp for the algorithm.
//
// Format is picked from --graph's extension: ".npy" uses the npy [N,K]
// int64 path (e.g. neighbors.npy), anything else uses the raw forward-graph
// .bin path (e.g. vector_knn.bin / cagra_graph.bin). --output should use a
// matching extension.
//
// Usage:
//   ./translate_ids \
//       -g output/.../cagra_graph_merged.bin \
//       -p output/.../inverse_perm.bin \
//       -o output/.../cagra_graph_original_order.bin
//
//   ./translate_ids \
//       -g output/.../neighbors.npy \
//       -p output/.../inverse_perm.bin \
//       -o output/.../neighbors_original_order.npy

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include "translate_graph_ids.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv)
{
    try {
        po::options_description desc(
            "Translate a graph/index from reordered id space back to original id order");
        desc.add_options()
            ("help,h", "Show help")
            ("graph,g", po::value<std::string>()->required(),
                "Final graph in new/reordered id space (e.g. optimize_chunked's output, "
                "or a neighbors.npy)")
            ("inverse-perm,p", po::value<std::string>()->required(),
                "inverse_perm.bin (from bucket2's inline reorder, inverse_perm[new_id]=old_id)")
            ("output,o", po::value<std::string>()->required(),
                "Output graph, same format, rows and neighbor ids translated to original order");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc << "\n"; return 0; }
        po::notify(vm);

        const std::string graph_path = vm["graph"].as<std::string>();
        const std::string inv_perm_path = vm["inverse-perm"].as<std::string>();
        const std::string output_path = vm["output"].as<std::string>();

        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();

        const bool is_npy = std::filesystem::path(graph_path).extension() == ".npy";
        if (is_npy) {
            translate_ids::translate_npy_to_original_order(graph_path, inv_perm_path, output_path);
        } else {
            translate_ids::translate_graph_to_original_order(graph_path, inv_perm_path, output_path);
        }

        double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
        std::cout << "Wrote " << output_path << " [" << elapsed << "s]\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
