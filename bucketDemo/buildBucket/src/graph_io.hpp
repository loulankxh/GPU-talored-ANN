// graph_io.hpp
// Shared binary I/O for the graph-optimize pipeline (optimize_chunked.cu,
// prune_windowed.cu, reorder.cpp). Pulled out of those three files so the
// file formats have exactly one reader/writer each instead of three
// slightly-drifting copies.
//
// Formats:
//   forward/cagra graph (read_forward_graph / write_graph):
//     int64_t  N
//     int32_t  K_out
//     uint32_t graph[N * K_out]
//
//   bucket_offsets.bin (read_bucket_offsets), produced by reorder.cpp /
//   bucket2 --reorder, indexed by *processing position* (not raw bucket id
//   -- position pos's bucket occupies new-id range [offsets[pos], offsets[pos+1])):
//     int32_t  n_buckets
//     uint32_t offsets[n_buckets + 1]
//
//   centroid_knn.bin (CentroidKnn / read_centroid_knn), written by bucket.cu
//   --reorder: bucket -> its K nearest buckets, used to derive both the
//   DiskJoin processing order and the prune visibility window:
//     int64_t  n_centroids
//     int32_t  K
//     uint32_t graph[n_centroids * K]
//
//   vector_knn(.bin|_reordered.bin) (VectorKnn / read_vector_knn /
//   write_vector_knn), per-point unpruned KNN candidate list. -1 = padding.
//     int64_t  N
//     int32_t  M
//     int32_t  graph[N * M]

#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace graph_io {

// ---------------------------------------------------------------------
// Forward / CAGRA graph (uint32_t neighbor ids, fixed degree K_out)
// ---------------------------------------------------------------------

inline void read_forward_graph(const std::string& path,
                               std::vector<uint32_t>& graph,
                               int64_t& N, int32_t& K_out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open: " + path);
    in.read(reinterpret_cast<char*>(&N), sizeof(int64_t));
    in.read(reinterpret_cast<char*>(&K_out), sizeof(int32_t));
    if (!in.good() || N <= 0 || K_out <= 0)
        throw std::runtime_error("Invalid forward graph header: " + path);

    const size_t cnt = static_cast<size_t>(N) * static_cast<size_t>(K_out);
    graph.resize(cnt);
    in.read(reinterpret_cast<char*>(graph.data()),
            static_cast<std::streamsize>(cnt * sizeof(uint32_t)));
    if (!in.good())
        throw std::runtime_error("Failed to read forward graph payload: " + path);
}

// Header-only peek, no payload read -- Method E streams the graph in
// chunks instead of loading it whole, so it only needs N/K_out up front.
inline void read_forward_graph_header(const std::string& path, int64_t& N, int32_t& K_out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open: " + path);
    in.read(reinterpret_cast<char*>(&N), sizeof(int64_t));
    in.read(reinterpret_cast<char*>(&K_out), sizeof(int32_t));
    if (!in.good() || N <= 0 || K_out <= 0)
        throw std::runtime_error("Invalid forward graph header: " + path);
}

inline void write_graph(const std::string& path,
                        const uint32_t* graph,
                        int64_t N, int32_t K_out)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open: " + path);
    out.write(reinterpret_cast<const char*>(&N), sizeof(int64_t));
    out.write(reinterpret_cast<const char*>(&K_out), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(graph),
              static_cast<std::streamsize>(static_cast<size_t>(N) * K_out * sizeof(uint32_t)));
    if (!out.good())
        throw std::runtime_error("Failed to write graph: " + path);
}

// Streaming writer for callers (Method E) that produce output chunk-by-chunk
// and never hold the full [N, K_out] array in host memory. Usage:
//   StreamingGraphWriter w(path, N, K_out);
//   for each chunk in increasing row order: w.write_rows(ptr, row_lo, n_rows);
class StreamingGraphWriter {
public:
    StreamingGraphWriter(const std::string& path, int64_t N, int32_t K_out)
        : N_(N), K_out_(K_out)
    {
        out_.open(path, std::ios::binary);
        if (!out_.is_open())
            throw std::runtime_error("Cannot open for streaming write: " + path);
        out_.write(reinterpret_cast<const char*>(&N), sizeof(int64_t));
        out_.write(reinterpret_cast<const char*>(&K_out), sizeof(int32_t));
        if (!out_.good())
            throw std::runtime_error("Failed to write graph header: " + path);
    }

    // Rows must be supplied in strictly increasing, contiguous row_lo order
    // (chunk c's call must start where chunk c-1's left off) -- this is a
    // pure sequential append, no seeking.
    void write_rows(const uint32_t* rows, int64_t row_lo, int64_t n_rows) {
        if (row_lo != rows_written_)
            throw std::runtime_error(
                "StreamingGraphWriter: out-of-order write (expected row " +
                std::to_string(rows_written_) + ", got " + std::to_string(row_lo) + ")");
        out_.write(reinterpret_cast<const char*>(rows),
                   static_cast<std::streamsize>(static_cast<size_t>(n_rows) * K_out_ * sizeof(uint32_t)));
        if (!out_.good())
            throw std::runtime_error("StreamingGraphWriter: write failed");
        rows_written_ += n_rows;
    }

    ~StreamingGraphWriter() {
        // Best-effort: don't throw from a destructor. Callers should check
        // rows_written() == N themselves before letting this go out of scope.
    }

    int64_t rows_written() const { return rows_written_; }

private:
    std::ofstream out_;
    int64_t N_;
    int32_t K_out_;
    int64_t rows_written_ = 0;
};

// ---------------------------------------------------------------------
// bucket_offsets.bin
// ---------------------------------------------------------------------

inline std::vector<uint32_t> read_bucket_offsets(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open bucket-offsets: " + path);
    int32_t n_buckets = 0;
    in.read(reinterpret_cast<char*>(&n_buckets), sizeof(int32_t));
    if (!in.good() || n_buckets <= 0)
        throw std::runtime_error("Invalid bucket-offsets header: " + path);
    std::vector<uint32_t> offsets(n_buckets + 1);
    in.read(reinterpret_cast<char*>(offsets.data()),
            static_cast<std::streamsize>((n_buckets + 1) * sizeof(uint32_t)));
    if (!in.good())
        throw std::runtime_error("Failed to read bucket-offsets payload: " + path);
    return offsets;
}

inline void write_bucket_offsets(const std::string& path,
                                  const std::vector<uint32_t>& offsets_new)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open: " + path);
    int32_t n_buckets = static_cast<int32_t>(offsets_new.size()) - 1;
    out.write(reinterpret_cast<const char*>(&n_buckets), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(offsets_new.data()),
              static_cast<std::streamsize>(offsets_new.size() * sizeof(uint32_t)));
}

// ---------------------------------------------------------------------
// bucket_process_order.bin: order[pos] = original bucket id placed at
// processing position pos (bucket_offsets.bin's index space). Written by
// reorder.cpp alongside bucket_offsets.bin so downstream consumers (like
// prune_windowed) can recover "which original bucket owns new-id range
// [offsets[pos], offsets[pos+1])" *exactly*, without recomputing
// bucket_order::compute_bucket_processing_order a second time and risking
// silent drift if its --order-window doesn't match what reorder used.
// ---------------------------------------------------------------------

inline void write_bucket_process_order(const std::string& path,
                                        const std::vector<int32_t>& order)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open: " + path);
    int32_t n_buckets = static_cast<int32_t>(order.size());
    out.write(reinterpret_cast<const char*>(&n_buckets), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(order.data()),
              static_cast<std::streamsize>(order.size() * sizeof(int32_t)));
}

inline std::vector<int32_t> read_bucket_process_order(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open bucket-process-order: " + path);
    int32_t n_buckets = 0;
    in.read(reinterpret_cast<char*>(&n_buckets), sizeof(int32_t));
    if (!in.good() || n_buckets <= 0)
        throw std::runtime_error("Invalid bucket-process-order header: " + path);
    std::vector<int32_t> order(static_cast<size_t>(n_buckets));
    in.read(reinterpret_cast<char*>(order.data()),
            static_cast<std::streamsize>(order.size() * sizeof(int32_t)));
    if (!in.good())
        throw std::runtime_error("Failed to read bucket-process-order payload: " + path);
    return order;
}

// ---------------------------------------------------------------------
// centroid_knn.bin: bucket -> K nearest buckets (drives both DiskJoin
// processing order and the prune visibility window)
// ---------------------------------------------------------------------

struct CentroidKnn {
    int64_t n_centroids = 0;
    int32_t K = 0;
    std::vector<uint32_t> graph;   // [n_centroids * K], row-major
};

inline CentroidKnn read_centroid_knn(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open centroid_knn: " + path);

    CentroidKnn ck;
    in.read(reinterpret_cast<char*>(&ck.n_centroids), sizeof(int64_t));
    in.read(reinterpret_cast<char*>(&ck.K), sizeof(int32_t));
    if (!in.good() || ck.n_centroids <= 0 || ck.K <= 0)
        throw std::runtime_error("Invalid centroid_knn header: " + path);
    ck.graph.resize(static_cast<size_t>(ck.n_centroids) * ck.K);
    in.read(reinterpret_cast<char*>(ck.graph.data()),
            static_cast<std::streamsize>(ck.graph.size() * sizeof(uint32_t)));
    if (!in.good())
        throw std::runtime_error("Failed to read centroid_knn payload: " + path);
    return ck;
}

// ---------------------------------------------------------------------
// vector_knn(.bin|_reordered.bin): per-point unpruned KNN candidates
// ---------------------------------------------------------------------

struct VectorKnn {
    int64_t N = 0;
    int32_t M = 0;
    std::vector<int32_t> graph;   // [N * M], row-major, -1 = padding
};

inline VectorKnn read_vector_knn(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open vector_knn: " + path);

    VectorKnn vk;
    in.read(reinterpret_cast<char*>(&vk.N), sizeof(int64_t));
    in.read(reinterpret_cast<char*>(&vk.M), sizeof(int32_t));
    if (!in.good() || vk.N <= 0 || vk.M <= 0)
        throw std::runtime_error("Invalid vector_knn header: " + path);
    vk.graph.resize(static_cast<size_t>(vk.N) * vk.M);
    in.read(reinterpret_cast<char*>(vk.graph.data()),
            static_cast<std::streamsize>(vk.graph.size() * sizeof(int32_t)));
    if (!in.good())
        throw std::runtime_error("Failed to read vector_knn payload: " + path);
    return vk;
}

// Header-only peek (doesn't load the payload) -- prune_windowed reads rows
// on demand through a persistent ifstream instead of loading the whole file.
inline void read_vector_knn_header(const std::string& path, int64_t& N, int32_t& M)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open vector_knn: " + path);
    in.read(reinterpret_cast<char*>(&N), sizeof(int64_t));
    in.read(reinterpret_cast<char*>(&M), sizeof(int32_t));
    if (!in.good() || N <= 0 || M <= 0)
        throw std::runtime_error("Invalid vector_knn header: " + path);
}

inline void write_vector_knn(const std::string& path,
                             const std::vector<int32_t>& graph,
                             int64_t N, int32_t M)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open: " + path);
    out.write(reinterpret_cast<const char*>(&N), sizeof(int64_t));
    out.write(reinterpret_cast<const char*>(&M), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(graph.data()),
              static_cast<std::streamsize>(graph.size() * sizeof(int32_t)));
}

// Persistent-handle random-access reader for vector_knn(_reordered).bin --
// used by prune_windowed's BeladyBucketCache loader to pull one bucket's
// contiguous row range (seek + sequential read) without loading the whole
// file. One instance per (single-threaded) reader; not thread-safe (shares
// one seek position / one ifstream).
class VectorKnnReader {
public:
    explicit VectorKnnReader(const std::string& path) : in_(path, std::ios::binary) {
        if (!in_.is_open())
            throw std::runtime_error("Cannot open vector_knn: " + path);
        in_.read(reinterpret_cast<char*>(&N_), sizeof(int64_t));
        in_.read(reinterpret_cast<char*>(&M_), sizeof(int32_t));
        if (!in_.good() || N_ <= 0 || M_ <= 0)
            throw std::runtime_error("Invalid vector_knn header: " + path);
        header_bytes_ = static_cast<std::streamoff>(sizeof(int64_t) + sizeof(int32_t));
    }

    int64_t N() const { return N_; }
    int32_t M() const { return M_; }

    // Reads rows [row_lo, row_hi) into out (resized to (row_hi-row_lo)*M()).
    void read_rows(int64_t row_lo, int64_t row_hi, std::vector<int32_t>& out) {
        int64_t n_rows = row_hi - row_lo;
        out.resize(static_cast<size_t>(n_rows) * M_);
        in_.seekg(header_bytes_ + static_cast<std::streamoff>(row_lo) * M_ * sizeof(int32_t));
        in_.read(reinterpret_cast<char*>(out.data()),
                 static_cast<std::streamsize>(out.size() * sizeof(int32_t)));
        if (!in_.good())
            throw std::runtime_error("VectorKnnReader: read_rows failed");
    }

private:
    std::ifstream in_;
    int64_t N_ = 0;
    int32_t M_ = 0;
    std::streamoff header_bytes_ = 0;
};

// Generic flat binary vector read/write, used for perm.bin / inverse_perm.bin.
template <typename T>
inline void write_vector_bin(const std::string& path, const std::vector<T>& v) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open: " + path);
    out.write(reinterpret_cast<const char*>(v.data()),
              static_cast<std::streamsize>(v.size() * sizeof(T)));
}

}  // namespace graph_io
