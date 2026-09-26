// translate_graph_ids.hpp
// Translates a [N, K] uint32 graph/index file (forward-graph format: int64 N,
// int32 K, then N*K uint32) from "new" (reordered) id space back to
// "original" id space, using inverse_perm[new_id] = old_id -- exactly the
// file bucket2's inline reorder (bucket.cu's Step 4.5) writes.
//
// Both the ROW POSITION (which point this row describes) and every VALUE in
// the row (each entry references another point by id) get translated:
//   output_row[inverse_perm[new_id]][k] = inverse_perm[ input_row[new_id][k] ]
//
// This is currently the only place this codebase needs old<->new
// translation at all: no build-time consumer reads inverse_perm.bin (see
// bucket.cu's Step 4.5 comment on why perm.bin was dropped but
// inverse_perm.bin was kept) -- it exists for exactly this use, translating
// a finished index/search-result back to the coordinate system ground-truth
// files and any external caller use.
//
// inverse_perm must be fully resident (O(N) uint32, much smaller than the
// graph itself whenever K>1) -- a row's destination depends on it, and any
// value in any row could reference any other point, so there's no way to
// stream around needing random access to the whole table. The graph itself
// is read/written in a streaming fashion (row batches via pwrite to each
// row's known destination), never fully resident.

#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "graph_io.hpp"

namespace translate_ids {

inline std::vector<uint32_t> read_inverse_perm(const std::string& path, int64_t N) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("translate_ids: cannot open " + path);
    in.seekg(0, std::ios::end);
    std::streamoff sz = in.tellg();
    if (sz != static_cast<std::streamoff>(N) * static_cast<std::streamoff>(sizeof(uint32_t)))
        throw std::runtime_error(
            "translate_ids: " + path + " size (" + std::to_string(sz) +
            " bytes) doesn't match graph N=" + std::to_string(N) +
            " (expected " + std::to_string(N * sizeof(uint32_t)) + " bytes)");
    in.seekg(0);
    std::vector<uint32_t> inv(static_cast<size_t>(N));
    in.read(reinterpret_cast<char*>(inv.data()), sz);
    if (!in.good())
        throw std::runtime_error("translate_ids: failed reading " + path);
    return inv;
}

inline void translate_graph_to_original_order(
    const std::string& graph_path,
    const std::string& inverse_perm_path,
    const std::string& output_path,
    int64_t batch_rows = 1 << 14)
{
    int64_t N = 0; int32_t K_out = 0;
    graph_io::read_forward_graph_header(graph_path, N, K_out);

    std::vector<uint32_t> inverse_perm = read_inverse_perm(inverse_perm_path, N);

    std::ifstream in(graph_path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("translate_ids: cannot open " + graph_path);
    const std::streamoff header_bytes = sizeof(int64_t) + sizeof(int32_t);
    in.seekg(header_bytes);

    int fd = ::open(output_path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd < 0)
        throw std::runtime_error("translate_ids: cannot open " + output_path + " for writing");

    const int64_t row_bytes = static_cast<int64_t>(K_out) * sizeof(uint32_t);
    {
        char hdr[12];
        std::memcpy(hdr, &N, sizeof(int64_t));
        std::memcpy(hdr + sizeof(int64_t), &K_out, sizeof(int32_t));
        if (::pwrite(fd, hdr, sizeof(hdr), 0) != static_cast<ssize_t>(sizeof(hdr))) {
            ::close(fd);
            throw std::runtime_error("translate_ids: header write failed");
        }
    }
    if (::ftruncate(fd, static_cast<off_t>(header_bytes) + N * row_bytes) != 0) {
        ::close(fd);
        throw std::runtime_error("translate_ids: ftruncate failed");
    }

    std::vector<uint32_t> batch(static_cast<size_t>(batch_rows) * K_out);
    std::atomic<bool> failed{false};
    std::atomic<int64_t> failed_new_id{-1};

    for (int64_t lo = 0; lo < N; lo += batch_rows) {
        int64_t rows = std::min<int64_t>(batch_rows, N - lo);
        in.read(reinterpret_cast<char*>(batch.data()),
                static_cast<std::streamsize>(static_cast<size_t>(rows) * K_out * sizeof(uint32_t)));
        if (!in.good()) {
            ::close(fd);
            throw std::runtime_error("translate_ids: failed reading graph batch at row " + std::to_string(lo));
        }

        // pwrite is thread-safe (doesn't share a file position), so this can
        // translate + write each row independently in parallel. Exceptions
        // can't safely cross an OpenMP region boundary, so failures are
        // recorded in a shared flag and thrown after the loop instead.
        #pragma omp parallel for schedule(static)
        for (int64_t r = 0; r < rows; ++r) {
            if (failed.load(std::memory_order_relaxed)) continue;
            int64_t new_id = lo + r;
            uint32_t* row = batch.data() + static_cast<size_t>(r) * K_out;
            for (int32_t k = 0; k < K_out; ++k)
                row[k] = inverse_perm[row[k]];

            uint32_t old_id = inverse_perm[static_cast<size_t>(new_id)];
            off_t off = static_cast<off_t>(header_bytes) + static_cast<off_t>(old_id) * row_bytes;
            ssize_t written = ::pwrite(fd, row, static_cast<size_t>(row_bytes), off);
            if (written != static_cast<ssize_t>(row_bytes)) {
                failed.store(true, std::memory_order_relaxed);
                failed_new_id.store(new_id, std::memory_order_relaxed);
            }
        }
        if (failed.load()) {
            ::close(fd);
            throw std::runtime_error("translate_ids: pwrite failed for new_id " +
                                     std::to_string(failed_new_id.load()));
        }
    }

    if (::close(fd) != 0)
        throw std::runtime_error("translate_ids: close failed for " + output_path);
}

}  // namespace translate_ids
