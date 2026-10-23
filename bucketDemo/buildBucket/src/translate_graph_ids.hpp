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
// is never fully resident beyond the bounded output window (see
// translate_graph_to_original_order).
//
// translate_npy_to_original_order() below does the same translation for a
// .npy [N, K] int64 graph (e.g. neighbors.npy) instead of the raw
// forward-graph .bin format; unlike the .bin path it is fully in-memory
// (npy graphs here are small enough) and treats -1 entries as padding,
// leaving them untranslated.

#pragma once

#include <algorithm>
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
#include "load.hpp"

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

// Output is assembled in an in-memory window of destination rows
// [win_lo, win_hi) (at most out_buffer_bytes), filled by streaming the input
// graph once per window, then written with one sequential write. Replaces
// one pwrite syscall per row (~2s for 1M rows of 128 B) with ~nothing; for
// N*K*4 > out_buffer_bytes it takes ceil(N*K*4 / out_buffer_bytes) passes
// over the (sequentially read) input, so host memory stays bounded.
inline void translate_graph_to_original_order(
    const std::string& graph_path,
    const std::string& inverse_perm_path,
    const std::string& output_path,
    int64_t batch_rows = 1 << 14,
    size_t out_buffer_bytes = size_t(1) << 30)
{
    int64_t N = 0; int32_t K_out = 0;
    graph_io::read_forward_graph_header(graph_path, N, K_out);

    std::vector<uint32_t> inverse_perm = read_inverse_perm(inverse_perm_path, N);

    const int in_fd = ::open(graph_path.c_str(), O_RDONLY);
    if (in_fd < 0)
        throw std::runtime_error("translate_ids: cannot open " + graph_path);
    const std::streamoff header_bytes = sizeof(int64_t) + sizeof(int32_t);

    int fd = ::open(output_path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd < 0)
        throw std::runtime_error("translate_ids: cannot open " + output_path + " for writing");
    auto fail = [&](const std::string& msg) {
        ::close(fd); ::close(in_fd);
        throw std::runtime_error("translate_ids: " + msg);
    };
    auto pwrite_all = [&](const char* p, size_t n, off_t off) {
        while (n > 0) {
            ssize_t w = ::pwrite(fd, p, n, off);
            if (w <= 0) fail("pwrite failed at offset " + std::to_string(off));
            p += w; n -= static_cast<size_t>(w); off += w;
        }
    };

    const int64_t row_bytes = static_cast<int64_t>(K_out) * sizeof(uint32_t);
    {
        char hdr[12];
        std::memcpy(hdr, &N, sizeof(int64_t));
        std::memcpy(hdr + sizeof(int64_t), &K_out, sizeof(int32_t));
        pwrite_all(hdr, sizeof(hdr), 0);
    }

    const int64_t win_rows = std::max<int64_t>(1, std::min<int64_t>(
        N, static_cast<int64_t>(out_buffer_bytes / static_cast<size_t>(row_bytes))));
    std::vector<uint32_t> window(static_cast<size_t>(win_rows) * K_out);
    std::atomic<int64_t> bad_new_id{-1};

    for (int64_t win_lo = 0; win_lo < N; win_lo += win_rows) {
        const int64_t win_hi = std::min<int64_t>(N, win_lo + win_rows);
        int64_t filled = 0;
        // 每个窗都要把整份输入扫一遍 (SIFT100M: 13 窗 × 12.8 GB)；原来单线程 ifstream 按 2 MB 批读，
        // 读是瓶颈 (translate 55 s)。现在各线程自己 pread 自己的批次、直接填窗 (窗内目标行互不相同，
        // 无冲突)，结果逐字节相同。参考 TAGORE order_merge_new: 多段并行流式读回局部结果。
        const int64_t n_batches = (N + batch_rows - 1) / batch_rows;
        std::atomic<bool> read_fail{false};
        #pragma omp parallel reduction(+:filled)
        {
            std::vector<uint32_t> tb(static_cast<size_t>(batch_rows) * K_out);
            #pragma omp for schedule(dynamic, 4)
            for (int64_t bi = 0; bi < n_batches; ++bi) {
                const int64_t lo = bi * batch_rows;
                const int64_t rows = std::min<int64_t>(batch_rows, N - lo);
                char* p = reinterpret_cast<char*>(tb.data());
                size_t want = static_cast<size_t>(rows) * K_out * sizeof(uint32_t);
                off_t off = static_cast<off_t>(header_bytes) + static_cast<off_t>(lo) * row_bytes;
                while (want > 0) {
                    ssize_t r = ::pread(in_fd, p, want, off);
                    if (r <= 0) break;
                    p += r; want -= static_cast<size_t>(r); off += r;
                }
                if (want != 0) { read_fail.store(true, std::memory_order_relaxed); continue; }
                for (int64_t r = 0; r < rows; ++r) {
                    int64_t new_id = lo + r;
                    uint32_t old_id = inverse_perm[static_cast<size_t>(new_id)];
                    if (old_id < win_lo || old_id >= win_hi) continue;   // also rejects old_id >= N
                    const uint32_t* src = tb.data() + static_cast<size_t>(r) * K_out;
                    uint32_t* dst = window.data() + static_cast<size_t>(old_id - win_lo) * K_out;
                    for (int32_t k = 0; k < K_out; ++k) {
                        uint32_t v = src[k];
                        if (v >= static_cast<uint64_t>(N)) {   // bounds check: neighbor id out of range
                            bad_new_id.store(new_id, std::memory_order_relaxed);
                            v = 0;
                        }
                        dst[k] = inverse_perm[v];
                    }
                    ++filled;
                }
            }
        }
        if (read_fail.load()) fail("failed reading graph batch");
        if (bad_new_id.load() >= 0)
            fail("neighbor id >= N in row (new id) " + std::to_string(bad_new_id.load()));
        // inverse_perm is a permutation iff every destination row is hit once
        if (filled != win_hi - win_lo)
            fail("inverse_perm is not a permutation: window [" + std::to_string(win_lo) + ", " +
                 std::to_string(win_hi) + ") got " + std::to_string(filled) + " rows");
        pwrite_all(reinterpret_cast<const char*>(window.data()),
                   static_cast<size_t>(win_hi - win_lo) * static_cast<size_t>(row_bytes),
                   static_cast<off_t>(header_bytes) + static_cast<off_t>(win_lo) * row_bytes);
    }

    ::close(in_fd);
    if (::close(fd) != 0)
        throw std::runtime_error("translate_ids: close failed for " + output_path);
}

// Same translation as translate_graph_to_original_order, but for a .npy
// [N, K] int64 graph (e.g. neighbors.npy) instead of the raw forward-graph
// .bin format. -1 entries are padding (short neighbor lists) and are left
// as -1 rather than translated.
inline void translate_npy_to_original_order(
    const std::string& npy_path,
    const std::string& inverse_perm_path,
    const std::string& output_path)
{
    std::vector<int64_t> graph;
    int64_t N = 0, K_out = 0;
    load::read_npy_int64_2d(npy_path, graph, N, K_out);

    std::vector<uint32_t> inverse_perm = read_inverse_perm(inverse_perm_path, N);

    std::vector<int64_t> out(static_cast<size_t>(N) * K_out, -1);

    #pragma omp parallel for schedule(static)
    for (int64_t new_id = 0; new_id < N; ++new_id) {
        const int64_t* src = graph.data() + static_cast<size_t>(new_id) * K_out;
        int64_t* dst = out.data() + static_cast<size_t>(inverse_perm[new_id]) * K_out;
        for (int64_t k = 0; k < K_out; ++k) {
            int64_t nb = src[k];
            dst[k] = (nb < 0) ? -1 : static_cast<int64_t>(inverse_perm[static_cast<size_t>(nb)]);
        }
    }

    load::write_npy_int64_2d(output_path, out, N, K_out);
}

}  // namespace translate_ids
