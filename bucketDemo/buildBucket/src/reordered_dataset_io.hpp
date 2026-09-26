// reordered_dataset_io.hpp
// Pass 3 of bucket2's inline round-1 reorder (see bucket.cu's main loop):
// once Pass 2 (compute_bucket_reorder) has computed every bucket's exact
// final size, the actual data-move can skip both the "hold two full copies
// in RAM" approach and the "random-seek gather from the input file"
// approach -- each bucket's final byte range in the single output file is
// already known, so per-bucket write buffers (same accounting as
// BucketVectorAccumulator, bucket.cu:767) can flush straight to that exact
// range via pwrite, with no later "merge many files into one" pass.
//
// Input scan order is a single sequential pass over the original dataset;
// output lands via one pwrite per flush, each confined to its own bucket's
// non-overlapping region -- so within a bucket, writes are strictly
// sequential (append-style), and the whole thing needs no more host memory
// than the shared buffer budget (same "total budget / bucket count"
// convention as BucketVectorAccumulator::create).
//
// Ordering contract: add() MUST be called for old_id = 0, 1, 2, ..., N-1 in
// that exact order (matching compute_bucket_reorder's own perm[] derivation
// in bucket.cu, which assigns each bucket's records in ascending old_id
// order) -- and, like BucketVectorAccumulator's existing add() calls at
// Step4, from a single thread/serially. A bucket's position-within-region
// is implicit (this writer's own running cursor), not passed in; if callers
// scanned out of order, the resulting file would still be a valid
// permutation of the dataset, but a *different* one than perm.bin/
// inverse_perm.bin (computed separately by compute_bucket_reorder) claims
// it is -- so ordering must match exactly, not just "some order".
//
// No CUDA dependency -- included from bucket.cu, but also compiled and
// tested standalone via tests/test_reordered_dataset_io.cpp.

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace reordered_io {

// ---------------------------------------------------------------------
// ReorderedDatasetWriter<DataT>: Pass 3's writer.
// ---------------------------------------------------------------------
template <typename DataT>
class ReorderedDatasetWriter {
public:
    struct BucketBuf {
        std::vector<DataT> data;       // capacity_records * D
        size_t capacity_records = 0;
        size_t count = 0;              // pending, not yet flushed
        int64_t cursor_records = 0;    // absolute record index of this bucket's next write
        int64_t region_end_records = 0;  // for bounds-checking (== offsets_new[pos_of[b]+1])
    };

    int fd = -1;
    int64_t D = 0;
    int64_t header_bytes = 0;
    std::vector<BucketBuf> bufs;       // indexed by RAW bucket id

    ~ReorderedDatasetWriter() {
        if (fd >= 0) ::close(fd);
    }
    ReorderedDatasetWriter() = default;
    ReorderedDatasetWriter(const ReorderedDatasetWriter&) = delete;
    ReorderedDatasetWriter& operator=(const ReorderedDatasetWriter&) = delete;
    ReorderedDatasetWriter(ReorderedDatasetWriter&& o) noexcept { *this = std::move(o); }
    ReorderedDatasetWriter& operator=(ReorderedDatasetWriter&& o) noexcept {
        if (this != &o) {
            if (fd >= 0) ::close(fd);
            fd = o.fd; o.fd = -1;
            D = o.D;
            header_bytes = o.header_bytes;
            bufs = std::move(o.bufs);
        }
        return *this;
    }

    // offsets_new: size n_buckets+1, indexed by PROCESSING POSITION (as
    // bucket_offsets.bin already is). pos_of: size n_buckets, raw bucket id
    // -> its processing position (inverse of bucket_process_order).
    // total_buffer_budget_bytes is shared across all buckets, same
    // convention as BucketVectorAccumulator::create.
    static ReorderedDatasetWriter create(
        const std::string& path, int64_t D, int64_t n_buckets,
        const std::vector<uint32_t>& offsets_new,
        const std::vector<int32_t>& pos_of,
        size_t total_buffer_budget_bytes)
    {
        ReorderedDatasetWriter w;
        w.D = D;
        int64_t total_records = static_cast<int64_t>(offsets_new.back());
        int64_t row_bytes = D * static_cast<int64_t>(sizeof(DataT));
        w.header_bytes = 2 * static_cast<int64_t>(sizeof(int32_t));

        w.fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
        if (w.fd < 0)
            throw std::runtime_error("ReorderedDatasetWriter: cannot open " + path);

        // Header: int32 N, int32 D (matches load::read_fbin_header's format).
        int32_t hdrN = static_cast<int32_t>(total_records);
        int32_t hdrD = static_cast<int32_t>(D);
        char hdr[8];
        std::memcpy(hdr, &hdrN, 4);
        std::memcpy(hdr + 4, &hdrD, 4);
        if (::pwrite(w.fd, hdr, sizeof(hdr), 0) != static_cast<ssize_t>(sizeof(hdr)))
            throw std::runtime_error("ReorderedDatasetWriter: header write failed");

        // Pre-size the file so every bucket's region is real (not relying on
        // sparse-file semantics for correctness, only for space efficiency
        // on filesystems that support holes).
        if (::ftruncate(w.fd, w.header_bytes + total_records * row_bytes) != 0)
            throw std::runtime_error("ReorderedDatasetWriter: ftruncate failed");

        constexpr size_t kMinBufferBytes = 4096;
        size_t nb = static_cast<size_t>(std::max<int64_t>(1, n_buckets));
        size_t buf_bytes_per_bucket = std::max<size_t>(
            kMinBufferBytes, total_buffer_budget_bytes / nb);
        size_t cap_records = std::max<size_t>(1, buf_bytes_per_bucket / static_cast<size_t>(row_bytes));

        w.bufs.resize(static_cast<size_t>(n_buckets));
        for (int64_t b = 0; b < n_buckets; ++b) {
            auto& buf = w.bufs[static_cast<size_t>(b)];
            buf.capacity_records = cap_records;
            buf.data.resize(cap_records * static_cast<size_t>(D));
            int32_t pos = pos_of[static_cast<size_t>(b)];
            buf.cursor_records = static_cast<int64_t>(offsets_new[static_cast<size_t>(pos)]);
            buf.region_end_records = static_cast<int64_t>(offsets_new[static_cast<size_t>(pos) + 1]);
        }
        return w;
    }

    void flush(int64_t bucket_id) {
        auto& buf = bufs[static_cast<size_t>(bucket_id)];
        if (buf.count == 0) return;
        int64_t row_bytes = D * static_cast<int64_t>(sizeof(DataT));
        size_t nbytes = buf.count * static_cast<size_t>(row_bytes);
        off_t off = header_bytes + buf.cursor_records * row_bytes;
        ssize_t written = ::pwrite(fd, buf.data.data(), nbytes, off);
        if (written != static_cast<ssize_t>(nbytes))
            throw std::runtime_error("ReorderedDatasetWriter: flush failed for bucket " +
                                     std::to_string(bucket_id));
        buf.cursor_records += static_cast<int64_t>(buf.count);
        if (buf.cursor_records > buf.region_end_records)
            throw std::runtime_error("ReorderedDatasetWriter: bucket " + std::to_string(bucket_id) +
                                     " overflowed its region (more points added than its final size)");
        buf.count = 0;
    }

    // vec must point to exactly D elements. Bounds-checked against the
    // bucket's precomputed region *here*, not deferred to flush()/finish()
    // -- with a buffer capacity larger than the overflow amount, a flush
    // might not happen until well after the offending call, so checking
    // only at flush time can catch an over-filled bucket arbitrarily late
    // (or never, if the caller never flushes/finishes). Checking on every
    // add() call means the exact call that caused it throws immediately.
    void add(int64_t bucket_id, const DataT* vec) {
        auto& buf = bufs[static_cast<size_t>(bucket_id)];
        if (buf.cursor_records + static_cast<int64_t>(buf.count) >= buf.region_end_records)
            throw std::runtime_error("ReorderedDatasetWriter: bucket " + std::to_string(bucket_id) +
                                     " received more points than its precomputed region size allows "
                                     "(assignments[] and offsets_new/pos_of are inconsistent)");
        std::memcpy(buf.data.data() + buf.count * static_cast<size_t>(D), vec,
                    static_cast<size_t>(D) * sizeof(DataT));
        ++buf.count;
        if (buf.count == buf.capacity_records) flush(bucket_id);
    }

    void finish() {
        for (int64_t b = 0; b < static_cast<int64_t>(bufs.size()); ++b) {
            flush(b);
            if (bufs[static_cast<size_t>(b)].cursor_records !=
                bufs[static_cast<size_t>(b)].region_end_records)
                throw std::runtime_error("ReorderedDatasetWriter: bucket " + std::to_string(b) +
                                         " under-filled its region (fewer points added than its final size)");
        }
        if (::close(fd) != 0)
            throw std::runtime_error("ReorderedDatasetWriter: close failed");
        fd = -1;
    }
};

// ---------------------------------------------------------------------
// ReorderedDatasetBucketReader<DataT>: same count()/read_bucket()
// interface as BucketVectorAccumulator (bucket.cu:767's read_bucket_into /
// count), so build_vector_knn_with_tensorcore (bucket.cu:2874) can consume
// it via the same template parameter without any change to its body. Since
// the reordered file is laid out contiguously per bucket, ids_out is just
// an iota over the bucket's region -- no id list needs to be stored or read.
// ---------------------------------------------------------------------
template <typename DataT>
class ReorderedDatasetBucketReader {
public:
    ReorderedDatasetBucketReader(const std::string& path, int64_t D,
                                 std::vector<uint32_t> offsets_new,
                                 std::vector<int32_t> pos_of)
        : in_(path, std::ios::binary), D_(D),
          offsets_new_(std::move(offsets_new)), pos_of_(std::move(pos_of))
    {
        if (!in_.is_open())
            throw std::runtime_error("ReorderedDatasetBucketReader: cannot open " + path);
        header_bytes_ = 2 * static_cast<std::streamoff>(sizeof(int32_t));
    }

    int64_t count(int64_t bucket_id) const {
        int32_t pos = pos_of_[static_cast<size_t>(bucket_id)];
        return static_cast<int64_t>(offsets_new_[static_cast<size_t>(pos) + 1]) -
               static_cast<int64_t>(offsets_new_[static_cast<size_t>(pos)]);
    }

    void read_bucket(int64_t bucket_id, std::vector<int64_t>& ids_out, std::vector<DataT>& vecs_out) {
        int32_t pos = pos_of_[static_cast<size_t>(bucket_id)];
        int64_t lo = static_cast<int64_t>(offsets_new_[static_cast<size_t>(pos)]);
        int64_t hi = static_cast<int64_t>(offsets_new_[static_cast<size_t>(pos) + 1]);
        int64_t n = hi - lo;

        ids_out.resize(static_cast<size_t>(n));
        std::iota(ids_out.begin(), ids_out.end(), lo);

        vecs_out.resize(static_cast<size_t>(n) * static_cast<size_t>(D_));
        in_.seekg(header_bytes_ + static_cast<std::streamoff>(lo) * D_ * sizeof(DataT));
        in_.read(reinterpret_cast<char*>(vecs_out.data()),
                 static_cast<std::streamsize>(vecs_out.size() * sizeof(DataT)));
        if (!in_.good())
            throw std::runtime_error("ReorderedDatasetBucketReader: read failed for bucket " +
                                     std::to_string(bucket_id));
    }

private:
    std::ifstream in_;
    int64_t D_;
    std::streamoff header_bytes_ = 0;
    std::vector<uint32_t> offsets_new_;
    std::vector<int32_t> pos_of_;
};

}  // namespace reordered_io
