// load.hpp
// C++ helpers to read BIGANN-style data files (.fbin/.bin/.u8bin/.i8bin/.ibin)
// and write index in the SAME format as Python's: np.save("neighbors.npy", int64[N,M])
//
// Python format you used:
//   neighbors: shape (N, M), dtype int64, padding -1
//   np.save(path, neighbors)
//
// This file provides:
//   - read_fbin_f32 / read_u8bin_to_f32 / read_ibin_i32
//   - write_npy_int64_2d(path, data, N, M)    // writes .npy v1.0, little-endian int64
//
// Build note: header-only; just include in your .cpp and compile with -std=c++17.

#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>

struct Neighbor {
    int id;
    float dist;
    bool is_new;
};

namespace load {

// ---------- small utils ----------
inline void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

inline std::pair<int32_t,int32_t> read_bigann_header(std::ifstream& in) {
    int32_t N=0, D=0;
    in.read(reinterpret_cast<char*>(&N), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&D), sizeof(int32_t));
    require(in.good(), "Failed to read BIGANN header (N,D).");
    require(N > 0 && D > 0, "Invalid BIGANN header (N<=0 or D<=0).");
    return {N, D};
}

// ---------- data readers ----------
// .fbin / .bin: header int32 N,D then N*D float32
inline void read_fbin_f32(const std::string& path,
                          std::vector<float>& out,
                          int32_t& N, int32_t& D) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);

    auto [n, d] = read_bigann_header(in);
    N = n; D = d;

    const uint64_t cnt = static_cast<uint64_t>(N) * static_cast<uint64_t>(D);
    out.resize(cnt);

    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(cnt * sizeof(float)));
    require(in.good(), "Failed to read float payload from: " + path);
}

// .u8bin / .i8bin: header int32 N,D then N*D uint8, convert to float32
inline void read_u8bin_to_f32(const std::string& path,
                              std::vector<float>& out,
                              int32_t& N, int32_t& D) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);

    auto [n, d] = read_bigann_header(in);
    N = n; D = d;

    const uint64_t cnt = static_cast<uint64_t>(N) * static_cast<uint64_t>(D);

    std::vector<uint8_t> tmp(cnt);
    in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(cnt));
    require(in.good(), "Failed to read uint8 payload from: " + path);

    out.resize(cnt);
    for (uint64_t i = 0; i < cnt; ++i) out[i] = static_cast<float>(tmp[i]);
}

// .ibin: header int32 N,D then N*D int32
inline void read_ibin_i32(const std::string& path,
                          std::vector<int32_t>& out,
                          int32_t& N, int32_t& D) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);

    auto [n, d] = read_bigann_header(in);
    N = n; D = d;

    const uint64_t cnt = static_cast<uint64_t>(N) * static_cast<uint64_t>(D);
    out.resize(cnt);

    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(cnt * sizeof(int32_t)));
    require(in.good(), "Failed to read int32 payload from: " + path);
}

// ---------- partial readers (sampled rows only) ----------

// 只读 header，返回 (N, D)，不读 payload
inline std::pair<int32_t,int32_t> read_fbin_header(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);
    return read_bigann_header(in);
}

// 按 sorted 索引从 .fbin 文件中只读取指定行，避免读取整个数据集。
// indices 必须已排序（升序）；输出 out 按 indices 原始顺序排列。
// 对于 SSD / NVMe，排序后顺序读可以最大化吞吐。
inline void read_fbin_sampled(const std::string& path,
                              const std::vector<int64_t>& sorted_indices,
                              std::vector<float>& out,
                              int32_t& N, int32_t& D) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);

    auto [n, d] = read_bigann_header(in);
    N = n; D = d;

    const int64_t n_samples = static_cast<int64_t>(sorted_indices.size());
    out.resize(static_cast<size_t>(n_samples) * d);
    const std::streamsize row_bytes = static_cast<std::streamsize>(d) * sizeof(float);
    constexpr std::streamoff header_bytes = 2 * sizeof(int32_t);  // 8 bytes

    for (int64_t i = 0; i < n_samples; ++i) {
        std::streamoff offset = header_bytes +
            static_cast<std::streamoff>(sorted_indices[i]) * static_cast<std::streamoff>(d) * sizeof(float);
        in.seekg(offset);
        in.read(reinterpret_cast<char*>(out.data() + i * d), row_bytes);
        require(in.good(), "Failed to read sampled row " + std::to_string(sorted_indices[i]));
    }
}

// ---------- chunked sequential readers (contiguous row range) ----------
// 按连续行区间 [start_row, start_row+num_rows) 读取数据块，用于分块（streaming）
// 处理超大数据集，避免一次性把整个文件读进内存。

inline void read_fbin_chunk(const std::string& path,
                            int64_t start_row, int64_t num_rows,
                            std::vector<float>& out,
                            int32_t& D) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);

    auto [n, d] = read_bigann_header(in);
    D = d;
    require(start_row >= 0 && start_row + num_rows <= n,
            "read_fbin_chunk: row range out of bounds");

    constexpr std::streamoff header_bytes = 2 * sizeof(int32_t);
    std::streamoff offset = header_bytes +
        static_cast<std::streamoff>(start_row) * d * sizeof(float);
    in.seekg(offset);

    const uint64_t cnt = static_cast<uint64_t>(num_rows) * d;
    out.resize(cnt);
    in.read(reinterpret_cast<char*>(out.data()),
            static_cast<std::streamsize>(cnt * sizeof(float)));
    require(in.good(), "Failed to read chunk [" + std::to_string(start_row) + ", " +
                       std::to_string(start_row + num_rows) + ") from: " + path);
}

// .u8bin / .i8bin 版本：读原始 uint8 数据块，转换为 float32
inline void read_u8bin_chunk_to_f32(const std::string& path,
                                    int64_t start_row, int64_t num_rows,
                                    std::vector<float>& out,
                                    int32_t& D) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);

    auto [n, d] = read_bigann_header(in);
    D = d;
    require(start_row >= 0 && start_row + num_rows <= n,
            "read_u8bin_chunk_to_f32: row range out of bounds");

    constexpr std::streamoff header_bytes = 2 * sizeof(int32_t);
    std::streamoff offset = header_bytes +
        static_cast<std::streamoff>(start_row) * d * sizeof(uint8_t);
    in.seekg(offset);

    const uint64_t cnt = static_cast<uint64_t>(num_rows) * d;
    std::vector<uint8_t> tmp(cnt);
    in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(cnt));
    require(in.good(), "Failed to read chunk from: " + path);

    out.resize(cnt);
    for (uint64_t i = 0; i < cnt; ++i) out[i] = static_cast<float>(tmp[i]);
}

// .ibin 版本：读原始 int32 数据块，转换为 float32
inline void read_ibin_chunk_to_f32(const std::string& path,
                                   int64_t start_row, int64_t num_rows,
                                   std::vector<float>& out,
                                   int32_t& D) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);

    auto [n, d] = read_bigann_header(in);
    D = d;
    require(start_row >= 0 && start_row + num_rows <= n,
            "read_ibin_chunk_to_f32: row range out of bounds");

    constexpr std::streamoff header_bytes = 2 * sizeof(int32_t);
    std::streamoff offset = header_bytes +
        static_cast<std::streamoff>(start_row) * d * sizeof(int32_t);
    in.seekg(offset);

    const uint64_t cnt = static_cast<uint64_t>(num_rows) * d;
    std::vector<int32_t> tmp(cnt);
    in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(cnt * sizeof(int32_t)));
    require(in.good(), "Failed to read chunk from: " + path);

    out.resize(cnt);
    for (uint64_t i = 0; i < cnt; ++i) out[i] = static_cast<float>(tmp[i]);
}

// ---------- Templated raw readers (no type conversion) ----------
//
// 读取 BIGANN-style 文件 (header int32 N,D + payload N*D 元素) 而不做类型转换。
// 用于模板化 pipeline，不必把 uint8 数据集膨胀到 float32。
//
// 调用示例：
//   std::vector<uint8_t> X; int32_t N, D;
//   load::read_bigann_raw<uint8_t>("data.u8bin", X, N, D);

template <typename T>
inline void read_bigann_raw(const std::string& path,
                            std::vector<T>& out,
                            int32_t& N, int32_t& D) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);

    auto [n, d] = read_bigann_header(in);
    N = n; D = d;

    const uint64_t cnt = static_cast<uint64_t>(N) * static_cast<uint64_t>(D);
    out.resize(cnt);
    in.read(reinterpret_cast<char*>(out.data()),
            static_cast<std::streamsize>(cnt * sizeof(T)));
    require(in.good(), "Failed to read raw payload (" +
                       std::to_string(sizeof(T)) + "B/elem) from: " + path);
}

// 按 sorted 索引从 BIGANN 文件中只读取指定行（不做类型转换，element type = T）。
// indices 必须已排序（升序）；输出按 indices 原始顺序排列。
template <typename T>
inline void read_bigann_raw_sampled(const std::string& path,
                                    const std::vector<int64_t>& sorted_indices,
                                    std::vector<T>& out,
                                    int32_t& N, int32_t& D) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);

    auto [n, d] = read_bigann_header(in);
    N = n; D = d;

    const int64_t n_samples = static_cast<int64_t>(sorted_indices.size());
    out.resize(static_cast<size_t>(n_samples) * d);
    const std::streamsize row_bytes =
        static_cast<std::streamsize>(d) * sizeof(T);
    constexpr std::streamoff header_bytes = 2 * sizeof(int32_t);

    for (int64_t i = 0; i < n_samples; ++i) {
        std::streamoff offset = header_bytes +
            static_cast<std::streamoff>(sorted_indices[i])
            * static_cast<std::streamoff>(d) * sizeof(T);
        in.seekg(offset);
        in.read(reinterpret_cast<char*>(out.data() + i * d), row_bytes);
        require(in.good(), "Failed to read sampled row "
                           + std::to_string(sorted_indices[i]));
    }
}

// 从 BIGANN 文件里读一段连续的行 [start_row, start_row+num_rows)（不做类型
// 转换，element type = T）。一次 seek + 一次顺序 bulk read，用于"这批 id 基本
// 连续"的场景 (比如按升序处理的 batch)，比逐行 seek 快得多。out 按行序排列，
// 调用方自己按 (global_id - start_row) 换算行内偏移。
template <typename T>
inline void read_bigann_raw_range(const std::string& path,
                                  int64_t start_row, int64_t num_rows,
                                  std::vector<T>& out,
                                  int32_t& N, int32_t& D) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);

    auto [n, d] = read_bigann_header(in);
    N = n; D = d;
    require(start_row >= 0 && start_row + num_rows <= n,
            "read_bigann_raw_range: row range out of bounds");

    constexpr std::streamoff header_bytes = 2 * sizeof(int32_t);
    std::streamoff offset = header_bytes +
        static_cast<std::streamoff>(start_row) * static_cast<std::streamoff>(d) * sizeof(T);
    in.seekg(offset);

    const uint64_t cnt = static_cast<uint64_t>(num_rows) * static_cast<uint64_t>(d);
    out.resize(cnt);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(cnt * sizeof(T)));
    require(in.good(), "read_bigann_raw_range: failed to read rows ["
                       + std::to_string(start_row) + ", " + std::to_string(start_row + num_rows) + ")");
}

// 推断 BIGANN 文件扩展名对应的元素类型大小（字节数）。
inline size_t element_size_for_ext(const std::string& ext) {
    if (ext == ".fbin" || ext == ".bin")    return sizeof(float);
    if (ext == ".u8bin")                    return sizeof(uint8_t);
    if (ext == ".i8bin")                    return sizeof(int8_t);
    if (ext == ".ibin")                     return sizeof(int32_t);
    if (ext == ".ubin")                     return sizeof(uint32_t);
    if (ext == ".hbin")                     return sizeof(uint16_t);  // fp16, treated as raw
    require(false, "Unsupported BIGANN extension: " + ext);
    return 0;
}

// ---------- NPY writer (matches np.save for a contiguous int64 2D array) ----------
// Writes .npy v1.0, little-endian int64, C-order, shape (N,M).
//
// Equivalent in Python:
//   np.save(path, arr.astype(np.int64, copy=False))
inline void write_npy_int64_2d(const std::string& path,
                               const int64_t* data,
                               int64_t N, int64_t M) {
    require(N > 0 && M > 0, "write_npy_int64_2d: N and M must be > 0");
    std::ofstream out(path, std::ios::binary);
    require(out.is_open(), "Cannot open for write: " + path);

    // magic + version
    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);
    const uint8_t ver_major = 1;
    const uint8_t ver_minor = 0;
    out.put(static_cast<char>(ver_major));
    out.put(static_cast<char>(ver_minor));

    // build header dict (must end with newline, and header padding to 16-byte alignment of preamble+header)
    // dtype: little-endian int64 = "<i8"
    // fortran_order: False
    // shape: (N, M)
    std::string header = "{'descr': '<i8', 'fortran_order': False, 'shape': (";
    header += std::to_string(N);
    header += ", ";
    header += std::to_string(M);
    header += "), }";

    // header must be padded with spaces to reach alignment, and end with '\n'
    // For v1.0: header length is stored as uint16 little-endian
    // Total: magic(6)+ver(2)+hlen(2)+header = preamble(10)+header
    const size_t preamble = 10;
    // +1 for '\n'
    size_t header_len = header.size() + 1;
    // pad so that (preamble + header_len) % 16 == 0
    size_t pad = (16 - ((preamble + header_len) % 16)) % 16;
    header.append(pad, ' ');
    header.push_back('\n');
    header_len = header.size();

    require(header_len <= 65535, "NPY v1.0 header too large.");

    // write header length (uint16 LE)
    uint16_t hlen = static_cast<uint16_t>(header_len);
    out.write(reinterpret_cast<const char*>(&hlen), sizeof(uint16_t));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    // write data payload (C-order contiguous)
    const uint64_t cnt = static_cast<uint64_t>(N) * static_cast<uint64_t>(M);
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(cnt * sizeof(int64_t)));
    require(out.good(), "Failed while writing npy payload.");
}

// Convenience overload for std::vector<int64_t> storing row-major (N*M)
inline void write_npy_int64_2d(const std::string& path,
                               const std::vector<int64_t>& data,
                               int64_t N, int64_t M) {
    require(static_cast<int64_t>(data.size()) == N * M, "Data size != N*M");
    write_npy_int64_2d(path, data.data(), N, M);
}

void save_neighbors_npy(const std::string& path,
                        const std::vector<std::vector<Neighbor>>& graph,
                        int K) {
    int64_t N = static_cast<int64_t>(graph.size());
    int64_t M = static_cast<int64_t>(K);

    std::vector<int64_t> buf(N * M, -1);  // 默认填 -1

    for (int64_t i = 0; i < N; ++i) {
        const auto& neighs = graph[i];

        // 拷贝一份排序（防止原 graph 顺序被改）
        std::vector<Neighbor> tmp = neighs;
        std::sort(tmp.begin(), tmp.end(),
                  [](const Neighbor& a, const Neighbor& b) {
                      return a.dist < b.dist;
                  });

        int limit = std::min<int>(K, tmp.size());
        for (int j = 0; j < limit; ++j) {
            buf[i * M + j] = static_cast<int64_t>(tmp[j].id);
        }
    }

    write_npy_int64_2d(path, buf.data(), N, M);
}


inline void write_npy_f32_2d(const std::string& path,
                             const float* data,
                             int64_t N, int64_t M) {
    require(N > 0 && M > 0, "write_npy_f32_2d: N and M must be > 0");
    std::ofstream out(path, std::ios::binary);
    require(out.is_open(), "Cannot open for write: " + path);

    // magic + version
    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);
    const uint8_t ver_major = 1;
    const uint8_t ver_minor = 0;
    out.put(static_cast<char>(ver_major));
    out.put(static_cast<char>(ver_minor));

    // dtype: little-endian float32 = "<f4"
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    header += std::to_string(N);
    header += ", ";
    header += std::to_string(M);
    header += "), }";

    const size_t preamble = 10;
    size_t header_len = header.size() + 1;
    size_t pad = (16 - ((preamble + header_len) % 16)) % 16;
    header.append(pad, ' ');
    header.push_back('\n');
    header_len = header.size();

    require(header_len <= 65535, "NPY v1.0 header too large.");

    uint16_t hlen = static_cast<uint16_t>(header_len);
    out.write(reinterpret_cast<const char*>(&hlen), sizeof(uint16_t));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    const uint64_t cnt = static_cast<uint64_t>(N) * static_cast<uint64_t>(M);
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(cnt * sizeof(float)));

    require(out.good(), "Failed while writing npy payload.");
}

// ---------- Row-wise (out-of-core) NPY writers ----------
// 用于结果按任意顺序算出来（比如按 bucket 顺序），但要按行号（global id）写回
// 文件的场景：先建好 header 定好文件形状，可选地分块把整份文件填成 sentinel，
// 再在计算过程中随时按行号 seek 写入 —— 不需要把 N*M 的结果整个攒在内存里。

inline size_t create_npy_int64_2d(const std::string& path, int64_t N, int64_t M) {
    require(N > 0 && M > 0, "create_npy_int64_2d: N and M must be > 0");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    require(out.is_open(), "Cannot open for write: " + path);

    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);
    out.put(static_cast<char>(1));
    out.put(static_cast<char>(0));

    std::string header = "{'descr': '<i8', 'fortran_order': False, 'shape': (";
    header += std::to_string(N) + ", " + std::to_string(M) + "), }";
    const size_t preamble = 10;
    size_t header_len = header.size() + 1;
    size_t pad = (16 - ((preamble + header_len) % 16)) % 16;
    header.append(pad, ' ');
    header.push_back('\n');
    header_len = header.size();
    require(header_len <= 65535, "NPY v1.0 header too large.");

    uint16_t hlen = static_cast<uint16_t>(header_len);
    out.write(reinterpret_cast<const char*>(&hlen), sizeof(uint16_t));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    require(out.good(), "Failed while writing npy header: " + path);
    return preamble + header_len;
}

inline size_t create_npy_f32_2d(const std::string& path, int64_t N, int64_t M) {
    require(N > 0 && M > 0, "create_npy_f32_2d: N and M must be > 0");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    require(out.is_open(), "Cannot open for write: " + path);

    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);
    out.put(static_cast<char>(1));
    out.put(static_cast<char>(0));

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    header += std::to_string(N) + ", " + std::to_string(M) + "), }";
    const size_t preamble = 10;
    size_t header_len = header.size() + 1;
    size_t pad = (16 - ((preamble + header_len) % 16)) % 16;
    header.append(pad, ' ');
    header.push_back('\n');
    header_len = header.size();
    require(header_len <= 65535, "NPY v1.0 header too large.");

    uint16_t hlen = static_cast<uint16_t>(header_len);
    out.write(reinterpret_cast<const char*>(&hlen), sizeof(uint16_t));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    require(out.good(), "Failed while writing npy header: " + path);
    return preamble + header_len;
}

// 分块顺序把整份文件填成 sentinel（-1）。理论上 Step 4 会给每一行都写一次，
// 这一步只是防御性的：如果某个点因为某种原因没有被写到（不应该发生，因为
// 每个点必然会被分配到某个桶），它的行至少是有意义的 -1 padding，而不是
// 文件空洞被操作系统清零后看起来像"邻居 0，距离 0"的脏数据。
inline void prefill_npy_int64_2d(std::fstream& f, size_t header_bytes,
                                 int64_t N, int64_t M, size_t chunk_bytes_budget) {
    int64_t chunk_rows = std::max<int64_t>(
        1, static_cast<int64_t>(chunk_bytes_budget / (static_cast<size_t>(M) * sizeof(int64_t))));
    chunk_rows = std::min(chunk_rows, N);
    std::vector<int64_t> buf(static_cast<size_t>(chunk_rows) * M, -1);

    f.seekp(static_cast<std::streamoff>(header_bytes));
    for (int64_t start = 0; start < N; start += chunk_rows) {
        int64_t cur = std::min(chunk_rows, N - start);
        f.write(reinterpret_cast<const char*>(buf.data()),
               static_cast<std::streamsize>(cur * M * sizeof(int64_t)));
        require(f.good(), "Failed prefilling npy payload");
    }
}

inline void prefill_npy_f32_2d(std::fstream& f, size_t header_bytes,
                               int64_t N, int64_t M, size_t chunk_bytes_budget) {
    int64_t chunk_rows = std::max<int64_t>(
        1, static_cast<int64_t>(chunk_bytes_budget / (static_cast<size_t>(M) * sizeof(float))));
    chunk_rows = std::min(chunk_rows, N);
    std::vector<float> buf(static_cast<size_t>(chunk_rows) * M,
                           std::numeric_limits<float>::infinity());

    f.seekp(static_cast<std::streamoff>(header_bytes));
    for (int64_t start = 0; start < N; start += chunk_rows) {
        int64_t cur = std::min(chunk_rows, N - start);
        f.write(reinterpret_cast<const char*>(buf.data()),
               static_cast<std::streamsize>(cur * M * sizeof(float)));
        require(f.good(), "Failed prefilling npy payload");
    }
}

inline void write_npy_row_int64(std::fstream& f, size_t header_bytes,
                                int64_t row, int64_t M, const int64_t* data) {
    f.seekp(static_cast<std::streamoff>(header_bytes) +
           static_cast<std::streamoff>(row) * static_cast<std::streamoff>(M) * sizeof(int64_t));
    f.write(reinterpret_cast<const char*>(data),
           static_cast<std::streamsize>(M * sizeof(int64_t)));
    require(f.good(), "Failed writing npy row " + std::to_string(row));
}

inline void write_npy_row_f32(std::fstream& f, size_t header_bytes,
                              int64_t row, int64_t M, const float* data) {
    f.seekp(static_cast<std::streamoff>(header_bytes) +
           static_cast<std::streamoff>(row) * static_cast<std::streamoff>(M) * sizeof(float));
    f.write(reinterpret_cast<const char*>(data),
           static_cast<std::streamsize>(M * sizeof(float)));
    require(f.good(), "Failed writing npy row " + std::to_string(row));
}

void save_distances_npy(const std::string& path,
                        const std::vector<std::vector<Neighbor>>& graph,
                        int K) {
    int64_t N = static_cast<int64_t>(graph.size());
    int64_t M = static_cast<int64_t>(K);

    std::vector<float> buf(N * M, std::numeric_limits<float>::infinity());

    for (int64_t i = 0; i < N; ++i) {
        std::vector<Neighbor> tmp = graph[i];
        std::sort(tmp.begin(), tmp.end(),
                  [](const Neighbor& a, const Neighbor& b) {
                      return a.dist < b.dist;
                  });

        int limit = std::min<int>(K, tmp.size());
        for (int j = 0; j < limit; ++j) {
            buf[i * M + j] = tmp[j].dist;
        }
    }

    write_npy_f32_2d(path, buf.data(), N, M);
}


} // namespace load
