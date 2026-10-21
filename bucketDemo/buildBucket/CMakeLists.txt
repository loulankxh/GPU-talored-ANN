cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

# 依赖：cuvs-bench conda 环境 (libcuvs / libraft / librmm 26.10, nvcc 12.9)
# 编译器：cuvs-build 环境 (gcc 14 —— libcuvs 用 gcc 14 编译, 需要 CXXABI 1.3.15)
set(CUVS_ENV /home/lanlu/miniconda3/envs/cuvs-bench)
set(TOOL_ENV /home/lanlu/miniconda3/envs/cuvs-build)
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER ${CUVS_ENV}/bin/nvcc)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER ${TOOL_ENV}/bin/x86_64-conda-linux-gnu-g++)
endif()
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
if(NOT CMAKE_BUILD_TYPE)  # CMake 4 pre-defines it as "", so NOT DEFINED never fires
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_PREFIX_PATH ${CUVS_ENV} ${TOOL_ENV})

project(
    buildBucket
    LANGUAGES CXX CUDA)

# raft/cuvs 26.10 头文件需要 C++20 (std::source_location)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
# 当前只为本机 GPU 编译以加快编译速度 (V100 sm_70)
set(CMAKE_CUDA_ARCHITECTURES 70)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-gpu-targets -Xcompiler -fopenmp -Xcompiler -fopenmp-simd")

find_package(Boost REQUIRED COMPONENTS program_options CONFIG PATHS ${TOOL_ENV} NO_DEFAULT_PATH)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(cuvs CONFIG REQUIRED PATHS ${CUVS_ENV} NO_DEFAULT_PATH)

set(CONDA_LIB ${CUVS_ENV}/lib)
set(TOOL_LIB ${TOOL_ENV}/lib)

set(BUILD_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(BUILD_SOURCES
    bucket_build.cu)
list(TRANSFORM BUILD_SOURCES PREPEND ${BUILD_SRC_PATH}/)

# legacy builder (raft 24.12 kmeans internals), superseded by bucket2; not built by default
add_executable(bucket EXCLUDE_FROM_ALL ${BUILD_SOURCES})

target_link_libraries(bucket
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    OpenMP::OpenMP_CXX
    cuvs::cuvs
    Boost::program_options
)

set_target_properties(bucket PROPERTIES
    BUILD_RPATH "${CONDA_LIB};${TOOL_LIB}"
)

# bucket2: bucket.cu (with tensor core matmul + per-vector KNN)
add_executable(bucket2 ${BUILD_SRC_PATH}/bucket.cu)

target_link_libraries(bucket2
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    OpenMP::OpenMP_CXX
    cuvs::cuvs
    Boost::program_options
)

set_target_properties(bucket2 PROPERTIES
    BUILD_RPATH "${CONDA_LIB};${TOOL_LIB}"
)

# optimize: standalone tool that runs CAGRA sort_knn_graph + optimize on
# the vector_knn.bin produced by bucket.cu, yielding a pruned CAGRA graph.
add_executable(optimize ${BUILD_SRC_PATH}/optimize.cu)

target_link_libraries(optimize
    PRIVATE
    CUDA::cudart
    OpenMP::OpenMP_CXX
    cuvs::cuvs
    Boost::program_options
)

set_target_properties(optimize PROPERTIES
    BUILD_RPATH "${CONDA_LIB};${TOOL_LIB}"
)

# optimize_chunked: chunked reverse-graph + merge for huge graphs that don't
# fit on a single GPU. Implements 5 methods (A: kernel-filter, B: pre-shuffle,
# C: bucket-aligned + overflow, D: 3-slot sliding window, E: per-chunk disk-
# backed scatter buffer, rank-priority preserving, streams fwd/output graph
# from/to disk instead of requiring them host-resident) selectable via --method.
add_executable(optimize_chunked ${BUILD_SRC_PATH}/optimize_chunked.cu)

target_link_libraries(optimize_chunked
    PRIVATE
    CUDA::cudart
    OpenMP::OpenMP_CXX
    Boost::program_options
)

set_target_properties(optimize_chunked PROPERTIES
    BUILD_RPATH "${CONDA_LIB};${TOOL_LIB}"
)

# reorder: CPU-only tool that reorders dataset + vector_knn.bin by bucket
# layout (consecutive new IDs per bucket), enabling optimize_chunked -m C.
add_executable(reorder ${BUILD_SRC_PATH}/reorder.cpp)

target_link_libraries(reorder
    PRIVATE
    OpenMP::OpenMP_CXX
    Boost::program_options
)

set_target_properties(reorder PROPERTIES
    BUILD_RPATH "${CONDA_LIB};${TOOL_LIB}"
)

# prune_windowed: CPU-only (OpenMP) approximate CAGRA prune for graphs too
# large for raft's own sort_knn_graph+optimize() to hold resident. Windows
# the "does my neighbor's neighbor connect back to me" lookup through
# bucket_order.hpp's Belady bucket cache instead of requiring the whole
# candidate graph resident -- see prune_windowed.cpp's header comment for
# why this is CPU-only rather than a CUDA kernel. Feeds optimize_chunked's
# forward-graph input (any method, but paired with --method E and the same
# --bucket-offsets is the intended large-scale path).
add_executable(prune_windowed ${BUILD_SRC_PATH}/prune_windowed.cpp ${BUILD_SRC_PATH}/prune_windowed_gpu.cu)

target_link_libraries(prune_windowed
    PRIVATE
    CUDA::cudart
    OpenMP::OpenMP_CXX
    Boost::program_options
)

set_target_properties(prune_windowed PROPERTIES
    BUILD_RPATH "${CONDA_LIB};${TOOL_LIB}"
)

# translate_ids: CPU-only tool that translates a final graph/index from the
# reordered ("new") id space bucket2/optimize_chunked/prune_windowed all work
# in back to the original dataset's id order, using inverse_perm.bin (from
# bucket2's inline reorder) -- e.g. for comparing against ground truth files
# that reference original ids.
add_executable(translate_ids ${BUILD_SRC_PATH}/translate_ids.cpp)

target_link_libraries(translate_ids
    PRIVATE
    OpenMP::OpenMP_CXX
    Boost::program_options
)

set_target_properties(translate_ids PROPERTIES
    BUILD_RPATH "${CONDA_LIB};${TOOL_LIB}"
)
