cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# 必须在 project() 之前设置，否则无效（用 conda 工具链，避免系统 gcc/libgomp 版本冲突）
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER /home/lanlu/miniconda3/envs/rapids_raft/bin/nvcc)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER /home/lanlu/miniconda3/envs/rapids_raft/bin/x86_64-conda-linux-gnu-g++)
endif()
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(
    buildCAGRA
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp -Xcompiler=-mavx2 -Xcompiler=-mfma -Xcompiler=-msse4.2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2 -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
# -Xcompiler=-fopenmp: OpenMP::OpenMP_CXX 只作用于 CXX，不加的话 .cu 里 raft 的 host 端
# #pragma omp（NN-Descent / optimize / refine）全部单线程，NN-Descent 建图慢 6 倍
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE --extended-lambda --expt-relaxed-constexpr")

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
# 强制使用本地 raft 源码的 cmake config，禁止 fallback 到 conda
find_package(raft CONFIG REQUIRED
    PATHS /home/lanlu/raft/cpp/build
    NO_DEFAULT_PATH
)


if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_INCLUDE $ENV{CONDA_PREFIX}/include)
    set(CONDA_LIB $ENV{CONDA_PREFIX}/lib)
else()
    message(FATAL_ERROR "CONDA_PREFIX not found. Activate your conda env first.")
endif()

set(RAFT_INCLUDE /home/lanlu/raft/cpp/include)
set(RAFT_LIB /home/lanlu/raft/build/lib)
include_directories(
    /home/lanlu/raft/cpp/include/
    /home/lanlu/miniconda3/envs/rapids_raft/targets/x86_64-linux/include
    /home/lanlu/miniconda3/envs/rapids_raft/include
    /home/lanlu/miniconda3/envs/rapids_raft/include/rapids
    /home/lanlu/miniconda3/envs/rapids_raft/include/rapids/libcudacxx
    /home/lanlu/raft/cpp/build/_deps/nlohmann_json-src/include
    /home/lanlu/raft/cpp/build/_deps/benchmark-src/include
    /home/lanlu/scaleGANN/DiskANN/include
    ${Boost_INCLUDE_DIRS}
    # ${CONDA_INCLUDE}
)

link_directories(
    /home/lanlu/raft/cpp/build/_deps/benchmark-build/src
    # ${CONDA_LIB}
)


set(BUILD_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(BUILD_APP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/apps)


set(BUILD_SOURCES
    build.cu)
list(TRANSFORM BUILD_SOURCES PREPEND ${BUILD_SRC_PATH}/)


add_executable(nnbuild ${BUILD_SOURCES})

target_link_libraries(nnbuild
    PRIVATE
    CUDA::cudart
    OpenMP::OpenMP_CXX
    # raft
    raft::raft
    fmt
)

# CAGRA build with per-stage timing instrumentation
add_executable(nnbuild_timing ${BUILD_SRC_PATH}/build_timing.cu)

target_link_libraries(nnbuild_timing
    PRIVATE
    CUDA::cudart
    OpenMP::OpenMP_CXX
    raft::raft
    fmt
)

set_target_properties(nnbuild_timing PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)

# Exact ann-bench path: cagra::detail::build (NND + sort + optimize fused)
add_executable(nnbuild_exact ${BUILD_SRC_PATH}/build_exact.cu)

target_link_libraries(nnbuild_exact
    PRIVATE
    CUDA::cudart
    CUDA::cublasLt
    CUDA::cusolver
    OpenMP::OpenMP_CXX
    raft::raft
    fmt
)

set_target_properties(nnbuild_exact PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)
# set(COMMON_LIBS
#     cudart
#     dl
#     benchmark
#     pthread
#     fmt
#     mkl_rt
#     m
# )
# foreach(target IN LISTS executeDiskPartition executeMerge searchScaleGANN)
#     target_link_libraries(${target} PRIVATE ${COMMON_LIBS})
# endforeach()

set_target_properties(nnbuild PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
    # BUILD_RPATH ${RAFT_LIB}
)
# CAGRA build-time benchmark (IVF-PQ vs NN-Descent), driven by run_build_bench.sh
add_executable(build_bench ${BUILD_SRC_PATH}/build_bench.cu)

target_link_libraries(build_bench
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::curand
    CUDA::cusolver
    CUDA::cusparse
    OpenMP::OpenMP_CXX
    raft::raft
    fmt
)

set_target_properties(build_bench PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)
