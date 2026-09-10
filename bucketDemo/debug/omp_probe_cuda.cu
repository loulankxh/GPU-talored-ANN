// Same probe as omp_probe.cpp, but compiled with nvcc and linked against
// cudart + cublas (mirroring bucket2's exact link set), to isolate whether
// merely being an nvcc-compiled binary that links CUDA runtime/cuBLAS is
// enough to break OpenMP's num_threads() clause -- independent of whether
// any CUDA API is actually called.
//
// Build (adjust CONDA path if different):
//   CONDA=/home/lanlu/miniconda3/envs/rapids_raft
//   $CONDA/bin/nvcc -std=c++17 --extended-lambda --expt-relaxed-constexpr \
//     -ccbin $CONDA/bin/x86_64-conda-linux-gnu-g++ \
//     -gencode arch=compute_70,code=sm_70 \
//     -Xcompiler -fopenmp \
//     -o omp_probe_cuda omp_probe_cuda.cu \
//     -L$CONDA/lib -Wl,-rpath,$CONDA/lib \
//     -lcudart -lcublas -lgomp
//
// Run:
//   ./omp_probe_cuda

#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <atomic>
#include <cstdio>

static void probe(const char* label) {
    std::atomic<int> actual{0};
    #pragma omp parallel num_threads(11)
    {
        #pragma omp single
        { actual = omp_get_num_threads(); }
    }
    printf("[%s] omp_get_max_threads()=%d omp_get_num_procs()=%d actual_num_threads(requested 11)=%d\n",
           label, omp_get_max_threads(), omp_get_num_procs(), actual.load());
}

int main() {
    probe("A: before any CUDA API call (cudart+cublas linked but unused)");

    cudaError_t rc = cudaFree(0);  // common idiom to force CUDA context creation
    printf("  cudaFree(0) rc=%d (%s)\n", rc, cudaGetErrorString(rc));
    probe("B: after cudaFree(0) forces CUDA context init");

    cublasHandle_t handle;
    cublasStatus_t crc = cublasCreate(&handle);
    printf("  cublasCreate rc=%d\n", crc);
    probe("C: after cublasCreate");

    cublasDestroy(handle);
    return 0;
}
