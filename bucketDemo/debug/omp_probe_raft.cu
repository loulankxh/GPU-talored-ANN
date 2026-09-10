// Same probe pattern as omp_probe.cpp / omp_probe_cuda.cu, but built through
// the project's own CMake target so include paths, RPATH and link
// resolution exactly match bucket2. cudart/cublas were already ruled out
// (omp_probe_cuda.cu: all 11). This isolates whether linking raft::raft
// (and whatever it transitively pulls in, e.g. RMM's allocator) is what
// breaks num_threads() -- independent of calling any RAFT algorithm.

#include <omp.h>
#include <atomic>
#include <cstdio>
#include <raft/core/resources.hpp>

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
    probe("A: before constructing raft::resources");
    raft::resources res;
    probe("B: after constructing raft::resources");
    return 0;
}
