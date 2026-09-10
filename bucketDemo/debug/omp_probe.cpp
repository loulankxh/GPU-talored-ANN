// Standalone probe for the "actual_num_threads=1" issue in bucket.cu's merge_iteration.
// No CUDA/RAFT dependency -- compiles in ~1s, isolates the root cause without touching bucket2.
//
// Build:  g++ -std=c++17 -fopenmp -O2 -o omp_probe omp_probe.cpp -lpthread
// Run:    ./omp_probe
//
// Run it in the EXACT same shell/session (same env, same cgroup/container) you use to launch bucket2.

#include <omp.h>
#include <pthread.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <thread>
#include <vector>

static void run_omp_probe(const char* label, bool use_clause) {
    printf("[%s] omp_get_max_threads()=%d omp_get_num_procs()=%d\n",
           label, omp_get_max_threads(), omp_get_num_procs());
    std::atomic<int> actual{0};
    if (use_clause) {
        #pragma omp parallel num_threads(11)
        {
            #pragma omp single
            { actual = omp_get_num_threads(); }
            #pragma omp critical
            { printf("[%s]   thread %d/%d alive\n", label, omp_get_thread_num(), omp_get_num_threads()); }
        }
    } else {
        #pragma omp parallel
        {
            #pragma omp single
            { actual = omp_get_num_threads(); }
            #pragma omp critical
            { printf("[%s]   thread %d/%d alive\n", label, omp_get_thread_num(), omp_get_num_threads()); }
        }
    }
    printf("[%s] actual_num_threads = %d\n\n", label, actual.load());
}

static int raw_pthread_create_test(int n) {
    std::vector<pthread_t> tids(static_cast<size_t>(n));
    auto fn = [](void*) -> void* {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return nullptr;
    };
    int ok = 0;
    for (int i = 0; i < n; ++i) {
        int rc = pthread_create(&tids[static_cast<size_t>(i)], nullptr, fn, nullptr);
        if (rc == 0) {
            ++ok;
        } else {
            printf("  pthread_create #%d failed: rc=%d (%s)\n", i, rc, strerror(rc));
        }
    }
    for (int i = 0; i < ok; ++i) pthread_join(tids[static_cast<size_t>(i)], nullptr);
    return ok;
}

int main() {
    printf("=== Relevant environment variables ===\n");
    const char* names[] = {"OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_DYNAMIC",
                            "OMP_NESTED", "OMP_MAX_ACTIVE_LEVELS",
                            "MKL_NUM_THREADS", "MKL_DYNAMIC", "KMP_NUM_THREADS"};
    for (const char* n : names) {
        const char* v = std::getenv(n);
        printf("  %-22s = %s\n", n, v ? v : "(unset)");
    }
    printf("\n");

    printf("=== Test 1: omp parallel num_threads(11), called from main thread ===\n");
    run_omp_probe("main-thread", /*use_clause=*/true);

    printf("=== Test 2: same, but called from a std::async-launched thread (mirrors bucket2's pipelining) ===\n");
    {
        auto fut = std::async(std::launch::async, [] { run_omp_probe("async-thread", true); });
        fut.get();
    }

    printf("=== Test 3: omp parallel num_threads(11), nested inside another active omp parallel region ===\n");
    #pragma omp parallel num_threads(2)
    {
        #pragma omp single
        { run_omp_probe("nested-inside-omp", true); }
    }

    printf("=== Test 4: raw pthread_create x11, no OpenMP involved (isolates ulimit -u / cgroup pids.max) ===\n");
    {
        int ok = raw_pthread_create_test(11);
        printf("  successfully created %d/11 raw pthreads\n\n", ok);
    }

    printf("=== Test 5: omp_set_num_threads(11) then omp parallel WITHOUT num_threads clause ===\n");
    omp_set_num_threads(11);
    run_omp_probe("after-omp_set_num_threads", /*use_clause=*/false);

    printf("Done. If Test 1/2 report actual_num_threads=1 while Test 4 creates 11/11 raw pthreads fine,\n"
           "the cause is an OpenMP-level cap (OMP_THREAD_LIMIT or something setting libgomp's ICV),\n"
           "not a system-level thread/process limit.\n"
           "If Test 4 also fails to create 11 threads, the cause is ulimit -u / cgroup pids.max.\n");
    return 0;
}
