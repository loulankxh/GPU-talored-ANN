#!/usr/bin/env bash
# One-shot environment diagnostic for the "actual_num_threads=1" OpenMP issue.
# Run this in the exact same shell/session you use to launch bucket2.

echo "===== OpenMP-related environment variables ====="
env | grep -i omp
echo "(if nothing printed above, none of OMP_NUM_THREADS / OMP_THREAD_LIMIT / OMP_DYNAMIC / OMP_NESTED / OMP_MAX_ACTIVE_LEVELS are set)"
echo

echo "===== MKL / other threading env vars (can also throttle OpenMP via shared runtime) ====="
env | grep -iE "mkl_num_threads|mkl_dynamic|kmp_|numexpr_num_threads"
echo

echo "===== CPU count ====="
echo "nproc (affinity-aware): $(nproc)"
echo "nproc --all (ignores affinity): $(nproc --all)"
echo

echo "===== cgroup CPU quota ====="
if [ -f /sys/fs/cgroup/cpu.max ]; then
    echo "cgroup v2 cpu.max: $(cat /sys/fs/cgroup/cpu.max)"
else
    echo "cgroup v1 cfs_quota_us: $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)"
    echo "cgroup v1 cfs_period_us: $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null)"
fi
echo

echo "===== cgroup cpuset ====="
cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null
cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null
echo

echo "===== thread/process limits (libgomp silently falls back to 1 thread if pthread_create fails) ====="
echo "ulimit -u (max user processes/threads): $(ulimit -u)"
if [ -f /sys/fs/cgroup/pids.max ]; then
    echo "cgroup v2 pids.max: $(cat /sys/fs/cgroup/pids.max)"
    echo "cgroup v2 pids.current: $(cat /sys/fs/cgroup/pids.current)"
else
    echo "cgroup v1 pids.max: $(cat /sys/fs/cgroup/pids/pids.max 2>/dev/null)"
    echo "cgroup v1 pids.current: $(cat /sys/fs/cgroup/pids/pids.current 2>/dev/null)"
fi
echo

echo "===== conda env activate.d scripts (rapids_raft env sometimes auto-sets OMP_NUM_THREADS/OMP_THREAD_LIMIT) ====="
CONDA_ENV_DIR="$(dirname "$(dirname "$(which nvcc 2>/dev/null || which python 2>/dev/null)")")"
echo "Detected conda env dir guess: $CONDA_ENV_DIR"
for d in "$CONDA_ENV_DIR/etc/conda/activate.d" "/home/lanlu/miniconda3/envs/rapids_raft/etc/conda/activate.d"; do
    if [ -d "$d" ]; then
        echo "--- $d ---"
        ls "$d"
        grep -riH "omp\|mkl_num_threads\|kmp_" "$d"/*.sh 2>/dev/null
    fi
done
echo

echo "===== done ====="
