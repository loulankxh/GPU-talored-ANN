#pragma once
// ============================================================================
// spot_sched.hpp — spot instance 中断 / 恢复 (本机模拟) 的共享状态客户端 + 段调度器
//
// 设计见 bucketDemo/SPOT_DESIGN.md。要点:
//   * host CPU 常驻、不会被抢占；Step 4 / Step 6 / prune 的 GPU worker 是 spot 实例。
//   * 工作单元按全局编号 [0, n_units) 排好 (Step 4: non-centroid 行; Step 6: bucket order 位置;
//     prune: bucket 位置)。每段 Segment{begin, cursor, end, owner}: [begin, cursor) 已提交，
//     [cursor, end) 未做。worker 交回一个完整单元 → host 推进 cursor ("先提交后推进")，
//     所以 cursor 就是"这之前全部做完"的 flag；多 GPU = 多个 flag。
//   * 抢占: 段的 owner 被回收, [cursor, end) 变成空闲段；空闲 worker / 新加入的实例去领，
//     没有空闲段就从剩余最多的段尾部偷一半 (切点 ≥ 原 owner 已发出的单元，不浪费在途工作)。
//   * 实例状态来自外部模拟器 spot_sim.py (trace 回放) 写的 mmap 共享文件 + SIGUSR1 唤醒；
//     抢占时刻之后到达的提交一律拒收 (模拟"实例已经没了")，所以通知只是优化，不是正确性前提。
//   * 未设 BUCKET2_SPOT_DIR → Client 关闭: 所有实例永远 Up，调度器退化为静态切分 (行为同非 spot)。
//
// 只用 C++17 + POSIX，不依赖 CUDA (bucket.cu / prune_windowed / 单元测试共用)。
// ============================================================================
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace spot {

// ---------------------------------------------------------------------------
// 共享状态文件布局 (spot_sim.py 写，这里只读)。小端，seqlock: 写者先把 seq 加成奇数，
// 改完实例记录再加成偶数；读者读到奇数或前后 seq 不一致就重读。
// ---------------------------------------------------------------------------
constexpr uint32_t kMagic   = 0x544F5053u;   // "SPOT"
constexpr uint32_t kVersion = 1;
enum : uint32_t { kUp = 0, kNotice = 1, kDown = 2 };

#pragma pack(push, 1)
struct FileHeader {
    uint32_t magic, version, n_inst, pad;
    uint64_t seq;
    int64_t  t0_ns;      // 模拟器启动时刻 (CLOCK_REALTIME ns)
    double   scale;      // trace 时间压缩比例 (只用于日志)
    uint8_t  reserved[24];
};
struct FileInst {
    uint32_t state;      // kUp / kNotice / kDown
    uint32_t ondemand;   // 1 = 不会被抢占 (模拟器不给它发事件)
    uint64_t epoch;      // 每次 join (新实例) +1
    int64_t  deadline_ns;   // kNotice: 将被抢占的时刻; kDown: 被抢占的时刻
    int64_t  changed_ns;
    uint32_t n_notice, n_preempt, n_join, pad;
    uint8_t  reserved[16];
};
#pragma pack(pop)
static_assert(sizeof(FileHeader) == 64, "FileHeader layout");
static_assert(sizeof(FileInst) == 64, "FileInst layout");

inline int64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}
inline double ns2s(int64_t ns) { return static_cast<double>(ns) * 1e-9; }

struct InstView {
    uint32_t state = kUp;
    uint64_t epoch = 0;
    int64_t  deadline_ns = 0;
};

// ---------------------------------------------------------------------------
// Client: 进程内单例。main() 最开始 (任何线程 / CUDA 之前) 调 Client::get().init()，
// 这样 SIGUSR1 在之后创建的所有线程里都是屏蔽的，只由监视线程 sigtimedwait 收。
// ---------------------------------------------------------------------------
class Client {
public:
    static Client& get() { static Client* c = new Client; return *c; }   // 故意泄漏: 退出时监视线程还可能在用

    void init() { std::call_once(once_, [this] { do_init(); }); }
    bool enabled() const { return enabled_; }
    int  n_inst() const { return n_inst_; }
    const std::string& dir() const { return dir_; }

    InstView view(int inst) const {
        InstView v;
        if (!enabled_ || inst < 0 || inst >= n_inst_) return v;   // 不在文件里的实例 (on-demand) 永远 Up
        if (test_mode_) {
            std::lock_guard<std::mutex> lk(test_mtx_);
            return test_inst_[static_cast<size_t>(inst)];
        }
        for (int tries = 0;; ++tries) {
            const uint64_t s0 = __atomic_load_n(&hdr_->seq, __ATOMIC_ACQUIRE);
            if (s0 & 1u) { if (tries > 1000) std::this_thread::yield(); continue; }
            const FileInst* fi = inst_ + inst;
            v.state       = __atomic_load_n(&fi->state, __ATOMIC_RELAXED);
            v.epoch       = __atomic_load_n(&fi->epoch, __ATOMIC_RELAXED);
            v.deadline_ns = __atomic_load_n(&fi->deadline_ns, __ATOMIC_RELAXED);
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            if (__atomic_load_n(&hdr_->seq, __ATOMIC_RELAXED) == s0) return v;
        }
    }

    // 有变化 (或收到信号) 时 +1；等待者用 wait_change 阻塞
    uint64_t generation() const { return gen_.load(std::memory_order_acquire); }
    void wait_change(uint64_t seen_gen, double timeout_s) {
        std::unique_lock<std::mutex> lk(cv_mtx_);
        cv_.wait_for(lk, std::chrono::duration<double>(timeout_s), [&] { return gen_.load() != seen_gen; });
    }
    // 调度器状态变化 (提交 / 释放 / 偷) 也走同一个 cv，worker 只需要等一个地方
    void kick() {
        { std::lock_guard<std::mutex> lk(cv_mtx_); gen_.fetch_add(1, std::memory_order_acq_rel); }
        cv_.notify_all();
    }

    // ---- 单元测试用: 不读文件，状态由测试代码直接设 ----
    void test_enable(int n_inst) {
        std::lock_guard<std::mutex> lk(test_mtx_);
        enabled_ = true; test_mode_ = true; n_inst_ = n_inst;
        test_inst_.assign(static_cast<size_t>(n_inst), InstView{});
    }
    void test_set(int inst, uint32_t state, int64_t deadline_ns, bool new_epoch) {
        {
            std::lock_guard<std::mutex> lk(test_mtx_);
            auto& v = test_inst_.at(static_cast<size_t>(inst));
            v.state = state; v.deadline_ns = deadline_ns;
            if (new_epoch) ++v.epoch;
        }
        kick();
    }

private:
    Client() = default;
    Client(const Client&) = delete;

    void do_init() {
        const char* d = std::getenv("BUCKET2_SPOT_DIR");
        if (!d || !*d) return;   // 关闭
        dir_ = d;
        // 1) 先屏蔽 SIGUSR1 (之后创建的线程继承)，再装一个空 handler 兜底 (万一某个早于 init 的线程没屏蔽，
        //    默认动作是终止进程)。SA_RESTART: 被打断的系统调用自动重启。
        struct sigaction sa;
        std::memset(&sa, 0, sizeof(sa));
        sa.sa_handler = [](int) {};
        sa.sa_flags = SA_RESTART;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGUSR1, &sa, nullptr);
        sigset_t set;
        sigemptyset(&set);
        sigaddset(&set, SIGUSR1);
        pthread_sigmask(SIG_BLOCK, &set, nullptr);
        // 2) 映射状态文件 (模拟器可能稍晚启动: 最多等 60 s)
        const std::string path = dir_ + "/state.bin";
        int fd = -1;
        for (int i = 0; i < 600 && fd < 0; ++i) {
            fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (fd < 0) throw std::runtime_error("spot: cannot open " + path + " (is spot_sim.py running?)");
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(FileHeader)))
            throw std::runtime_error("spot: bad state file " + path);
        map_len_ = static_cast<size_t>(st.st_size);
        void* p = mmap(nullptr, map_len_, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("spot: mmap failed on " + path);
        hdr_ = static_cast<const FileHeader*>(p);
        if (hdr_->magic != kMagic || hdr_->version != kVersion)
            throw std::runtime_error("spot: state file magic / version mismatch");
        n_inst_ = static_cast<int>(hdr_->n_inst);
        if (map_len_ < sizeof(FileHeader) + static_cast<size_t>(n_inst_) * sizeof(FileInst))
            throw std::runtime_error("spot: state file too short for n_inst");
        inst_ = reinterpret_cast<const FileInst*>(static_cast<const char*>(p) + sizeof(FileHeader));
        enabled_ = true;
        // 3) 登记 pid (模拟器每个事件给登记过的进程发 SIGUSR1)
        ::mkdir((dir_ + "/pids").c_str(), 0755);
        pidfile_ = dir_ + "/pids/" + std::to_string(::getpid());
        if (FILE* f = std::fopen(pidfile_.c_str(), "w")) { std::fprintf(f, "%d\n", ::getpid()); std::fclose(f); }
        static std::string s_pidfile;
        s_pidfile = pidfile_;
        std::atexit([] { ::unlink(s_pidfile.c_str()); });
        // 4) 监视线程: 收 SIGUSR1 (最多等 10 ms，兼作轮询)，seq 变了就唤醒所有等待者
        std::thread([this] {
            sigset_t s;
            sigemptyset(&s);
            sigaddset(&s, SIGUSR1);
            const timespec to{0, 10 * 1000 * 1000};
            uint64_t last = __atomic_load_n(&hdr_->seq, __ATOMIC_ACQUIRE);
            for (;;) {
                const int sig = sigtimedwait(&s, nullptr, &to);
                const uint64_t cur = __atomic_load_n(&hdr_->seq, __ATOMIC_ACQUIRE);
                if (cur != last || sig == SIGUSR1) { last = cur; kick(); }
            }
        }).detach();
        std::cout << "[spot] client: " << n_inst_ << " instances, state " << path << "\n";
    }

    std::once_flag once_;
    bool enabled_ = false, test_mode_ = false;
    int n_inst_ = 0;
    std::string dir_, pidfile_;
    const FileHeader* hdr_ = nullptr;
    const FileInst* inst_ = nullptr;
    size_t map_len_ = 0;
    std::atomic<uint64_t> gen_{0};
    mutable std::mutex cv_mtx_;
    std::condition_variable cv_;
    mutable std::mutex test_mtx_;
    std::vector<InstView> test_inst_;
};

// ---------------------------------------------------------------------------
// 段调度器
// ---------------------------------------------------------------------------
struct WorkerSpec {
    int  inst = -1;        // 状态文件里的实例号 (-1 = 不在文件里 → 永远 Up)
    int  device = 0;       // CUDA 设备号 (调用方 cudaSetDevice 用)
    bool ondemand = false; // on-demand worker: 最后才领、只领小段; 不会被抢占
};

struct Options {
    double drain_margin_s = 0.25;   // 通知期: 预计完成时刻 + margin 超过 deadline 就不再开新单元
    double ondemand_share = 0.25;   // on-demand 每次最多拿 平均份额 × 这个比例 的单元
    bool   ondemand_participates = true;   // false (shared 模式): on-demand 只在没有 spot worker 时救场
    double rescue_after_s = 2.0;    // 所有 spot worker 都不可用多久之后 on-demand 救场
    int64_t min_split_units = 1;    // 偷 / 切分后每一半至少这么多单元
    bool   allow_steal = true;      // spot 关闭时 options_from_env 设成 false → 跟原来的静态切分完全一样
    int64_t min_steal_units = 0;    // stage 速率未知时: 剩余 (从 issued 起) 少于这么多单元就不偷 (防止尾部无限对半)
    double min_steal_s = 0.0;       // 只偷预计剩余时间 (从 issued 起，按 victim 本租约的每单元 EWMA) ≥ 这么多秒的段
                                    // (每个租约有固定开销的阶段: 小段来回偷比不偷还慢)。victim 还没有提交 (EWMA 未知) → 按代价照偷
    bool   verbose = true;

    // 恢复粒度 (实验: exp2 segment-restart, 2026-11-24)。CursorResume (默认，原行为): 被抢占 / 回收后
    // [cursor, end) 放回池子，下次从 cursor 继续。SegmentRestart: 放回 [本租约的 begin, end)，下次从本
    // 租约开始重新计算 — [本租约 begin, 旧 cursor) 这一段会真的在 GPU 上重算一遍，但 commit() 发现这段
    // 已经提交过 (no_resubmit_before 门槛) 就不再调用 fn，不会重复写入 host 结构 (Step 6 的 append 尤其
    // 要靠这个门槛，因为它是只增不查重的)。两种策略共用同一个 may_start / steal / commit 路径，唯一区别
    // 是 grant_locked 发出的 Lease.begin 和 reap/release 时段的回退点。
    enum class Recovery { CursorResume, SegmentRestart };
    Recovery recovery = Recovery::CursorResume;
};

struct Lease {
    int      seg = -1;
    int      worker = -1;
    uint64_t token = 0;   // 每次分配唯一; owner 校验用
    int64_t  begin = 0;   // 领到时的 cursor (从这里开始做)
};

class SegmentScheduler {
public:
    enum class Acq { Got, Done, Lost };

    // init_cuts: 初始切分点 (size = n_init + 1, 首 0 尾 n_units)，第 i 段初始归 init_owner[i] (worker 下标，
    //            -1 = 空闲)。unit_cost (可选, size n_units): 偷的时候按代价对半切。
    SegmentScheduler(std::string stage, int64_t n_units, std::vector<WorkerSpec> workers,
                     const std::vector<int64_t>& init_cuts, const std::vector<int>& init_owner,
                     Options opt = Options{}, std::vector<double> unit_cost = {})
        : stage_(std::move(stage)), n_units_(n_units), workers_(std::move(workers)), opt_(opt),
          cost_prefix_(), wst_(workers_.size()), cl_(Client::get()) {
        if (n_units_ < 0) throw std::runtime_error("spot: negative n_units");
        if (init_cuts.size() < 2 || init_cuts.front() != 0 || init_cuts.back() != n_units_ ||
            init_owner.size() + 1 != init_cuts.size())
            throw std::runtime_error("spot: bad initial cuts for stage " + stage_);
        if (!unit_cost.empty()) {
            if (static_cast<int64_t>(unit_cost.size()) != n_units_) throw std::runtime_error("spot: unit_cost size");
            cost_prefix_.resize(unit_cost.size() + 1, 0.0);
            for (size_t i = 0; i < unit_cost.size(); ++i) cost_prefix_[i + 1] = cost_prefix_[i] + std::max(0.0, unit_cost[i]);
        }
        for (const auto& w : workers_) if (!w.ondemand) ++n_spot_;
        const int64_t t = now_ns();
        for (size_t i = 0; i + 1 < init_cuts.size(); ++i) {
            if (init_cuts[i] > init_cuts[i + 1]) throw std::runtime_error("spot: initial cuts not sorted");
            if (init_cuts[i] == init_cuts[i + 1]) continue;
            auto s = std::make_unique<Seg>();
            s->begin = s->cursor = s->issued = init_cuts[i];
            s->end = init_cuts[i + 1];
            s->lease_begin = s->begin;
            s->no_resubmit_before = s->begin;   // 没有重放过, 门槛就是段起点: 永不触发
            const int o = init_owner[i];
            if (o >= 0 && o < static_cast<int>(workers_.size())) s->reserved_for = o;   // 预留: o 第一次 acquire 时领走
            push_seg(std::move(s));
        }
        t0_ = t;
        last_spot_alive_ns_ = t;
    }

    const std::string& stage() const { return stage_; }
    const std::vector<WorkerSpec>& workers() const { return workers_; }
    bool spot_enabled() const { return cl_.enabled(); }
    // 重放时 commit() 发现已经提交过而跳过 fn 的单元总数 (segment-restart 才会 > 0；cursor-resume 恒 0)
    int64_t redone_units() const { return redone_units_.load(); }

    bool done() const {
        std::lock_guard<std::mutex> lk(gm_);
        return all_done_locked();
    }

    // 实例当前是否可以接新段 (Up)。on-demand / 不在文件里 → 永远 true
    bool worker_up(int w, uint64_t* epoch = nullptr) const {
        const auto v = cl_.view(workers_[static_cast<size_t>(w)].inst);
        if (epoch) *epoch = v.epoch;
        return v.state == kUp;
    }
    // worker 这一代实例是否已经没了 (被抢占 / 换了 epoch)。Notice 期间不算 lost (还能把手上的单元做完)
    bool lost(int w, uint64_t epoch) const {
        const auto v = cl_.view(workers_[static_cast<size_t>(w)].inst);
        return v.state == kDown || v.epoch != epoch;
    }

    // 阻塞到 worker 的实例 Up 且 epoch > after_epoch (第一次传 UINT64_MAX 表示任意 epoch)；stage 做完 → false
    bool wait_worker_up(int w, uint64_t after_epoch, uint64_t* epoch) {
        for (;;) {
            const uint64_t g = cl_.generation();
            if (done()) return false;
            const auto v = cl_.view(workers_[static_cast<size_t>(w)].inst);
            if (v.state == kUp && (after_epoch == UINT64_MAX || v.epoch > after_epoch)) { *epoch = v.epoch; return true; }
            cl_.wait_change(g, 0.05);
        }
    }

    // 领一段。Got: L 有效，从 L.begin 开始做；Done: 整个 stage 做完；Lost: 实例不可用 (被抢占 / 通知期)
    // block=false: 只试一次 (救场 worker 用: 没活就结束这一代、释放显存)，没分到 → Lost
    Acq acquire(int w, Lease& L, bool block = true) {
        const bool od = workers_[static_cast<size_t>(w)].ondemand;
        bool counted_wait = false;
        for (;;) {
            const uint64_t g = cl_.generation();
            {
                std::lock_guard<std::mutex> lk(gm_);
                reap_locked();
                if (all_done_locked()) { if (counted_wait) --waiting_spot_; return Acq::Done; }
                if (!worker_up(w)) { if (counted_wait) --waiting_spot_; return Acq::Lost; }
                if (!od && !counted_wait) { ++waiting_spot_; counted_wait = true; }
                if (try_assign_locked(w, od, L)) {
                    if (counted_wait) --waiting_spot_;
                    ++wst_[static_cast<size_t>(w)].leases;
                    return Acq::Got;
                }
                if (!block) { if (counted_wait) --waiting_spot_; return Acq::Lost; }
            }
            cl_.wait_change(g, 0.02);
        }
    }

    // 声明要开始做 [u0, u1) (u0 = 上一次声明的 u1，或 L.begin)。false → 停止 (段被偷到这里 / 实例被抢占 /
    // 通知期来不及做完)，调用方把已发出的单元做完提交后 release。
    bool may_start(const Lease& L, int64_t u0, int64_t u1) {
        Seg& s = seg(static_cast<size_t>(L.seg));
        const auto v = cl_.view(workers_[static_cast<size_t>(L.worker)].inst);
        std::lock_guard<std::mutex> lk(s.m);
        if (s.owner != L.worker || s.token != L.token) return false;
        if (u0 >= s.end || u1 > s.end || u1 <= u0) return false;
        if (v.state == kDown) return false;
        if (rescue_should_yield(L.worker)) return false;   // shared 模式: spot 实例回来了 → 救场 worker 让出
        if (v.state == kNotice) {
            const double per = s.ewma_unit_s > 0 ? s.ewma_unit_s : 0.0;
            const double est = per * static_cast<double>((std::max<int64_t>(s.issued, u0) - s.cursor) + (u1 - u0));
            if (ns2s(now_ns()) + est + opt_.drain_margin_s > ns2s(v.deadline_ns)) {
                if (!s.drain_logged) { s.drain_logged = true; ++drains_; }
                return false;
            }
        }
        s.issued = std::max(s.issued, u1);
        return true;
    }

    // 同 may_start，但 u1 越过 (被偷后变小的) 段 end 时截到 end: 返回实际声明的 u1' (u0 < u1' ≤ u1)，不能开始 → -1。
    // 单元大小可变的阶段 (prune 的块) 用: 块跨过新 end 时截断，而不是停掉整个租约。
    int64_t may_start_clip(const Lease& L, int64_t u0, int64_t u1) {
        int64_t e;
        {
            const Seg& s = seg(static_cast<size_t>(L.seg));
            std::lock_guard<std::mutex> lk(s.m);
            e = s.end;
        }
        const int64_t u1c = std::min(u1, e);
        if (u1c <= u0) return -1;
        // end 只会变小 (偷的切点 ≥ issued，而 issued 在 may_start 里推进)；在两次加锁之间又被切小 → may_start 失败，重试一次
        if (may_start(L, u0, u1c)) return u1c;
        {
            const Seg& s = seg(static_cast<size_t>(L.seg));
            std::lock_guard<std::mutex> lk(s.m);
            e = s.end;
        }
        const int64_t u1d = std::min(u1, e);
        if (u1d <= u0 || u1d == u1c) return -1;
        return may_start(L, u0, u1d) ? u1d : -1;
    }

    // 提交 [cursor, new_cursor)。合法 (还是 owner、实例没过抢占时刻、单调) 时在段的提交锁内执行 fn
    // (把结果写进 host 端结构) 再推进 cursor，返回 true；否则丢弃 (返回 false，fn 不执行)。
    // u0 (可选): 这次提交的单元起点 (= 上一次提交的 new_cursor / L.begin)。拒收时丢失的单元按 new_cursor - u0 计
    // (不给 u0 时按段当前 cursor 算: 段被回收、重新分配给别人之后那是别人的 cursor，会多算)。
    bool commit(const Lease& L, int64_t new_cursor, const std::function<void()>& fn = {}, int64_t u0 = -1) {
        Seg& s = seg(static_cast<size_t>(L.seg));
        auto& ws = wst_[static_cast<size_t>(L.worker)];
        std::lock_guard<std::mutex> ck(s.cm);   // 提交锁: 回收 (reap) 要等正在进行的提交做完
        int64_t old_cursor;
        bool suppress;
        {
            std::lock_guard<std::mutex> lk(s.m);
            const auto v = cl_.view(workers_[static_cast<size_t>(L.worker)].inst);
            const int64_t t = now_ns();
            const bool gone = (v.state == kDown) || (v.state == kNotice && t >= v.deadline_ns);
            if (s.owner != L.worker || s.token != L.token || gone || new_cursor <= s.cursor || new_cursor > s.end) {
                ws.rejected_commits.fetch_add(1);
                ws.rejected_units.fetch_add(std::max<int64_t>(0, new_cursor - (u0 >= 0 ? u0 : s.cursor)));
                return false;
            }
            old_cursor = s.cursor;
            // segment-restart 重放: [old_cursor, new_cursor) 之前已经提交过一次 (no_resubmit_before 是上次
            // 提交时的 cursor)。真的发生了 GPU 重算 (调用方一定跑到这里才能 commit)，但不能让 fn 把同一批
            // 结果再写一遍 (Step 6 append 只增不查重)。单元粒度在两次经过时确定性相同 (重放从同一个
            // lease_begin 起步，批 / bucket 边界只是位置的函数)，所以 new_cursor 要么 ≤ no_resubmit_before
            // 要么 == 它 (不会跨过去)；用 assert 兜底，防止未来某个 proc() 实现违反这个前提却被默默吞掉。
            suppress = (new_cursor <= s.no_resubmit_before);
            if (!suppress && old_cursor < s.no_resubmit_before)
                throw std::runtime_error("spot: commit straddles no_resubmit_before (" + stage_ +
                                          "): [" + std::to_string(old_cursor) + "," + std::to_string(new_cursor) +
                                          ") vs " + std::to_string(s.no_resubmit_before));
        }
        if (fn && !suppress) fn();
        if (suppress) redone_units_.fetch_add(new_cursor - old_cursor);
        const int64_t t = now_ns();
        bool finished;
        {
            std::lock_guard<std::mutex> lk(s.m);
            const double dt = ns2s(t - (s.last_commit_ns ? s.last_commit_ns : s.lease_t0));
            const double per = dt / static_cast<double>(new_cursor - old_cursor);
            s.ewma_unit_s = s.ewma_unit_s > 0 ? 0.7 * s.ewma_unit_s + 0.3 * per : per;
            {   // stage 级的 秒 / 代价 (偷的时候估计剩余时间: 新租约还没有 EWMA 时也能用)
                std::lock_guard<std::mutex> rl(rate_m_);
                rate_time_s_ += dt;
                rate_cost_ += cost(old_cursor, new_cursor);
            }
            s.last_commit_ns = t;
            s.cursor = new_cursor;
            s.issued = std::max(s.issued, s.cursor);
            finished = (s.cursor == s.end);
        }
        ws.units.fetch_add(new_cursor - old_cursor);
        ws.commits.fetch_add(1);
        if (finished) cl_.kick();
        return true;
    }

    // 结束租约 (段做完 / 被偷到 cursor / 停止)。剩下的 [cursor, end) 变成空闲段。
    void release(const Lease& L) {
        std::lock_guard<std::mutex> lk(gm_);
        Seg& s = seg(static_cast<size_t>(L.seg));
        std::lock_guard<std::mutex> ck(s.cm);
        std::lock_guard<std::mutex> sl(s.m);
        if (s.owner == L.worker && s.token == L.token) {
            // s.cursor < s.end: 这个租约还有没做完的工作就被收回了 (抢占 / 通知期 drain / shared 模式救场
            // 让出)，不是正常做完一段去领下一段。segment-restart 时把 [本租约 begin, 旧 cursor) 也退回去重算。
            if (opt_.recovery == Options::Recovery::SegmentRestart && s.cursor < s.end) {
                if (opt_.verbose && cl_.enabled())
                    log_line("release: segment-restart redo [" + std::to_string(s.lease_begin) + "," +
                             std::to_string(s.cursor) + ") (already-committed prefix, output suppressed)");
                s.no_resubmit_before = std::max(s.no_resubmit_before, s.cursor);
                s.cursor = s.lease_begin;
            }
            s.owner = -1;
            s.issued = s.cursor;
            s.drain_logged = false;
            s.ewma_unit_s = 0; s.last_commit_ns = 0;
        }
        cl_.kick();
    }

    // 当前段 end (可能被偷而变小)
    int64_t end_of(const Lease& L) const {
        const Seg& s = seg(static_cast<size_t>(L.seg));
        std::lock_guard<std::mutex> lk(s.m);
        return s.end;
    }

    // ---- 通用 worker 循环 ----
    // setup(): 冷启动 (建索引 / 分配显存)，每个实例 (epoch) 一次; proc(L): 从 L.begin 起用 may_start / commit 做单元，
    // 停下就返回; teardown(): 释放这一代实例的资源。setup 是懒的 (领到第一段才做)。
    // spot 关闭时: setup → 领初始段 → teardown，跟原来的每卡一个线程一样。
    template <class Setup, class Proc, class Teardown>
    void run_worker(int w, Setup&& setup, Proc&& proc, Teardown&& teardown) {
        auto& ws = wst_[static_cast<size_t>(w)];
        uint64_t last = UINT64_MAX, ep = 0;
        while (wait_worker_up(w, last, &ep)) {
            bool ready = false;   // 懒 setup: 领到第一段才冷启动 (救场 worker 不干活就不占显存)
            Lease L;
            Acq a;
            while ((a = acquire(w, L)) == Acq::Got) {
                if (!ready) {
                    const int64_t ts = now_ns();
                    setup();
                    ws.setups.fetch_add(1);
                    ws.setup_ns.fetch_add(now_ns() - ts);
                    ready = true;
                }
                const int64_t c0 = cursor_of(L);
                proc(L);
                const int64_t c1 = cursor_of(L);
                release(L);
                if (lost(w, ep)) {
                    if (opt_.verbose && cl_.enabled())
                        log_line("worker " + std::to_string(w) + " (dev " + std::to_string(workers_[static_cast<size_t>(w)].device) +
                                 ") preempted; this lease committed [" + std::to_string(c0) + "," + std::to_string(c1) + ")");
                    break;
                }
            }
            if (ready) teardown();
            if (a == Acq::Done) break;
            last = ep;
            ws.preempts.fetch_add(1);   // Lost: 被抢占 (或通知期退出)，等新实例 (epoch 变大)
        }
    }

    // ---- 按实例代 (epoch) 的 worker 循环 (setup / 处理 / teardown 写在一个函数里的阶段用，例如 Step 4) ----
    // 每一代: 等实例 Up → 领到第一段 → body(L, next)。body 里: (懒) setup，调 note_setup；处理 L (may_start /
    // commit)；然后 next(L): 释放当前租约、领下一段 (true，L 换成新租约) 或者这一代结束 (false: 被抢占 / stage 做完 /
    // 救场 worker 没活了)；false 之后 body teardown 并返回。body 没调到 false 就返回 (异常以外不应发生) → 这里释放。
    template <class Body>
    void run_worker_epochs(int w, Body&& body) {
        auto& ws = wst_[static_cast<size_t>(w)];
        const bool rescue = workers_[static_cast<size_t>(w)].ondemand && !opt_.ondemand_participates && cl_.enabled();
        uint64_t last = UINT64_MAX, ep = 0;
        while (wait_worker_up(w, last, &ep)) {
            Lease L;
            Acq a = acquire(w, L);
            if (a == Acq::Done) break;
            if (a == Acq::Lost) { last = ep; ws.preempts.fetch_add(1); continue; }
            bool active = true, done = false, was_lost = false;
            int64_t c0 = cursor_of(L);
            Lease* cur = &L;
            std::function<bool(Lease&)> next = [&](Lease& X) -> bool {
                cur = &X;
                const int64_t c1 = cursor_of(X);
                release(X);
                active = false;
                if (lost(w, ep)) {
                    if (opt_.verbose)
                        log_line("worker " + std::to_string(w) + " (dev " + std::to_string(workers_[static_cast<size_t>(w)].device) +
                                 ") preempted; this lease committed [" + std::to_string(c0) + "," + std::to_string(c1) + ")");
                    was_lost = true;
                    return false;
                }
                const Acq b = acquire(w, X, !rescue);
                if (b == Acq::Got) { active = true; c0 = cursor_of(X); return true; }
                if (b == Acq::Done) done = true;
                else if (!rescue) was_lost = true;
                return false;
            };
            try {
                body(L, next);
            } catch (...) {
                if (active) release(*cur);   // 别让其他 worker 永远等这一段
                throw;
            }
            if (active) release(*cur);
            if (done) break;
            if (was_lost) { last = ep; ws.preempts.fetch_add(1); }
            else last = UINT64_MAX;   // 救场 worker 让出 (不是抢占): 下次任意 epoch
        }
    }
    void note_setup(int w, double seconds) {
        auto& ws = wst_[static_cast<size_t>(w)];
        ws.setups.fetch_add(1);
        ws.setup_ns.fetch_add(static_cast<int64_t>(seconds * 1e9));
    }

    void print_stats(std::ostream& os) const {
        std::lock_guard<std::mutex> lk(gm_);
        const double wall = ns2s(now_ns() - t0_);
        int64_t rej = 0, setups = 0, pre = 0;
        for (const auto& w : wst_) { rej += w.rejected_units.load(); setups += w.setups.load(); pre += w.preempts.load(); }
        os << "[spot] stage=" << stage_ << " units=" << n_units_ << " segments=" << nsegs_.load()
           << " steals=" << steals_ << " drains=" << drains_ << " reaped=" << reaped_
           << " preempts=" << pre << " rejected_units=" << rej << " redone_units=" << redone_units_.load()
           << " setups=" << setups << std::fixed << std::setprecision(3) << " wall=" << wall << "s\n";
        for (size_t i = 0; i < wst_.size(); ++i) {
            const auto& w = wst_[i];
            os << "[spot]   " << stage_ << " w" << i << " dev=" << workers_[i].device << (workers_[i].ondemand ? " ondemand" : " spot")
               << " units=" << w.units.load() << " commits=" << w.commits.load() << " leases=" << w.leases.load()
               << " rejected=" << w.rejected_commits.load() << "/" << w.rejected_units.load()
               << " setups=" << w.setups.load() << " setup_s=" << ns2s(w.setup_ns.load())
               << " preempts=" << w.preempts.load() << "\n";
        }
    }

    // 校验: 所有段都做完且首尾相接覆盖 [0, n_units) (测试 / 收尾断言用)
    bool verify_complete(std::string* why = nullptr) const {
        std::lock_guard<std::mutex> lk(gm_);
        std::vector<std::pair<int64_t, int64_t>> iv;
        for (size_t i = 0; i < nsegs_.load(); ++i) {
            const Seg& s = seg(i);
            std::lock_guard<std::mutex> sl(s.m);
            if (s.cursor != s.end) { if (why) *why = "segment not finished"; return false; }
            if (s.begin < s.end) iv.emplace_back(s.begin, s.end);
        }
        std::sort(iv.begin(), iv.end());
        int64_t at = 0;
        for (auto& p : iv) { if (p.first != at) { if (why) *why = "gap/overlap at " + std::to_string(at); return false; } at = p.second; }
        if (at != n_units_) { if (why) *why = "coverage ends at " + std::to_string(at); return false; }
        return true;
    }

    void log_line(const std::string& msg) const {
        static std::mutex m;
        std::lock_guard<std::mutex> lk(m);
        std::cout << "[spot] " << std::fixed << std::setprecision(3) << ns2s(now_ns() - t0_) << "s " << stage_ << ": " << msg << "\n";
        std::cout.flush();
    }

private:
    struct Seg {
        int64_t begin = 0, cursor = 0, end = 0, issued = 0;
        int owner = -1;
        int reserved_for = -1;   // 初始切分: 留给这个 worker (它的实例不可用时对别人就是空闲段)
        uint64_t token = 0;
        int64_t lease_t0 = 0, last_commit_ns = 0;
        double ewma_unit_s = 0;
        bool drain_logged = false;
        int64_t lease_begin = 0;          // 当前 (或最近一次) 租约的起点; grant_locked 每次授予都重设
        int64_t no_resubmit_before = 0;   // < 这个位置的 commit 只推进 cursor, 不调用 fn (已经真提交过一次了);
                                           // segment-restart 回退时设成回退前的 cursor, 偷/切分时原样传给新段
        mutable std::mutex m;    // 段状态
        std::mutex cm;           // 提交锁
    };
    struct WStat {
        std::atomic<int64_t> units{0}, commits{0}, leases{0}, rejected_commits{0}, rejected_units{0};
        std::atomic<int64_t> setups{0}, setup_ns{0}, preempts{0};
    };

    // shared 模式的救场 worker: 有 spot 实例 Up 就停止开新单元 (它跟 spot 实例同卡，不能同时占显存)
    bool rescue_should_yield(int w) const {
        if (!workers_[static_cast<size_t>(w)].ondemand || opt_.ondemand_participates || !cl_.enabled()) return false;
        for (size_t i = 0; i < workers_.size(); ++i)
            if (!workers_[i].ondemand && worker_up(static_cast<int>(i))) return true;
        return false;
    }

    // [from, s.end) 一个 worker 做完的预计秒数: stage 级 秒/代价 (所有提交累计)；还没有提交 → 本租约 EWMA；都没有 → -1
    double est_time_locked(const Seg& s, int64_t from) const {
        double rt = 0, rc = 0;
        { std::lock_guard<std::mutex> rl(rate_m_); rt = rate_time_s_; rc = rate_cost_; }
        if (rc > 0 && rt > 0) return rt / rc * cost(from, s.end);
        if (s.ewma_unit_s > 0) return s.ewma_unit_s * static_cast<double>(s.end - from);
        return -1.0;
    }

    int64_t cursor_of(const Lease& L) const {
        const Seg& s = seg(static_cast<size_t>(L.seg));
        std::lock_guard<std::mutex> lk(s.m);
        return s.cursor;
    }

    double cost(int64_t a, int64_t b) const {   // [a, b) 的代价
        if (cost_prefix_.empty()) return static_cast<double>(b - a);
        return cost_prefix_[static_cast<size_t>(b)] - cost_prefix_[static_cast<size_t>(a)];
    }
    // [a, b) 里从 a 起代价累计到 target 的位置 (a < p < b)
    int64_t cut_at_cost(int64_t a, int64_t b, double target) const {
        if (cost_prefix_.empty()) return a + static_cast<int64_t>(target);
        const double base = cost_prefix_[static_cast<size_t>(a)];
        auto it = std::lower_bound(cost_prefix_.begin() + a, cost_prefix_.begin() + b + 1, base + target);
        return static_cast<int64_t>(it - cost_prefix_.begin());
    }

    bool all_done_locked() const {
        for (size_t i = 0; i < nsegs_.load(); ++i) {
            const Seg& s = seg(i);
            std::lock_guard<std::mutex> sl(s.m);
            if (s.cursor < s.end) return false;
        }
        return true;
    }

    // 回收: owner 的实例已经 Down (过了抢占时刻) 或换了代 → 段立即变空闲 (模拟 host 发现实例没了)。
    // 先拿提交锁，等正在进行的提交做完。
    void reap_locked() {
        if (!cl_.enabled()) return;
        bool any_spot_alive = false;
        for (size_t w = 0; w < workers_.size(); ++w)
            if (!workers_[w].ondemand && worker_up(static_cast<int>(w))) any_spot_alive = true;
        if (any_spot_alive || n_spot_ == 0) last_spot_alive_ns_ = now_ns();
        for (size_t i = 0; i < nsegs_.load(); ++i) {
            Seg& s = seg(i);
            int owner;
            { std::lock_guard<std::mutex> sl(s.m); owner = s.owner; }
            if (owner < 0) continue;
            const auto v = cl_.view(workers_[static_cast<size_t>(owner)].inst);
            if (v.state != kDown) continue;
            std::lock_guard<std::mutex> ck(s.cm);
            std::lock_guard<std::mutex> sl(s.m);
            if (s.owner != owner) continue;
            if (opt_.verbose)
                log_line("reap w" + std::to_string(owner) + ": [" + std::to_string(s.cursor) + "," + std::to_string(s.end) +
                         ") back to the pool (issued-uncommitted " + std::to_string(s.issued - s.cursor) + " units dropped)");
            if (opt_.recovery == Options::Recovery::SegmentRestart && s.cursor < s.end) {
                if (opt_.verbose)
                    log_line("reap: segment-restart redo [" + std::to_string(s.lease_begin) + "," +
                             std::to_string(s.cursor) + ") (already-committed prefix, output suppressed)");
                s.no_resubmit_before = std::max(s.no_resubmit_before, s.cursor);
                s.cursor = s.lease_begin;
            }
            s.owner = -1; s.issued = s.cursor; s.drain_logged = false; s.ewma_unit_s = 0; s.last_commit_ns = 0;
            ++reaped_;
        }
    }

    int64_t fair_share() const { return std::max<int64_t>(1, n_units_ / std::max(1, n_spot_)); }

    // 在 gm_ 下: 给 w 找一段。spot: 空闲段 (整段) > 偷剩余最多段的后一半。on-demand: 只拿小段，spot 有人在等就让。
    bool try_assign_locked(int w, bool od, Lease& L) {
        const int64_t t = now_ns();
        if (od) {
            if (waiting_spot_ > 0) return false;   // spot 优先
            if (!opt_.ondemand_participates && cl_.enabled() && n_spot_ > 0 &&
                ns2s(t - last_spot_alive_ns_) < opt_.rescue_after_s)
                return false;   // shared 模式: 只救场
            if (!opt_.ondemand_participates && !cl_.enabled() && n_spot_ > 0) return false;
        }
        const int64_t od_cap = std::max<int64_t>(opt_.min_split_units,
                                                 static_cast<int64_t>(opt_.ondemand_share * static_cast<double>(fair_share())));
        // 0) 预留给自己的初始段
        for (size_t i = 0; i < nsegs_.load(); ++i) {
            Seg& s = seg(i);
            std::lock_guard<std::mutex> sl(s.m);
            if (s.reserved_for == w && s.owner < 0 && s.cursor < s.end) { s.reserved_for = -1; grant_locked(s, i, w, t, L); return true; }
        }
        // 1) 空闲段: 起点最小的那段 (保持 bucket order 局部性)。预留给别的 (可用的) worker 的段不拿。
        Seg* best = nullptr; size_t best_i = 0;
        for (size_t i = 0; i < nsegs_.load(); ++i) {
            Seg& s = seg(i);
            std::lock_guard<std::mutex> sl(s.m);
            if (s.reserved_for >= 0 && s.reserved_for != w) {
                if (worker_up(s.reserved_for)) continue;
                s.reserved_for = -1;   // 预留的实例不可用 → 放开
            }
            if (s.owner < 0 && s.cursor < s.end && (!best || s.cursor < best->cursor)) { best = &s; best_i = i; }
        }
        if (best) {
            std::lock_guard<std::mutex> sl(best->m);
            // segment-restart 重放中的段 (cursor < no_resubmit_before) 不切: 切点是代价/固定份额算出来的，
            // 不保证落在这个 stage 的"单元"边界上 (Step 4 的 batch_end_of 只是 batch_start 的函数, 不知道
            // no_resubmit_before 在哪)，切出来的新段可能跟已提交门槛错位。等重放追上 no_resubmit_before
            // 再切——通常很快 (重放的只是上次被抢占时丢的那一小段)。
            if (od && best->cursor >= best->no_resubmit_before &&
                best->end - best->cursor > od_cap + opt_.min_split_units && can_split()) {
                // on-demand 只拿头部一小段，剩下的仍空闲 (另起一段)
                auto rest = std::make_unique<Seg>();
                rest->begin = rest->cursor = rest->issued = best->cursor + od_cap;
                rest->end = best->end;
                rest->no_resubmit_before = best->no_resubmit_before;   // 门槛是绝对位置, 跨切分点也要带过去
                best->end = rest->begin;
                push_seg(std::move(rest));
            }
            grant_locked(*best, best_i, w, t, L);
            return true;
        }
        if (!opt_.allow_steal || !can_split()) return false;
        // 2) 偷: 剩余 (从已发出位置算) 代价最大的段，从尾部切
        Seg* vic = nullptr; double vic_cost = 0;
        for (size_t i = 0; i < nsegs_.load(); ++i) {
            Seg& s = seg(i);
            std::lock_guard<std::mutex> sl(s.m);
            if (s.owner < 0 || s.owner == w) continue;
            if (s.cursor < s.no_resubmit_before) continue;   // 同上: 重放中的段不偷, 切点可能跟门槛错位
            const int64_t from = std::max(s.issued, s.cursor);
            if (s.end - from < 2 * opt_.min_split_units) continue;
            if (opt_.min_steal_s > 0) {   // 剩下的预计时间太少，不值得偷 (新租约有固定开销)
                const double est = est_time_locked(s, from);
                if (est >= 0 && est < opt_.min_steal_s) continue;
                if (est < 0 && s.end - from < opt_.min_steal_units) continue;
            }
            const double c = cost(from, s.end);
            if (c > vic_cost) { vic_cost = c; vic = &s; }
        }
        if (!vic) return false;
        int64_t mid;
        size_t new_idx = 0;
        {
            std::lock_guard<std::mutex> sl(vic->m);
            const int64_t from = std::max(vic->issued, vic->cursor);
            if (vic->end - from < 2 * opt_.min_split_units) return false;
            if (od) {
                const int64_t take = std::min<int64_t>(od_cap, (vic->end - from) / 2);
                mid = vic->end - std::max<int64_t>(take, opt_.min_split_units);
            } else {
                mid = cut_at_cost(from, vic->end, 0.5 * cost(from, vic->end));
            }
            mid = std::max(mid, from + opt_.min_split_units);
            mid = std::min(mid, vic->end - opt_.min_split_units);
            if (mid <= from || mid >= vic->end) return false;
            auto ns = std::make_unique<Seg>();
            ns->begin = ns->cursor = ns->issued = mid;
            ns->end = vic->end;
            ns->no_resubmit_before = vic->no_resubmit_before;   // 同上: 偷的切点可能落在 victim 的重放门槛之前
            vic->end = mid;
            new_idx = push_seg(std::move(ns));
        }
        ++steals_;
        Seg& ns = seg(new_idx);
        std::lock_guard<std::mutex> sl(ns.m);
        grant_locked(ns, new_idx, w, t, L);
        if (opt_.verbose && cl_.enabled())
            log_line("w" + std::to_string(w) + (od ? " (ondemand)" : "") + " steals [" + std::to_string(ns.begin) + "," +
                     std::to_string(ns.end) + ")");
        return true;
    }

    void grant_locked(Seg& s, size_t idx, int w, int64_t t, Lease& L) {   // s.m 已锁
        s.owner = w; s.token = ++token_ctr_; s.lease_t0 = t; s.last_commit_ns = 0; s.ewma_unit_s = 0;
        s.issued = s.cursor; s.drain_logged = false;
        s.lease_begin = s.cursor;   // 这次租约的起点 (segment-restart 回退目标; cursor-resume 不用这个字段)
        L.seg = static_cast<int>(idx); L.worker = w; L.token = s.token; L.begin = s.cursor;
    }

    std::string stage_;
    int64_t n_units_;
    std::vector<WorkerSpec> workers_;
    Options opt_;
    std::vector<double> cost_prefix_;
    std::vector<WStat> wst_;
    Client& cl_;
    mutable std::mutex gm_;
    // 段表: 固定容量数组，只增不删不搬 (无锁读者用下标直接访问，不能用会扩容的 vector)
    static constexpr size_t kMaxSegs = 1 << 16;
    std::unique_ptr<std::unique_ptr<Seg>[]> seg_arr_{new std::unique_ptr<Seg>[kMaxSegs]};
    std::atomic<size_t> nsegs_{0};
    Seg& seg(size_t i) const {
        if (i >= nsegs_.load(std::memory_order_acquire)) throw std::runtime_error("spot: bad segment index");
        return *seg_arr_[i];
    }
    size_t push_seg(std::unique_ptr<Seg> p) {   // gm_ 下 (或构造时)
        const size_t i = nsegs_.load(std::memory_order_relaxed);
        if (i >= kMaxSegs) throw std::runtime_error("spot: too many segments");
        seg_arr_[i] = std::move(p);
        nsegs_.store(i + 1, std::memory_order_release);
        return i;
    }
    bool can_split() const { return nsegs_.load() + 1 < kMaxSegs; }
    uint64_t token_ctr_ = 0;
    int n_spot_ = 0;
    int waiting_spot_ = 0;
    int64_t steals_ = 0, drains_ = 0, reaped_ = 0;
    std::atomic<int64_t> redone_units_{0};   // 写在 commit() 里, 不持 gm_ 锁 (各段提交并行): 必须是原子的
    mutable std::mutex rate_m_;
    double rate_time_s_ = 0, rate_cost_ = 0;
    int64_t t0_ = 0, last_spot_alive_ns_ = 0;
};

// ---------------------------------------------------------------------------
// 工具: 从环境变量组 worker 列表。
//   G = 设备数 (CUDA 可见设备 0..G-1)。模式 BUCKET2_SPOT_MODE:
//     dedicated (G ≥ 2 默认): 设备 0 = on-demand worker (最后才领、领小段)，设备 1..G-1 = spot 实例 1..G-1
//     shared    (G = 1 默认): 每个设备都是 spot 实例 (实例号 = 设备号)；on-demand 角色跟设备 0 共用，只救场
//   spot 关闭时 (没有 BUCKET2_SPOT_DIR): 每个设备一个普通 worker，ondemand=false → 跟原来的静态切分一样。
// ---------------------------------------------------------------------------
inline std::vector<WorkerSpec> make_workers(int G, bool* shared_mode = nullptr) {
    std::vector<WorkerSpec> ws;
    Client& cl = Client::get();
    if (!cl.enabled()) {
        for (int g = 0; g < G; ++g) ws.push_back(WorkerSpec{-1, g, false});
        if (shared_mode) *shared_mode = false;
        return ws;
    }
    const char* m = std::getenv("BUCKET2_SPOT_MODE");
    const bool shared = m ? std::string(m) == "shared" : (G == 1);
    if (shared_mode) *shared_mode = shared;
    if (shared) {
        for (int g = 0; g < G; ++g) ws.push_back(WorkerSpec{g, g, false});
        ws.push_back(WorkerSpec{-1, 0, true});   // 救场 worker (设备 0)
    } else {
        ws.push_back(WorkerSpec{-1, 0, true});
        for (int g = 1; g < G; ++g) ws.push_back(WorkerSpec{g, g, false});
    }
    return ws;
}

inline Options options_from_env(bool shared_mode) {
    Options o;
    o.ondemand_participates = !shared_mode;
    o.allow_steal = Client::get().enabled();
    if (const char* e = std::getenv("BUCKET2_SPOT_STEAL")) o.allow_steal = e[0] != '0';
    if (const char* e = std::getenv("BUCKET2_SPOT_DRAIN_MARGIN_S")) o.drain_margin_s = std::atof(e);
    if (const char* e = std::getenv("BUCKET2_SPOT_OD_SHARE")) o.ondemand_share = std::atof(e);
    if (const char* e = std::getenv("BUCKET2_SPOT_RESCUE_S")) o.rescue_after_s = std::atof(e);
    if (const char* e = std::getenv("BUCKET2_SPOT_QUIET")) o.verbose = e[0] != '1';
    if (const char* e = std::getenv("BUCKET2_SPOT_MIN_STEAL_S")) o.min_steal_s = std::atof(e);
    // exp2 (segment-restart ablation, 2026-11-24): "cursor" (默认, 不设或设别的值都一样) / "restart"
    if (const char* e = std::getenv("BUCKET2_SPOT_RECOVERY"))
        o.recovery = (std::string(e) == "restart") ? Options::Recovery::SegmentRestart : Options::Recovery::CursorResume;
    return o;
}

}  // namespace spot
