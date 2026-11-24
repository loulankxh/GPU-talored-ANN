// 单元测试: spot_sched.hpp 段调度器 (CPU only, 用 Client 的测试模式代替 mmap 状态文件)
// 编译: g++ -std=c++17 -O2 -pthread -I.. test_spot_sched.cpp -o /tmp/test_spot_sched
#include "../spot_sched.hpp"

#include <cassert>
#include <deque>
#include <random>

using namespace spot;

static int g_fail = 0;
static bool g_clip = false;
#define CHECK(c, msg) do { if (!(c)) { std::cerr << "FAIL: " << msg << " (" #c ")\n"; ++g_fail; } } while (0)

// 一个 worker 的处理函数: 每次声明 [u, u+b) (b 随机 1..maxb)，"计算" 后提交。pipelined=true 时
// 最多 depth 个单元在途，按顺序提交 (模拟 Step 6 的 slot + 顺序 scatter)。
struct Sim {
    std::vector<std::atomic<int>> cnt;
    explicit Sim(int64_t n) : cnt(static_cast<size_t>(n)) { for (auto& c : cnt) c = 0; }
};

// 批大小是位置 u 的确定性函数 (跟真实系统的 batch_end_of(batch_start) 一样只依赖起点，不依赖历史 /
// 哪个 worker / 第几次经过)。这对 segment-restart 是必须的: 重放 [lease_begin, 旧 cursor) 时，切出来的
// 批边界必须跟原来那次完全一样，才能保证 no_resubmit_before 正好落在一个批边界上 (commit() 里有 assert
// 检查这件事; 这里如果用真随机 rng()，同一个 u 在原始 pass 和重放 pass 会切出不同的批，assert 就会炸)。
static int64_t det_batch_len(int64_t u, int maxb) {
    uint64_t h = static_cast<uint64_t>(u) * 2654435761ULL;
    h ^= h >> 13; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
    return 1 + static_cast<int64_t>(h % static_cast<uint64_t>(maxb));
}

static void proc_units(SegmentScheduler& S, const Lease& L, Sim& sim, int maxb, int us_per_unit, bool pipelined,
                       std::mt19937& rng) {
    (void)rng;   // 批大小已经是 u 的确定性函数 (见上)，这个 rng 留着只是不想改调用处的签名
    std::deque<std::pair<int64_t, int64_t>> inflight;
    int64_t u = L.begin;
    const int depth = pipelined ? 6 : 1;
    auto commit_front = [&] {
        auto [a, b] = inflight.front();
        inflight.pop_front();
        std::this_thread::sleep_for(std::chrono::microseconds(us_per_unit * (b - a)));
        S.commit(L, b, [&] { for (int64_t i = a; i < b; ++i) sim.cnt[static_cast<size_t>(i)].fetch_add(1); });
    };
    for (;;) {
        const int64_t e = S.end_of(L);
        if (u >= e) break;
        int64_t b = std::min<int64_t>(e, u + det_batch_len(u, maxb));
        if (g_clip) {   // prune 风格: 先按旧 end 规划一个可能越界的块，may_start_clip 截断
            b = u + det_batch_len(u, maxb);
            b = S.may_start_clip(L, u, b);
            if (b <= u) break;
            if (b > S.end_of(L)) { std::cerr << "FAIL: clip beyond end\n"; ++g_fail; }
        } else if (!S.may_start(L, u, b)) break;
        inflight.emplace_back(u, b);
        u = b;
        if (static_cast<int>(inflight.size()) >= depth) commit_front();
    }
    while (!inflight.empty()) commit_front();
}

static void run_case(const char* name, int G, bool spot_on, const char* mode, int64_t n, bool pipelined, int events,
                     bool hard_only = false, bool epochs = false, const char* recovery = "cursor") {
    std::cout << "=== " << name << " ===\n";
    Client& cl = Client::get();
    if (spot_on) cl.test_enable(G);
    if (mode) setenv("BUCKET2_SPOT_MODE", mode, 1); else unsetenv("BUCKET2_SPOT_MODE");
    setenv("BUCKET2_SPOT_QUIET", "1", 1);
    setenv("BUCKET2_SPOT_RECOVERY", recovery, 1);   // exp2: "cursor" (默认) 或 "restart"
    bool shared = false;
    auto ws = make_workers(G, &shared);
    auto opt = options_from_env(shared);
    opt.rescue_after_s = 0.05;
    // 初始切分: 只给 spot worker (dedicated 模式的 on-demand 从偷开始)
    std::vector<int> spot_w;
    for (size_t i = 0; i < ws.size(); ++i) if (!ws[i].ondemand) spot_w.push_back(static_cast<int>(i));
    if (spot_w.empty()) spot_w.push_back(0);
    std::vector<int64_t> cuts{0};
    std::vector<int> owners;
    for (size_t k = 0; k < spot_w.size(); ++k) { cuts.push_back(n * static_cast<int64_t>(k + 1) / static_cast<int64_t>(spot_w.size())); owners.push_back(spot_w[k]); }
    std::vector<double> ucost(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) ucost[static_cast<size_t>(i)] = 1.0 + static_cast<double>(i % 7);
    SegmentScheduler S(name, n, ws, cuts, owners, opt, ucost);
    Sim sim(n);
    std::atomic<bool> stop_ctrl{false};
    std::thread ctrl;
    if (spot_on && events > 0) {
        ctrl = std::thread([&] {
            std::mt19937 r(123);
            int done_ev = 0;
            while (!stop_ctrl && done_ev < events) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20 + r() % 40));
                // 挑一个 spot 实例 (文件里的实例 = ws[i].inst >= 0)
                std::vector<int> insts;
                for (auto& w : ws) if (w.inst >= 0) insts.push_back(w.inst);
                if (insts.empty()) break;
                const int inst = insts[r() % insts.size()];
                const auto v = cl.view(inst);
                if (v.state != kUp) continue;
                const bool hard = hard_only || (r() % 3 == 0);
                if (hard) {
                    cl.test_set(inst, kDown, now_ns(), false);
                } else {
                    const int64_t dl = now_ns() + static_cast<int64_t>(5 + r() % 30) * 1000000LL;
                    cl.test_set(inst, kNotice, dl, false);
                    std::this_thread::sleep_until(std::chrono::system_clock::time_point(std::chrono::nanoseconds(dl)));
                    cl.test_set(inst, kDown, dl, false);
                }
                ++done_ev;
                const int down_ms = 10 + static_cast<int>(r() % 60);
                std::thread([&cl, inst, down_ms] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(down_ms));
                    cl.test_set(inst, kUp, 0, true);   // 新实例
                }).detach();
            }
        });
    }
    std::vector<std::thread> th;
    std::atomic<int> setups{0}, teardowns{0};
    // epochs: run_worker_epochs (Step 4 风格) + 每设备一把 epoch 锁 (同卡的 worker 不能同时持有资源)
    std::vector<std::mutex> dev_m(static_cast<size_t>(G));
    std::vector<std::atomic<int>> dev_active(static_cast<size_t>(G));
    for (auto& a : dev_active) a = 0;
    std::atomic<int> dev_overlap{0};
    for (size_t w = 0; w < ws.size() && epochs; ++w) {
        th.emplace_back([&, w] {
            std::mt19937 rng(static_cast<unsigned>(w * 7919 + 1));
            const int d = ws[w].device;
            S.run_worker_epochs(static_cast<int>(w), [&](Lease& L, const std::function<bool(Lease&)>& next) {
                std::lock_guard<std::mutex> dl(dev_m[static_cast<size_t>(d)]);
                if (dev_active[static_cast<size_t>(d)].fetch_add(1) != 0) dev_overlap++;
                setups++;
                S.note_setup(static_cast<int>(w), 0.002);
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                do { proc_units(S, L, sim, 16, 20, pipelined, rng); } while (next(L));
                dev_active[static_cast<size_t>(d)].fetch_sub(1);
                teardowns++;
            });
        });
    }
    for (size_t w = 0; w < ws.size() && !epochs; ++w) {
        th.emplace_back([&, w] {
            std::mt19937 rng(static_cast<unsigned>(w * 7919 + 1));
            S.run_worker(static_cast<int>(w),
                         [&] { setups++; std::this_thread::sleep_for(std::chrono::milliseconds(2)); },
                         [&](const Lease& L) { proc_units(S, L, sim, 16, 20, pipelined, rng); },
                         [&] { teardowns++; });
        });
    }
    for (auto& t : th) t.join();
    stop_ctrl = true;
    if (ctrl.joinable()) ctrl.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(120));   // 让 detached 的 join 线程跑完
    int64_t bad = 0, zero = 0, dup = 0;
    for (auto& c : sim.cnt) { if (c == 0) ++zero; else if (c > 1) ++dup; if (c != 1) ++bad; }
    std::string why;
    CHECK(S.verify_complete(&why), name << ": verify_complete: " << why);
    // 核心正确性断言 (对两种恢复策略都成立，不止 cursor-resume): 每个单元的 fn 恰好跑一次。segment-restart
    // 下真的会重算 [lease_begin, 旧 cursor)，但 commit() 必须拦住重复的那一次 fn 调用 —— 这里的 bad==0
    // 就是在验证这件事：如果拦截逻辑 (no_resubmit_before) 有 bug，某个单元会被数成 2，dup 就会 > 0。
    CHECK(bad == 0, name << ": units committed != 1: missing=" << zero << " duplicated=" << dup);
    CHECK(setups == teardowns, name << ": setup/teardown mismatch");
    CHECK(dev_overlap == 0, name << ": two workers held the same device at once");
    if (std::string(recovery) == "restart" && spot_on && events > 0)
        CHECK(S.redone_units() > 0, name << ": segment-restart ran with " << events << " events but redone_units == 0 (mechanism never exercised -- is BUCKET2_SPOT_RECOVERY wired up?)");
    if (std::string(recovery) != "restart")
        CHECK(S.redone_units() == 0, name << ": cursor-resume must never suppress a commit (redone_units should stay 0)");
    S.print_stats(std::cout);
}

int main() {
    // 1) spot 关闭: 静态切分，不偷 (行为同原来)
    run_case("disabled_g4", 4, false, nullptr, 20000, false, 0);
    // 2) dedicated (on-demand worker + 3 spot)，混合通知 / 硬抢占，逐单元提交
    run_case("dedicated_g4", 4, true, nullptr, 40000, false, 12);
    // 3) 同上，流水线提交 (Step 6 风格)
    run_case("dedicated_g4_pipe", 4, true, nullptr, 40000, true, 12);
    // 4) G = 2 dedicated
    run_case("dedicated_g2", 2, true, nullptr, 30000, true, 8);
    // 5) shared G = 1: 唯一 spot 实例反复被抢占，on-demand 只救场
    run_case("shared_g1", 1, true, nullptr, 20000, true, 6);
    // 6) shared G = 4, 只有硬抢占
    run_case("shared_g4_hard", 4, true, "shared", 40000, true, 10, true);
    // 8-10) run_worker_epochs (Step 4 风格)
    run_case("epochs_dedicated_g4", 4, true, nullptr, 40000, true, 12, false, true);
    run_case("epochs_shared_g1", 1, true, nullptr, 20000, true, 8, false, true);
    // 11) may_start_clip (prune 风格) + 最小偷段时间
    g_clip = true;
    setenv("BUCKET2_SPOT_MIN_STEAL_S", "0.002", 1);
    run_case("clip_dedicated_g4", 4, true, nullptr, 40000, true, 12);
    run_case("clip_epochs_shared_g1", 1, true, nullptr, 20000, true, 8, false, true);
    g_clip = false;
    unsetenv("BUCKET2_SPOT_MIN_STEAL_S");
    // 7) n = 0
    run_case("empty", 2, true, nullptr, 0, false, 0);

    // --- exp2: segment-restart (BUCKET2_SPOT_RECOVERY=restart). 同样的事件驱动场景，换恢复策略；
    // run_case 的 bad==0 断言 (每个单元 fn 恰好一次) 和新加的 redone_units 断言一起验证: 重放真的发生了
    // (redone_units > 0) 且没有重复写 (bad == 0, 即没有一个单元被 fn 算了两次)。
    run_case("restart_dedicated_g4", 4, true, nullptr, 40000, false, 12, false, false, "restart");
    run_case("restart_dedicated_g4_pipe", 4, true, nullptr, 40000, true, 12, false, false, "restart");
    run_case("restart_dedicated_g2", 2, true, nullptr, 30000, true, 8, false, false, "restart");
    run_case("restart_shared_g1", 1, true, nullptr, 20000, true, 6, false, false, "restart");
    run_case("restart_shared_g4_hard", 4, true, "shared", 40000, true, 10, true, false, "restart");
    run_case("restart_epochs_dedicated_g4", 4, true, nullptr, 40000, true, 12, false, true, "restart");
    run_case("restart_epochs_shared_g1", 1, true, nullptr, 20000, true, 8, false, true, "restart");
    g_clip = true;
    setenv("BUCKET2_SPOT_MIN_STEAL_S", "0.002", 1);
    run_case("restart_clip_dedicated_g4", 4, true, nullptr, 40000, true, 12, false, false, "restart");
    run_case("restart_clip_epochs_shared_g1", 1, true, nullptr, 20000, true, 8, false, true, "restart");
    g_clip = false;
    unsetenv("BUCKET2_SPOT_MIN_STEAL_S");

    if (g_fail) { std::cerr << g_fail << " check(s) failed\n"; return 1; }
    std::cout << "ALL PASS\n";
    return 0;
}
