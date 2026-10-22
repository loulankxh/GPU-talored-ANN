# A1 — GNN-Descent 式 local join 精修（用户已批准，2026-10-21）

状态：**APPROVED**（用户 2026-10-21 批准，可以直接尝试）。
目的：bucket-pair KNN 漏掉的真邻居主要是 NPROBE 个邻居 bucket 之外的边界点；
local join（"邻居的邻居很可能也是邻居"）能跨 bucket 边界把它们补回来，
从而允许降低 `NPROBE` / `ITERATIONS`（A1 实验里可以和精修配套调这两个参数），
用更便宜的精修换掉一部分 bucket-pair 计算。

两条路径分别做：

| 路径 | 做法 | 位置 |
|---|---|---|
| GPU-resident（数据放得进 GPU，SIFT1M auto） | 直接参考 TAGORE 的 GNN-Descent（§1） | bucket2 产出 degree-64 `vector_knn` 之后、`optimize` 之前（或融进 bucket2 resident 的最后一步，图已经在 GPU 上） |
| 大数据集（out-of-core，SIFT1M large / SIFT10M） | 用户的思路：**prune 和 local join 融合成一趟**（§2） | 取代 `prune_windowed` 的计算部分；之后的反向边（`optimize_chunked -m E`）和 `translate_ids` 不变 |

不变的东西：bucketing、bucket 顺序、bucket-pair KNN 仍然是候选的主要来源（invariants 1–3）；
反向边 + merge（5）、ID translation（6）、build degree 64 / final degree 32、
两条路径的自动选择、大路径的 GPU / host 内存上界。

TAGORE 源码：`/home/lanlu/Tagore`（下文的文件都相对于它）。

---

## 1. GPU-resident 路径：照 TAGORE GNN-Descent 做

参考 `pybind.cu::GNN_descent`（驱动）和 `src/Tagore_src.cu`。
区别：TAGORE 从随机图开始迭代（`initialize_graph`），我们从 bucket2 的 degree-64
KNN 开始，所以只需要很少几趟（先试 1–2 趟），所有初始邻居一开始都标成 new。

每一趟（TAGORE 的 phase 1）：
1. **取样** — `nn_descent_opt_sample`：对每个点 u，按 visit 标记把 N(u) 分成 new / old，
   各取最多 SAMPLE（= 32，和 TAGORE 一样）个；同时用 `atomicAdd` 把 u 写进被采样邻居的
   `reverse_graph`（反向邻居 R(u)）。
2. **反向取样** — `nn_descent_opt_reverse_sample`：把反向邻居也并进 new / old 采样集并去重。
3. **算距离** — `nn_descent_opt_cal`：采样向量载入 shared memory，用 wmma
   tensor core 分块矩阵乘，dist(a,b)=‖a‖²+‖b‖²−2a·b（范数由 `cal_power` 预先算好）；
   只算 new×new 和 new×old 的对。
4. **合并** — `nn_descent_opt_merge`：先用当前列表第 64 近的距离做阈值（`min_ele`）
   过滤掉不可能进榜的候选，再 bitonic sort 去重 + 二分插入；被插入的点标成 new，
   下一趟只处理有变化的部分。

可选 phase 2（`sample_kernel6` + `merge_reverse_plus`）先不做，看 phase 1 效果再说。
resident 路径照 TAGORE 用反向邻居（第 2 步）；不带反向邻居的版本作为变体可以单独测。

移植注意：TAGORE 写死 K=96 / DIM=128、fp16 数据、几乎没有边界检查、反向表
（RESERVENUM=128）满了会静默丢边。我们的数据是 u8：转 fp16 是精确的，但点积用
fp32 累加；所有新的下标访问加边界检查。

---

## 2. 大数据集路径：prune + local join 融合（用户思路）

### 2.1 用户原话（保留，勿改）

第一次说明（2026-10-21）：

> 对于大数据集建图，我有一个思路是：我现在在做prune的时候也需要用到neighbor的neighbor，
> 所以可以和local join一起做。假设正在为点 \(p\) 选择邻居， \(u\) 是正在检查的邻居，
> \(V\) 是 \(u\) 的所有neighbor。我会假设 \(V\) 中此时还没被加入的（也就是local join需要
> 补加的那些点）离p的距离比u离p远，所以即使要被加入也要在u的后面。而对于在u后面的点，
> 我可以同时检查两项，是否要被RNG的剪枝prune掉（d(p,u)<d(p,v), & d(u,v)<d(p,v)），
> 没被prune的点如果此时小于p的最远neighbor的距离就可以被插入，并且插入后也不会影响到
> 之前已经处理完的prune的正确性，插入后的点会在后续被作为待检查的prune点（即u的角色）
> 去展开后续的prune。这个过程中，你需要按照Raft，Tagore以及我现在的实现，高效用tensor
> core的GEMM计算来高效完成上述计算。

第二次说明（2026-10-21，回答实现细节）：

> 2. 按照CAGRA原来的实现比吧（如果原来一个点的剪枝和所有的点比了，你现在就也和所有的点比，
> 否则就只和u比）。如果确实原来是和整个 S 比，那就把只和 u 比作为更便宜的变体单独测；
> 3. "假设不成立时（v 其实比 u 离 p 更近）：照你的假设仍然放在 u 后面，不回头改已经做完的
> 决定。这种情况占多大比例会统计进 log"按你说的做；
> 4. CAGRA有做这一步吗？如果CAGRA是这么做的，你也这么做；如果CAGRA不满 32 个 neighbor的
> 时候不补齐，那你也不补齐；
> 5. 是的，默认只展开标记为 new 的邻居，并且也有最多的SAMPLE 个（默认和tagore的32设置为
> 一样吧）；按 bucket 共享窗口，不在窗口里就不做这一部分的local join；另外，这里只统计
> forward edge（把reverse edge作为单独的变体测试，因为我感觉在我这里只做forward edge才是
> 几乎没有成本的）；统计实际展开量&point2假设不成立的比例，作为参数输出，之后我的实验数据
> 可能会需要用到

### 2.2 CAGRA 原来怎么做（决定 2.4 的默认值）

cuVS 26.12 `cpp/src/neighbors/detail/cagra/graph_core.cuh::kern_prune`（约 125–210 行），
我们的 `prune_windowed_core.hpp::compute_detour_counts` + `select_pruned_row` 是它的
windowed CPU 版：
- **和所有更近的候选比**：对 p 的每条边 p→B（rank kAB），遍历**所有** rank 更小的候选
  D（kAD < kAB，不管 D 自己最后保没保留），只要 B 出现在 D 的邻居行里就
  `num_detour[kAB]++`。不是只和已保留的集合比，也不是只和一个 u 比。
- **一定补齐到 32**：没有"保留 / 剪掉"的二值判断；按 detour 数从小到大（同数按 rank）
  选出 `output_graph_degree` = 32 条边。detour 为 0 的边不够 32 条时，自动用 detour
  最少的边补齐（只有候选行里有效 id 不够时才会不满）。
- 注意 detour 的判定条件：CAGRA 用的是 rank / 邻接（B ∈ N(D)），用户的融合版本按原话
  用 RNG 的距离条件（d(p,u)<d(p,v) 且 d(u,v)<d(p,v)）。计数对象（所有更近的候选）和选边方式
  （按 detour 数选 32 条、补齐）照 CAGRA。

### 2.3 融合后的逐点算法（对一个点 p）

记号：
- P = p 的候选表，按 d(p,·) 升序，容量 64（`NEIGHBORS_M`）；
  τ = d(p, P 中最远的点)（"p 的最远 neighbor 的距离"）。
- cnt[v] = v 的 detour 数（CAGRA 的 `num_detour`），这里一次 detour 指 RNG 条件成立：
  存在更近的候选 u，满足 d(p,u) < d(p,v) 且 d(u,v) < d(p,v)。

按 rank 从小到大扫描 P，当前位置的候选是 u（**每个候选都当一次 u**，和 CAGRA 每个 D
都读一行一样，不管它自己的 cnt 是多少）：
1. **对 P 中排在 u 后面的已有候选 v**：若 d(u,v) < d(p,v)，cnt[v]++。
2. **local join 展开**（只用 u 的 forward 邻居）：Vnew = N(u) 中标记为 new、
   在当前 bucket 窗口里、且不在 P 中的点，最多取 SAMPLE = 32 个。对每个 v ∈ Vnew：
   - 按用户的假设，v 视为排在 u 后面。如果实际 d(p,v) < d(p,u)（假设不成立），
     仍然放在 u 后面，不回头修改已经做完的部分，只计数（见 2.6）。
   - 计算 v 的 detour 数：和 CAGRA 一样，对**所有 rank 不大于 u 的候选** c（包括 u）
     检查 d(c,v) < d(p,v)，得到 cnt[v]（默认版本）。
   - **插入条件**：cnt[v] = 0（没有被任何更近的候选 prune）且 d(p,v) < τ →
     按 d(p,v) 插入 P 中 u 之后的对应位置（但不早于 u 之后），P 满时踢掉最远的点，
     更新 τ。被 prune 的或不够近的 v 丢弃。
   - 被插入的 v 在后续扫描到它时作为 u 的角色，继续剪后面的点、展开它自己的 N(v)；
     排在它和 u 之间、以及它之后的候选也会照第 1 步给它和别人加 cnt。
3. 前进到下一个位置，直到扫完 P。
4. **选边**：和 CAGRA 一样按 (cnt, rank) 从小到大选 32 条，补齐到 32。
   之后照旧加反向边（`optimize_chunked -m E`）。

**正确性**（用户的论证）：一个候选的 detour 只来自排在它前面的候选；新插入的 v 一律
排在当前 u 之后，所以已经处理完的位置（≤ u）的计数不会被改变。

### 2.4 变体（每个单独测，单独记录）

| 变体 | 和默认的区别 | 动机 |
|---|---|---|
| **默认** | 新候选 v 和所有 rank ≤ u 的候选比（CAGRA 语义）；只展开 forward 邻居；SAMPLE = 32；只在窗口内 | 用户要求按 CAGRA 原来的实现比 |
| **only-u** | 新候选 v 只和 u 比：d(u,v) < d(p,v) 就丢弃，否则（d(p,v) < τ 时）插入，初始 cnt = 0 | 更便宜：不用 [P_≤u] × Vnew 的 GEMM，只要 [p; u] × Vnew |
| **+reverse** | 展开时 Vnew 也包括 u 的反向邻居 R(u)（TAGORE `reverse_graph` 的做法） | 用户：只做 forward 才几乎没有成本；reverse 要额外维护反向表，作为单独变体测 |
| SAMPLE 调参 | SAMPLE ≠ 32 | 只在默认版本有结果之后再试 |

### 2.5 用 tensor core GEMM 高效完成（参考 raft / TAGORE / 现有实现）

把逐点的串行扫描拆成"批量算距离（GEMM）+ 便宜的串行计数"：

- **距离都写成 GEMM**：d(a,b)=‖a‖²+‖b‖²−2a·b，范数每个点只算一次
  （TAGORE `cal_power`；bucket2 Step 6 已经有 norms + fp16 GEMM）。fp16 输入，fp32 累加。
- **P 内部的距离**：p 的候选表 [p; P]（65×128）gather 进 shared memory，算一个
  65×64 的 Gram 矩阵（wmma 16×16 tile，补齐到 80×64；TAGORE `nn_descent_opt_cal`）。
  第 1 步的 RNG 计数变成在 shared memory 里查这个矩阵：一个 warp 对每个 u 并行检查
  后面所有 v，和 raft `kern_prune` 的计数循环同构，只是"v ∈ N(u)"换成距离比较。
- **展开时的距离**：u 的 Vnew（≤ 32，去重后）做一次 GEMM：行 = [p; P 中 rank ≤ u 的候选]，
  列 = Vnew，一次得到 d(p,v) 和所有 d(c,v)，直接给出默认版本的 cnt[v]。
  only-u 变体只需要行 = [p; u]。
  也可以批量：先把所有 u 的 Vnew 合并去重成 Vall，一次算 [p; P] × Vall，按 rank 掩码使用；
  插入的 v 之后作为 u 时需要的 d(v, 后面的新候选) 再补小 GEMM。哪种快用实测决定。
- **按 bucket 批量处理**：同一个 bucket 的点 p 共享同一个 bucket 窗口
  （现在的 `BucketWindow` / `BeladyBucketCache`），邻居和邻居的邻居大多在窗口里。
  一个 thread block 处理一个或几个 p；窗口向量常驻 GPU（受现有 GPU 预算约束），
  gather 留在 L2 / shared memory；不在窗口里的点不做这部分 local join。
  和 bucket2 out-of-core Step 6 的 slot / pinned 双缓冲是同一套机制。
- **new 标记**：每个点的邻居行带 new/old 位（TAGORE `nei_visit`）。bucket2 刚产出的
  邻居都是 new；某条边被展开过后标成 old，避免同一对被重复展开。
- **去重**：Vnew 和 P 的去重用 shared memory 里的 bitonic sort + 二分
  （TAGORE `bitonic_sort_id_new2`），或 `compute_detour_counts` 式的小 open-addressing 表。
- **阈值先过滤**：先和 τ 比，不进榜的点不参与后面的计数和排序（TAGORE `min_ele`）。

### 2.6 统计量（必须作为参数输出，用户的实验数据要用）

每次跑大路径时，融合 prune 的 stage 在日志里打印一行 `A1_stats:`（key=value），
`summarize.py` 把它收进 `metrics.json` 的 `a1_stats`，每个 A1 迭代的 log 条目都要贴：

| key | 含义 |
|---|---|
| `points` | 处理的点数 |
| `u_expanded_avg` | 每个 p 平均有多少个 u 真的做了展开（Vnew 非空） |
| `cand_considered_avg` | 每个 p 平均展开出的候选数（new、去重后、截断 SAMPLE 前 / 后各一个） |
| `cand_out_of_window_avg` | 每个 p 平均因为不在窗口里被跳过的候选数 |
| `dist_computed_avg` | 每个 p 平均为展开额外计算的距离数（GEMM 元素数） |
| `inserted_avg` | 每个 p 平均插入 P 的新点数 |
| `pruned_new_avg` | 每个 p 平均因 cnt > 0（或 only-u 里被 u prune）丢弃的新候选数 |
| `assumption_violated_ratio` | 假设不成立的比例：展开出的候选里 d(p,v) < d(p,u) 的占比（同时给出分子、分母） |
| `final_from_join_ratio` | 最终 32 条边中来自 local join 的比例 |
| `variant`、`sample` | 变体名和 SAMPLE 值 |

resident 路径（§1）打印同样格式的 `A1_stats:`，至少包括每趟的插入数、计算距离数、
被采样的 new / old 数。

### 2.7 代价估算（实现前先核对）

- 默认版本每个 p：65×64×128 的 Gram + 最多 64 次 (≤65)×32×128 的展开 GEMM，
  约 35 MFLOP；1M 点约 35 TFLOP，V100 tensor core 上是秒级。only-u 便宜得多。
- 真正的风险是 **gather 流量**：64 个 u × 32 个邻居 × 128 B（u8）≈ 256 KB/点，
  1M 点 ~256 GB。只展开 new、SAMPLE = 32、只在窗口内、按 bucket 共享窗口这几条
  是为了把它压下来。先用 SIFT1M 打印 2.6 的统计量，看实际展开量。

### 2.8 建议的迭代拆分

一次迭代只做一个可归因的改动；中间步骤单独过不了 judge 时，和下一步合在同一个迭代里做：
1. **A1-R**：resident 路径的 GNN-Descent（§1），1–2 趟；再配合降低 NPROBE / ITERATIONS。
2. **A1-L1**：大路径 prune 搬到 GPU：窗口内 Gram GEMM + RNG detour 计数 + 按 (cnt, rank)
   选 32 条，还不展开。
3. **A1-L2**：在 A1-L1 上加 local join 展开（默认变体），再配合降低 NPROBE / ITERATIONS。
4. **A1-L2 变体**：only-u、+reverse，各一个迭代。

每一步都记录：时间（所有 stage）、pre-prune KNN recall、最终 recall@10 + QPS、
大路径内存、2.6 的全部统计量。
