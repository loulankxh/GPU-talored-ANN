# bucketDemo 编译 / 运行 / 测试指南

## 0. 环境准备

```bash
conda activate rapids_raft
which nvcc && echo $CONDA_PREFIX   # 确认工具链存在
```

## 1. 编译

`buildBucket` 目录下已有一份陈旧的 `CMakeCache.txt`（源目录记录的是上层 `bucketDemo`，和这份 `CMakeLists.txt` 对不上），建议另起一个干净的 `build/` 目录，也是 `quick_test.sh` / `run_test.sh` 期望的路径：

```bash
cd bucketDemo/buildBucket
mkdir -p build && cd build
cmake ..
make -j$(nproc)          # 编译全部 target: bucket, bucket2, optimize, optimize_chunked, reorder, prune_windowed, translate_ids
# 或只编译某一个:
make -j$(nproc) bucket2
```

`search` 这个可执行文件是上一级 `bucketDemo/CMakeLists.txt` 里定义的（`search.cpp`），要单独编译：

```bash
cd bucketDemo
mkdir -p build && cd build
cmake ..
make -j$(nproc) search
```

## 2. 生成测试数据

```bash
cd bucketDemo/buildBucket
python3 generate_test_data.py
# 生成 test_data/vectors_{10k,100k,1m}_128d.fbin
```

## 3. 各可执行文件：命令 + 参数说明

每个都有 `--help`，先看一下最保险：`./build/<bin> --help`

参数表里"必需/可选"指该 flag 是否必须显式给值；"默认值"是不给时程序用的值。

### bucket（源码 `bucket_build.cu`）

```bash
./build/bucket \
    --data test_data/vectors_100k_128d.fbin \
    --k 128 --m 32 \
    --out_dir output/my_test \
    --seed 42 \
    --kmeans-iters 5 \
    --balance-slack 4 \
    --init-method kmeans \
    --t 1 \
    --cpu-limit $((16*1024*1024*1024)) \
    --gpu-limit $((2*1024*1024*1024)) \
    --sample-rate 0.1 \
    --centroid-ratio 0.01 \
    --pq-bits-start 8 --pq-bits-min 1
```

| 参数 | 必需/可选（默认值） | 说明 |
|---|---|---|
| `--data` | 必需 | 输入数据文件（.fbin/.ibin/.i8bin） |
| `--k` | 必需 | 桶（centroid）数量 |
| `--m` | 必需 | 每个点保留的近邻数 M |
| `--out_dir` | 必需 | 输出目录 |
| `--seed` | 可选（0） | 随机种子 |
| `--kmeans-iters` | 可选（5） | KMeans 迭代次数 |
| `--no-balance` | 可选（不加=启用均衡） | 关闭桶再均衡（开关，无值） |
| `--balance-slack` | 可选（4） | 均衡松弛度，桶大小允许偏离平均值的量 |
| `--init-method` | 可选（kmeans） | 初始化方式：`kmeans` / `kmeans-fast` / `random` |
| `--t` | 可选（1） | Step2-6 重复轮数（每轮不同 seed），去重合并 KNN |
| `--cpu-limit` | 可选（16GB） | CPU 内存限制（字节） |
| `--gpu-limit` | 可选（4GB） | GPU 内存限制（字节） |
| `--sample-rate` | 可选（0.1） | 采样比例 [0,1] |
| `--centroid-ratio` | 可选（0.01） | 桶心比例 [0,1]（相对全量数据），决定桶数/平均桶大小 |
| `--use-pq` | 可选（false） | 强制 PQ 量化（开关，无值） |
| `--pq-bits-start` | 可选（8） | PQ 起始比特数（原限制 [4,8]，本版本不再限制） |
| `--pq-bits-min` | 可选（4） | PQ 最小比特数（原限制 [4,8]，本版本不再限制，可到 1） |
| `--pq-train-fraction` | 可选（0.05） | PQ 训练数据比例 [0,1] |
| `--pq-train-max-rows` | 可选（65536） | PQ 训练数据行数上限 |
| `--order-window` | 可选（0=自动，4*k） | Step4 桶处理顺序的滑动窗口大小 |
| `--cache-mb` | 可选（0=自动，cpu-limit/2） | Step4 桶向量缓存预算（MB） |

### bucket2（源码 `bucket.cu`，tensor core matmul + GPU KMeans++ 版）

```bash
./build/bucket2 \
    -i test_data/vectors_100k_128d.fbin \
    -o output/bucket2_test \
    --cpu-limit $((16*1024*1024*1024)) \
    --gpu-limit $((2*1024*1024*1024)) \
    --sample-rate 1.0 \
    --centroid-ratio 0.01 \
    --use-pq false \
    --pq-bits-start 8 --pq-bits-min 4 \
    --seed 42 \
    --knn-k 32 \
    --nprobe 32 \
    --search-iters 64 \
    --neighbors-m 32 \
    --t 1 \
    --order-window 0
```

> **`--reorder` 已被移除**：按桶重排现在在 round 1 的 Step 4.5 里**无条件执行**（不再是可选开关），round 2..t 和最终合并出的 `vector_knn.bin` 从诞生起就已经是重排后坐标系，不需要也不能再传 `--reorder`。

| 参数 | 必需/可选（默认值） | 说明 |
|---|---|---|
| `-i, --input` | 必需 | 输入数据文件（.fbin/.bin/.u8bin/.i8bin/.ibin/.ubin） |
| `-o, --output` | 必需 | 输出根目录；实际产出文件会写在其下自动创建的 `k<knn-k>p<nprobe或knn-k>m<neighbors-m>t<iterations>/` 子目录里（例如下面示例参数会产出 `output/bucket2_test/k32p32m32t1/`），因为这几个参数都会实质性影响 `vector_knn.bin` 的内容，编码进目录名可以避免不同参数组合的跑批互相覆盖 |
| `--cpu-limit` | 可选（16GB） | CPU 内存限制（字节） |
| `--gpu-limit` | 可选（0=自动检测可用显存的 95%） | GPU 内存限制（字节） |
| `--bucket-vec-buffer` | 可选（1GB） | Step4 按 bucket 缓存原始向量 (BucketVectorAccumulator) 的写缓冲区**总**预算（字节），会按 n_centroids 平摊到每个桶，不是每个桶单独这么多；数据集越大、桶数越多，单个桶分到的缓冲区越小 |
| `--sample-rate` | 可选（0.1） | 采样比例 [0,1] |
| `--centroid-ratio` | 可选（0.01） | 桶心比例（相对采样数据），决定桶数/平均桶大小 |
| `--use-pq` | 可选（false） | 强制 PQ 量化 —— **注意**：这里是 `po::value<bool>`，必须显式传值，如 `--use-pq true`（不同于 `bucket` 的开关式 `--use-pq`） |
| `--pq-bits-start` | 可选（8） | PQ 起始比特数 |
| `--pq-bits-min` | 可选（4） | PQ 最小比特数 |
| `--seed` | 可选（42） | 随机种子 |
| `--knn-k` | 可选（32，范围 [1,1000]） | 桶（centroid）间 CAGRA 导航图度数 K |
| `--nprobe` | 可选（0=沿用 knn-k） | Step6 里每个点扩展检索的邻近桶数量。该图是 CAGRA 剪枝后的导航图、不是排序好的真 KNN 表，所以即使 `nprobe == knn-k`，代码也会在图上做一次真正的 greedy graph search 得到近似最近邻，而不是直接截断已有邻居列表 |
| `--search-iters` | 可选（64） | 图搜索最大迭代次数（当前实现中 CAGRA 自行决定迭代数，该参数暂未生效，仅接口保留） |
| `--neighbors-m` | 可选（0=跳过 Step6） | 每个点的 per-vector KNN 近邻数 M |
| `--t, --iterations` | 可选（1） | Step2-6 重复轮数（每轮不同 seed），按 per-vector KNN 去重合并；桶文件（Step5）只反映最后一轮 |
| `--order-window` | 可选（0=自动，4*knn-k） | round 1 按桶重排处理顺序的滑动窗口大小（DiskJoin 风格，始终会算，不再受开关控制）；Step6 内部也会用同一个值驱动下面 `--cache-mb` 的 Belady 缓存排序 |
| `--cache-mb` | 可选（0=自动，cpu-limit/2） | Step6 的 Belady 桶向量读缓存预算（MB）。原理见 `bucket_order.hpp`（DiskJoin 风格任务排序 + 最优离线缓存）：按处理顺序把"桶自己 + 近邻桶"的原始向量缓存住，命中就不用再碰磁盘；跟 `bucket_build.cu` 的 `--cache-mb` 是同一套机制 |

**输出文件**（都在 `-o` 下自动创建的 `k<knn-k>p<nprobe>m<neighbors-m>t<iterations>/` 子目录里，程序结束时也会原样打印一份）：

| 文件 | 说明 |
|---|---|
| `data_reordered.<ext>` | round-1 按桶重排后的数据集（保留输入格式） |
| `bucket_offsets.bin` | 桶边界（新 ID 空间），`optimize_chunked --method C/D/E`、`prune_windowed` 用 |
| `bucket_process_order.bin` | 桶处理顺序，`prune_windowed` 用 |
| `inverse_perm.bin` | ID 翻译表（新 id → 原始 id），见下面 `translate_ids` |
| `centroid_knn.bin` | round-1 centroid 间 KNN 图（K=`--knn-k`），`prune_windowed` 用 |
| `bucket_index.bin` / `bucket_data.bin` | 最后一轮的桶成员表（诊断用；里面的 id 已经是新/重排后坐标系） |
| `centroid_global_indices.bin` | 最后一轮的 centroid→数据点映射（诊断用） |
| `vector_knn.bin` | 仅 `--neighbors-m > 0` 时输出；逐点 M-NN 索引，**已经是重排后坐标系**，可以直接当 `prune_windowed` 的 `--vector-knn-reordered` 用，不需要再跑 `reorder` 工具 |
| `neighbors.npy` | 仅 `--neighbors-m > 0` 时输出；跟 `vector_knn.bin` 同内容，NumPy 格式，给 `search.py` 用 |

### optimize（CAGRA 图剪枝，吃 bucket2 产出的 `vector_knn.bin`）

```bash
./build/optimize \
    -i test_data/vectors_100k_128d.fbin \
    -g output/bucket2_test/k32p32m32t1/vector_knn.bin \
    -o output/bucket2_test/k32p32m32t1/cagra_graph.bin \
    --output-degree 32
```

| 参数 | 必需/可选（默认值） | 说明 |
|---|---|---|
| `-i, --input` | 必需 | 原始数据集（.fbin/.bin/.u8bin/.i8bin/.ibin），`sort_knn_graph` 算 L2 距离要用 |
| `-g, --knn-graph` | 必需 | `bucket.cu` 产出的 `vector_knn.bin`（中间 KNN 图） |
| `-o, --output` | 必需 | 输出的剪枝后 CAGRA 图（`cagra_graph.bin`） |
| `--output-degree` | 可选（32） | 目标输出图度数 K_out，必须 ≤ 输入图的 M（`--m`/`--neighbors-m`） |
| `--skip-sort` | 可选（false） | 跳过 `sort_knn_graph`，假定输入已按 L2 排序（开关，无值） |
| `--save-npy` | 可选（false） | 同时写一份 `.npy`（int64，无 padding）（开关，无值） |

### optimize_chunked（超大图分块反向边合并，5 种方法）

```bash
# 方案3 = Method C，需要 bucket_offsets.bin（bucket2 现在直接产出）
./build/optimize_chunked \
    -g output/bucket2_test/k32p32m32t1/cagra_graph.bin \
    -o output/bucket2_test/k32p32m32t1/cagra_graph_merged.bin \
    -m C \
    --bucket-offsets output/bucket2_test/k32p32m32t1/bucket_offsets.bin \
    --gpu-budget-mb 8000 \
    --rev-degree-cap 0

# Method E：前向图本身大到无法常驻内存时用，全程磁盘流式读写
./build/optimize_chunked \
    -g output/bucket2_test/k32p32m32t1/cagra_graph.bin \
    -o output/bucket2_test/k32p32m32t1/cagra_graph_merged.bin \
    -m E \
    --bucket-offsets output/bucket2_test/k32p32m32t1/bucket_offsets.bin \
    --gpu-budget-mb 8000 \
    --edge-buffer-mb 1024
```

| 参数 | 必需/可选（默认值） | 说明 |
|---|---|---|
| `-g, --forward-graph` | 必需 | 输入的剪枝后前向图（`cagra_graph.bin` 格式，或 `prune_windowed` 产出的 `forward_graph_reordered.bin`） |
| `-o, --output` | 必需 | 反向边合并后的输出图 |
| `-m, --method` | 可选（"A"） | 方法：A（kernel-filter，多轮 PCIe）/ B（预先按目的端分桶）/ C（bucket-aligned 单遍扫描 + overflow，无损）/ D（3-slot 滑动窗口，直接捕获 ±1 跨 chunk 边，无损）/ E（按 chunk 磁盘暂存的 scatter buffer，保留 rank 优先级，单遍扫描，前向图和输出都流式读写磁盘——不要求整张图常驻内存，是 A/B/C/D 都做不到这点时的退路） |
| `--gpu-budget-mb` | 可选（8000） | chunk 缓冲区的 GPU 显存预算（MB） |
| `--edge-buffer-mb` | 仅 Method E 用（可选，1024） | 所有 chunk 共享的 host 端暂存总预算（MB），语义跟 `bucket2` 的 `--bucket-vec-buffer` 一样 |
| `--tmp-dir` | 仅 Method E 用（可选，""） | per-chunk 边文件的临时目录（默认 `<output 所在目录>/.optimize_chunked_e_tmp`，成功后自动删除） |
| `--bucket-offsets` | 方法 C/D 必需，E 可选，其余不用（""） | 桶边界文件（`bucket2` 直接产出）。E 给了就按桶对齐 chunk，不给则退化成按 GPU 预算均匀分 chunk（E 的正确性本身不依赖桶局部性） |
| `--rev-degree-cap` | 可选（0=沿用输入图自带的 K_out） | 每个节点允许的反向边数上限，合并时保护前 K_out/2 条原始正向边；Method E 不用这个参数，它保留的是可变长候选表而不是固定上限 |
| `--save-npy` | 可选（false） | 同时写一份 `.npy`（开关，无值） |

### reorder（CPU-only，按桶重排数据，供 `optimize_chunked --method C/D` 用）

> **`bucket2` 现在已经自带 round-1 内联重排**（`data_reordered.*`/`bucket_offsets.bin`/`bucket_process_order.bin`/`centroid_knn.bin`/已重排的 `vector_knn.bin` 都会直接产出），所以这个独立工具主要用在：(a) 处理 `bucket`（非 tensor core 版，`bucket_build.cu`）的输出——它自己不做重排；(b) 需要脱离 `bucket2` 独立重跑一次重排的场景。如果数据是 `bucket2` 产出的，通常不需要再跑这一步。

```bash
./build/reorder \
    -i test_data/vectors_100k_128d.fbin \
    -x output/bucket2_test/k32p32m32t1/bucket_index.bin \
    -d output/bucket2_test/k32p32m32t1/bucket_data.bin \
    -k output/bucket2_test/k32p32m32t1/vector_knn.bin \
    -c output/bucket2_test/k32p32m32t1/centroid_knn.bin \
    --order-window 0 \
    -o output/bucket2_test/k32p32m32t1/reordered
```

| 参数 | 必需/可选（默认值） | 说明 |
|---|---|---|
| `-i, --input` | 必需 | 原始数据集（.fbin/.bin/.u8bin/.i8bin/.ibin） |
| `-x, --bucket-index` | 必需 | `bucket.cu` 产出的 `bucket_index.bin` |
| `-d, --bucket-data` | 必需 | `bucket.cu` 产出的 `bucket_data.bin` |
| `-k, --vector-knn` | 可选（""=不处理） | `vector_knn.bin`，给了就会重排 + 邻居 ID 重映射，产出 `vector_knn_reordered.bin` |
| `-c, --centroid-knn` | 可选（""=退化为原始桶编号顺序） | `centroid_knn.bin`；给了则按 DiskJoin 风格算出有空间局部性的桶处理顺序，让 `optimize_chunked --method C/D` 的 chunk 边界更贴合数据分布 |
| `--order-window` | 可选（0=自动，4*K） | `--centroid-knn` 桶排序用的滑动窗口大小 |
| `-o, --output-dir` | 必需 | 输出目录 |

### prune_windowed（CPU-only，内存受限场景下的近似 CAGRA 剪枝）

数据集太大、候选图无法像 raft 自己的实现那样整份常驻 GPU/host 时的替代方案：按桶处理顺序 + 滑动可见窗口（中心桶 + centroid KNN 邻居桶，经 Belady 最优缓存），窗口外的候选邻居保守地当作"无 detour"处理（只会少剪，不会错剪）。输入默认接 `bucket2` 自己的输出（已经是重排坐标系，不需要先跑 `reorder`）：

```bash
./build/prune_windowed \
    -k output/bucket2_test/k32p32m32t1/vector_knn.bin \
    -i output/bucket2_test/k32p32m32t1/data_reordered.fbin \
    --bucket-offsets output/bucket2_test/k32p32m32t1/bucket_offsets.bin \
    --bucket-process-order output/bucket2_test/k32p32m32t1/bucket_process_order.bin \
    -c output/bucket2_test/k32p32m32t1/centroid_knn.bin \
    -o output/bucket2_test/k32p32m32t1/forward_graph_reordered.bin \
    --output-degree 32 \
    --cache-mb 4000
```

| 参数 | 必需/可选（默认值） | 说明 |
|---|---|---|
| `-k, --vector-knn-reordered` | 必需 | 重排坐标系下的逐点 KNN 表——`bucket2` 的 `vector_knn.bin`，或 `reorder` 工具产出的 `vector_knn_reordered.bin` |
| `-i, --dataset` | 必需 | 重排后的数据集 `data_reordered.<ext>`（`bucket2` 或 `reorder` 产出） |
| `--bucket-offsets` | 必需 | `bucket_offsets.bin` |
| `--bucket-process-order` | 必需 | `bucket_process_order.bin` |
| `-c, --centroid-knn` | 必需 | `centroid_knn.bin` |
| `-o, --output` | 必需 | 剪枝后的前向图 `forward_graph_reordered.bin`（格式跟 `optimize_chunked` 的输入一致，可直接接 `-m E`，或配合同一份 `--bucket-offsets` 接 `-m C/D`） |
| `--output-degree` | 可选（32） | 目标输出度数 K_out，须 ≤ 输入度数 M |
| `--order-window` | 可选（0=自动，4*centroid-knn 的 K，最小 16） | 可见窗口大小（按桶数算）；**必须**跟产出 `bucket_process_order.bin` 时用的 `--order-window` 一致，否则窗口和实际处理顺序会错位 |
| `--cache-mb` | 可选（4000） | Belady 桶缓存（候选行 + 原始向量）的 host 内存预算（MB） |
| `--save-npy` | 可选（false） | 同时写一份 `.npy`（开关，无值） |

### translate_ids（把重排坐标系下的图/索引翻译回原始数据集顺序）

`bucket2` 的内联重排只保留 `inverse_perm.bin`（新 id → 原始 id），不写 `perm.bin`。要把一份最终图（`optimize_chunked` 的输出、`vector_knn.bin`、或 `neighbors.npy`）里的行位置和每个邻居 id 都换算回原始 id（比如要拿去跟原始顺序的 ground truth 比较），用这个工具。格式按 `-g` 的文件扩展名自动判断：

```bash
# .bin 格式（cagra_graph.bin / cagra_graph_merged.bin / vector_knn.bin / forward_graph_reordered.bin 都是同一种 forward-graph 格式）
./build/translate_ids \
    -g output/bucket2_test/k32p32m32t1/cagra_graph_merged.bin \
    -p output/bucket2_test/k32p32m32t1/inverse_perm.bin \
    -o output/bucket2_test/k32p32m32t1/cagra_graph_original_order.bin

# .npy 格式（neighbors.npy）
./build/translate_ids \
    -g output/bucket2_test/k32p32m32t1/neighbors.npy \
    -p output/bucket2_test/k32p32m32t1/inverse_perm.bin \
    -o output/bucket2_test/k32p32m32t1/neighbors_original_order.npy
```

| 参数 | 必需/可选（默认值） | 说明 |
|---|---|---|
| `-g, --graph` | 必需 | 重排坐标系下的最终图/索引；`.npy` 扩展名走 `neighbors.npy` 那种 `[N,K]` int64 格式（`-1` 当 padding，保留不翻译），其它扩展名走裸二进制的 forward-graph 格式 |
| `-p, --inverse-perm` | 必需 | `inverse_perm.bin`（`bucket2` 产出，`inverse_perm[new_id]=old_id`） |
| `-o, --output` | 必需 | 输出图，同样格式，行位置和每个邻居 id 都换算回原始 id |

### search（在 `bucketDemo/` 下编译，评估召回率 / 对比两份索引）

```bash
# search 子命令：算召回率
./build/search search \
    --base test_data/vectors_100k_128d.fbin \
    --query <query_file> \
    --index output/bucket2_test/k32p32m32t1/neighbors.npy \
    --gt <ground_truth.ibin> \
    --topk 10 --ef 200 --num-entry 4

# compare 子命令：对比两份索引结果的重合度
./build/search compare \
    --index1 <a.npy> --index2 <b.npy> \
    --k-cmp 10 --save-csv diff.csv
```

**search 子命令**

| 参数 | 必需/可选（默认值） | 说明 |
|---|---|---|
| `--base` | 必需 | base 向量文件 |
| `--query` | 必需 | query 向量文件 |
| `--index` | 必需 | 索引搜索结果（.npy） |
| `--gt` | 必需 | ground truth（.ibin） |
| `--topk` | 可选（10） | 评估的 top-K |
| `--ef` | 可选（200） | efSearch 搜索宽度 |
| `--num-entry` | 可选（4） | 随机入口点数量 |
| `--normalize` | 可选（false） | L2 归一化，用于余弦搜索（开关，无值） |

**compare 子命令**

| 参数 | 必需/可选（默认值） | 说明 |
|---|---|---|
| `--index1` | 必需 | 索引结果 1（.npy） |
| `--index2` | 必需 | 索引结果 2（.npy） |
| `--k-cmp` | 可选（-1=全部） | 比较的近邻数量 |
| `--save-csv` | 可选（不保存） | 对比结果保存到指定 CSV 路径 |

## 4. 一键测试脚本

```bash
cd bucketDemo/buildBucket

# 最简单：生成数据 + 编译 + 跑一次 100K 向量测试
bash quick_test.sh

# 更全面：3 个规模（10K/100K/1M）的场景测试
bash run_test.sh

# 端到端：编译 bucket2 + search，构建索引，可选做召回率评估
bash run_build_and_search.sh -b test_data/vectors_100k_128d.fbin
bash run_build_and_search.sh -b <base.fbin> --search -q <query.fbin> -g <gt.ibin>
```

## 5. 性能测试

```bash
/usr/bin/time -v ./build/bucket \
    --data test_data/vectors_100k_128d.fbin \
    --k 128 --m 32 --out_dir output/perf_test --sample-rate 0.1
```
