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
make -j$(nproc)          # 编译全部 target: bucket, bucket2, optimize, optimize_chunked, reorder
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
    --reorder \
    --order-window 0
```

| 参数 | 必需/可选（默认值） | 说明 |
|---|---|---|
| `-i, --input` | 必需 | 输入数据文件（.fbin/.bin/.u8bin/.i8bin/.ibin/.ubin） |
| `-o, --output` | 必需 | 输出目录 |
| `--cpu-limit` | 可选（16GB） | CPU 内存限制（字节） |
| `--gpu-limit` | 可选（0=自动检测可用显存的 95%） | GPU 内存限制（字节） |
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
| `--reorder` | 可选（false） | 额外输出按桶重排的文件（`data_reordered.*`、`vector_knn_reordered.bin`、`bucket_offsets.bin`、`perm.bin`、`inverse_perm.bin`），供 `optimize_chunked --method C/D` 使用（开关，无值） |
| `--order-window` | 可选（0=自动，4*knn-k） | `--reorder` 用的桶处理顺序滑动窗口大小 |

### optimize（CAGRA 图剪枝，吃 bucket2 产出的 `vector_knn.bin`）

```bash
./build/optimize \
    -i test_data/vectors_100k_128d.fbin \
    -g output/bucket2_test/vector_knn.bin \
    -o output/bucket2_test/cagra_graph.bin \
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

### optimize_chunked（超大图分块反向边合并，4 种方法）

```bash
# 方案3 = Method C，需要先跑过 --reorder 或 reorder 得到 bucket_offsets.bin
./build/optimize_chunked \
    -g output/bucket2_test/cagra_graph.bin \
    -o output/bucket2_test/cagra_graph_merged.bin \
    -m C \
    --bucket-offsets output/bucket2_test/bucket_offsets.bin \
    --gpu-budget-mb 8000 \
    --rev-degree-cap 0
```

| 参数 | 必需/可选（默认值） | 说明 |
|---|---|---|
| `-g, --forward-graph` | 必需 | 输入的剪枝后前向图（`cagra_graph.bin` 格式） |
| `-o, --output` | 必需 | 反向边合并后的输出图 |
| `-m, --method` | 可选（"A"） | 方法：A（kernel-filter，多轮 PCIe）/ B（预先按目的端分桶）/ C（bucket-aligned 单遍扫描 + overflow，无损）/ D（3-slot 滑动窗口，直接捕获 ±1 跨 chunk 边，无损） |
| `--gpu-budget-mb` | 可选（8000） | chunk 缓冲区的 GPU 显存预算（MB） |
| `--bucket-offsets` | 方法 C/D 必需，其余可选（""） | 桶边界文件（来自 `bucket2 --reorder` 或独立的 `reorder` 工具） |
| `--rev-degree-cap` | 可选（0=沿用输入图自带的 K_out） | 每个节点允许的反向边数上限，合并时保护前 K_out/2 条原始正向边 |
| `--save-npy` | 可选（false） | 同时写一份 `.npy`（开关，无值） |

### reorder（CPU-only，按桶重排数据，供 `optimize_chunked --method C/D` 用）

```bash
./build/reorder \
    -i test_data/vectors_100k_128d.fbin \
    -x output/bucket2_test/bucket_index.bin \
    -d output/bucket2_test/bucket_data.bin \
    -k output/bucket2_test/vector_knn.bin \
    -c output/bucket2_test/centroid_knn.bin \
    --order-window 0 \
    -o output/bucket2_test/reordered
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

### search（在 `bucketDemo/` 下编译，评估召回率 / 对比两份索引）

```bash
# search 子命令：算召回率
../search search \
    --base test_data/vectors_100k_128d.fbin \
    --query <query_file> \
    --index output/bucket2_test/neighbors.npy \
    --gt <ground_truth.ibin> \
    --topk 10 --ef 200 --num-entry 4

# compare 子命令：对比两份索引结果的重合度
../search compare \
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
