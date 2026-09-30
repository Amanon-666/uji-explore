# UJI 跨楼层 few-shot：CMANP 复现与可审计实验基线

**版本：v0.1；建立日期：2026-09-29。** 主线为 ICML 2024 的 **CMANP**，直接复用作者公开代码，不把普通 MLP 冒充元学习实现。

> **交付状态：代码、协议、29 项测试、完整 UJI CPU pilot 已运行；原文 100,000 步 × 5 种子数值复现及正式 UJI 多种子实验尚未完成。**
> pilot 中 CMANP 未优于 10/20-shot 的传统/微调基线，不能据此称它为 UJI SOTA。所有结果都标记 `pilot` 或 `smoke`，没有把调试运行包装成论文复现。

## 从哪里开始

先读 [研究设计](docs/RESEARCH_DESIGN.md)，再读 [实测结果与边界](reports/RESULTS.md)。参数来源见 [参数登记表](docs/PARAMETER_PROVENANCE.md)，原文与代码的对应、差异见 [复现审计](docs/REPRODUCTION_AUDIT.md)。

论文：**Memory Efficient Neural Processes via Constant Memory Attention Block**，ICML 2024，PMLR 235:13365–13386。

- 正式论文入口：<https://proceedings.mlr.press/v235/feng24i.html>
- 全文：<https://arxiv.org/pdf/2305.14567>
- 官方代码：<https://github.com/BorealisAI/constant-memory-anp>
- 固定提交：`8961cd940153d76918f401acc60aa858101a8949`

ICML 是 CCF A 类会议；按正式发表时间 2024 年计算，满足本项目“近三年”的筛选口径。**这篇论文研究通用小样本回归，不是无线定位论文，原文没有 UJI 跨楼层成绩。** 原文的高指标不能直接换算成本项目的定位误差。

## 核心问题

先看其他楼层的历史数据，到了一个训练时没见过的楼层，只知道 **5、10 或 20 个位置各一次扫描的 RSSI 和坐标**，能否预测该楼层其他位置的二维坐标？

CMANP 的方法是：历史楼层上学习“如何利用几条已知样本来预测其他样本”；新楼层的支持样本直接进入网络作为条件。**不进行 MAML 式内循环梯度更新，条件化也属于少样本适应。** 代码另提供 MLP 微调，用真实梯度步骤作对照。

## 安装与运行

建议 Python 3.11。在仓库根目录运行：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/fetch_upstream.py   # 已含完整 third_party 时仅校验，不重复下载
python -m pytest -q
```

PyTorch 的 CPU/CUDA 版本按机器环境安装。本次实测环境为 CPU、PyTorch 2.10.0；GPU 路径未在本次环境实测。依赖的精确实测版本见 `requirements-tested.txt`。GitHub 通过固定提交下载上游、通过固定哈希下载数据；对话压缩包已经包含二者。

**先做工程检查：**

```bash
bash scripts/run_smoke.sh runs/my_smoke
```

**正式 UJI 全流程：**

```bash
bash scripts/run_full.sh runs/full_v1 cuda
```

该命令依次准备数据与冻结划分、在开发楼层选择停止点和基线参数、用十个源楼层重新训练、执行跨楼层测试、执行共同采集群体限制下的 5-shot 对照，并生成逐楼层、宏平均和配对差值。正式配置有 5 个模型种子和最多 100,000 次更新；这不是短跑检查。

**分步运行或断点恢复：**

```bash
python -m uji.run prepare --out runs/full_v1 --profile full
python -m uji.run develop --out runs/full_v1 --device cuda
# 中断时仅恢复同一配置、同一划分、同一代码版本：
python -m uji.run develop --out runs/full_v1 --device cuda --resume
python -m uji.run final --out runs/full_v1 --device cuda --allow-test
python -m uji.run final --out runs/full_v1 --device cuda --allow-test --matched
python scripts/report.py --out runs/full_v1
```

不要先跑完整脚本再重复 prepare 到同一个目录。首次最终测试需显式 `--allow-test`；查看测试结果后程序阻止在同一目录继续开发调参。文件标记用于审计，不是防止人为删改的安全沙箱。

## 原文复现单独运行

```bash
python scripts/reproduce_gp.py --out runs/original_gp \
  --steps 100000 --seeds 0 1 2 3 4 --eval-batches 3000 \
  --kernel-spec code --device cuda
```

原文表 4 的 **CMANP**（不是 CMANP-AND）目标为 RBF 对数似然 **1.24 ± 0.01**、Matern 5/2 **0.80 ± 0.01**。论文正文与代码的 GP 长度尺度范围不同，详见复现审计。`--kernel-spec paper` 是独立敏感性检查，不能与 `code` 混报。

## 实验内容

|实验|回答的问题|数据与标签预算|
|---|---|---|
|Primary|未见整层如何少样本定位新位置？|trainingData 内整层留出；每层 5/10/20 个不同位置、各一次扫描|
|Matched-cohort restriction|限制设备、用户、日期重叠后如何？|共同 `(PHONEID, USERID, UTC日期)` 群体；每层 5 条标签|
|External stress|更晚采集、混合设备的新位置如何？|validationData 仅外部压力测试；排除目标训练位置的精确重复坐标|

Matched 只限制共同群体，未完全平衡群体比例、小时与路径，不代表纯空间变化的因果效应。外部测试也不是纯跨楼层。

## 固定空间划分

```text
开发拟合 7 层：B0F0 B0F1 | B1F0 B1F1 | B2F0 B2F1 B2F2
开发选择 3 层：B0F2      | B1F2      | B2F3
最终目标 3 层：B0F3      | B1F3      | B2F4
冻结后重训：上面的 7 + 3 = 10 个源楼层
```

每个目标楼层按精确坐标分组，固定约 30% 位置为查询，其余为支持池。不同 K 使用嵌套支持集与相同查询。禁止同位置重复扫描跨越两侧；不使用隐藏坐标优化支持点覆盖。

模型输入是 UJI 的同一套 520 个 WAP 身份列。PHONEID、USERID、TIMESTAMP、SPACEID、RELATIVEPOSITION 不进入网络；已知建筑编号只选择源数据拟合的坐标变换。本版不解决未见建筑的任意新 AP 编号。

## 输出与已有结果

运行生成 FROZEN.json、learning_curve.jsonl、episode_metrics.csv、floor_metrics.csv、macro_metrics.csv 和逐查询预测。支持集和模型种子重采样得到的区间固定这 3 个楼层，不能代表任意新楼层总体。覆盖诊断中的隐藏坐标不参与训练和选点。

主 pilot 的三楼层宏平均 MDE（UJI 平面坐标米制口径）：

|方法|5-shot|10-shot|20-shot|
|---|---:|---:|---:|
|CMANP|25.49|25.50|25.49|
|WKNN，开发选择 k=1|27.66|19.34|18.75|
|RBF 核回归|28.75|19.80|18.25|
|Source-only MLP，实际 0 标签|23.99|23.99|23.99|
|MLP-FT，固定 50 步|20.97|16.63|15.82|

这些来自 1 个模型种子、2 个支持种子、CMANP 300 步与 MLP 100 步，只是 pilot。不能称为 SOTA 或正式复现成功。完整逐样本记录、冻结清单、原始数据和 pilot 权重见本次对话交付 ZIP；仓库只保留可运行源码、来源锁定、设计、测试和结果摘要。

推理接口：

```bash
python scripts/predict.py --checkpoint exports/cmanp_pilot.pt \
  --support support.csv --query query.csv --building 0 --output predictions.csv
```

support.csv 需要 WAP001…WAP520 和 LONGITUDE/LATITUDE；query.csv 只需 WAP 列。ZIP 内 exports 仅为未收敛 pilot 权重，不能部署为已验证定位服务。导出推理权重使用 weights_only=True；训练 checkpoint 只加载受信任文件。

## 许可

上游 CMANP 和本项目研究新增代码为 CC BY-NC-SA 4.0；原始署名保留。UJI 数据为 CC BY 4.0。完整许可与上游归属见 LICENSE 及其链接、data/DATA_SOURCE.json。不得整体标为 MIT 或暗示未经授权的商业使用许可。
