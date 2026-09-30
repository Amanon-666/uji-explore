# UJI 图定位研究分支：GConvLoc 主线

**实测 GConvLoc 标准 UJI 测试 7.606 m，论文 7.59 m；完整留出 FLOOR=3 的零样本扩展 12.801 m。** 这是单种子独立重实现，不是恢复作者源码，也不代表 JPRL / UE-GLoc 全部复现成功。

分支：`research/gconvloc-jprl-uegloc-20260930`。原有 CMANP 程序保留不动，其使用说明完整保存在 [README_CMANP.md](README_CMANP.md)。本研究使用独立模块和依赖文件，建议使用新虚拟环境。

先读 [实际结果](reports/GRAPH_RESULTS.md) 和 [原文、公开代码与参数审计](docs/GRAPH_REPRODUCTION_AUDIT.md)。

## 安装与数据准备

```bash
git clone --branch research/gconvloc-jprl-uegloc-20260930 https://github.com/Amanon-666/uji-explore.git
cd uji-explore
python3.13 -m venv .venv-graph
source .venv-graph/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-graph.txt
python -m graph_repro.prepare
python -m pytest -q graph_tests
```

`prepare` 从 UCI 下载数据并校验 ZIP/CSV SHA256，再从固定 CNNLoc 提交恢复作者发布的 2,132 条内部开发样本。官方 1,111 条 validation 保持外部测试角色。生成的 `data/raw/graph/cnnloc_split_audit.json` 含完整开发行号；不依赖对话中的临时文件。

## 复跑常规定位：先开发，后冻结测试

```bash
python -m graph_repro.run --data data/raw/graph \
  --out graph_runs/gconvloc_cnnloc_s0 --dev-split cnnloc \
  --seed 0 --epochs 2000 --patience 200 --phase develop

python -m graph_repro.run --data data/raw/graph \
  --out graph_runs/gconvloc_cnnloc_s0 --phase test --allow-test
```

上面是产生 7.605711 m 结果的预算。模型只由内部开发集选定。已有 `FROZEN.json` 的目录不能继续开发；已有 `TEST_EVALUATED.json` 的目录不能重新评分。重新做独立实验请使用新目录并事先确定协议，不能通过新建目录隐瞒测试调参。

## 复跑真正的未见楼层实验

```bash
python -m graph_repro.run --data data/raw/graph \
  --out graph_runs/gconvloc_floor3_s0 --protocol floor-dg --target 3 \
  --dev-split cnnloc --seed 0 --epochs 1000 --patience 150 --phase develop

python -m graph_repro.run --data data/raw/graph \
  --out graph_runs/gconvloc_floor3_s0 --protocol floor-dg --target 3 \
  --phase test --allow-test
```

这里只用原始 FLOOR=0/1/2 训练，FLOOR=3 完全留出；FLOOR=4 不参与四域协议。每个原始楼层域横跨多栋建筑。这不是单独的 13 个 building-floor 任务，不是跨建筑 few-shot，也不是 UE-GLoc。

`--protocol building-dg --target 0/1/2` 提供整栋建筑留出接口，但本次没有运行并报告其结果。普通训练入口不含“给目标层几条标签再适应”的流程。

## JPRL：单独的原公式重实现

```bash
for t in 0 1 2 3; do
  python -m graph_repro.jprl --data data/raw/graph \
    --out graph_runs/jprl10k_f${t}_s0 --target "$t" \
    --steps 10000 --patience 2000 --seed 0 --phase develop || exit
done

# 全部任务开发与选参完成后，再打开目标评估。
for t in 0 1 2 3; do
  python -m graph_repro.jprl --data data/raw/graph \
    --out graph_runs/jprl10k_f${t}_s0 --target "$t" \
    --phase test --allow-test || exit
done
```

每个任务运行四个原文 lambda 候选和一个同骨干 ERM。当前 JPRL 四域等权平均 31.915 m，没有看到相对 ERM 的明显改善。归一化 MAE 之和 0.123353 不能与原文不同文件合并协议的 0.114 直接等同。更差的外部压力测试也完整保留。没有把同作者 JPDA 仓库误标成 JPRL。

## 已训模型推理与档案

完整训练权重、逐样本预测与日志已作为对话 ZIP 交付。将其中的 `graph_runs/` 放到项目根目录后，可直接使用：

```bash
python -m graph_repro.predict --data data/raw/graph \
  --checkpoint graph_runs/gconvloc_cnnloc_s0/best.pt \
  --query your_fingerprints.csv --out predicted_coordinates.csv
```

查询 CSV 只需 `WAP001` 到 `WAP520`，缺失写 100；无需查询坐标标签。输出 UJI 原坐标系中的 `pred_LONGITUDE`、`pred_LATITUDE`。

完整跨层运行也可从 [GitHub Actions](https://github.com/Amanon-666/uji-explore/actions/runs/36673143102) 的 `graph-cross-floor-results` 下载，保留 30 天。该 artifact 只包含跨层模型，标准模型及 JPRL 候选在对话完整包或按上述命令重训获取。

## 模块与边界

`graph_repro/core.py`：GAT、图构造、JPRL L2 损失、预处理与单位明确的指标。`run.py`：标准/整层/整栋建筑留出。`jprl.py`：源域独立选参和 ERM 对照。`prepare.py`：来源与数据划分审计。`predict.py`：无标签查询推理。`graph_tests/`：11 项等价性、梯度、单位、隔离测试。

所有缩放统计与图参考节点来自 source-fit。设备、用户、时间、建筑号、楼层号不输入坐标网络，元数据仅用于划分/审计。图不读取查询标签，不允许查询反向改变参考节点，也不允许查询间传消息。

UE-GLoc 原文和代码尚未取得充分核验，因此当前没有 UE-GLoc 模型。方法出处、所有未明参数与协议变更详见审计文档。研究代码遵守原仓库许可，第三方库和 UJI 数据分别保留原许可。
