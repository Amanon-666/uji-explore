# 三篇论文复现审计：GConvLoc / JPRL / UE-GLoc

审计日期：2026-09-30。研究分支 `research/gconvloc-jprl-uegloc-20260930`。本分支保留 CMANP 源码，图定位主线放在 `graph_repro/`，不把旧结果替换成新方法结果。

## 1. 来源与完成边界

|工作|原始依据|本次处理|
|---|---|---|
|GConvLoc，IEICE 2023|[论文 DOI](https://doi.org/10.1587/transinf.2022EDL8081)，第 2–3 节与表 2–3|主线；按公式和结构独立重实现，复用 PyG 官方 GATConv 参数、初始化和注意力工具。论文链接 `dongdokee/GConvLoc` 在本次 GitHub API 与公开下载检查中均返回 404，未声称恢复了作者源码。|
|JPRL，Neural Computing and Applications 2023|[全文](https://link.springer.com/article/10.1007/s00521-023-08520-1)，公式 6–11、第 5.2 节|独立实现回归版本 L2 联合-乘积分布损失、原文网络和参数网格。未找到 JPRL 官方实现。作者的 `Joint-Product-Distribution-Alignment` 仓库属于另一篇 JPDA 工作，不能冒充本论文源码。|
|UE-GLoc，ACM TOSN 2026|[DOI](https://doi.org/10.1145/3820501)|确认题录、摘要及检索到的跨建筑小节线索；全文下载返回 403，未取得可审计官方代码。因此没有实现“UE-GLoc”网络，也没有报告其复现性能。此前提到的 PNAConv 和逐楼数字不作为本分支依据。|

GConvLoc 的论文成绩是常规 UJI 测试的平均二维欧氏误差 **7.59 m**。JPRL 的 **0.114 是归一化两坐标 MAE 之和**，不是米数。训练集合、模型、随机种子或指标不同，不能用数值接近证明严格复现。

## 2. 尽量复用什么，哪些没有复用

GAT 来自 [PyTorch Geometric 2.6.1](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.GATConv.html)。`MemoryGAT` 继承官方层，仅把显式 E×D 消息张量改为等价稀疏加权求和；一头、零 dropout、显式自环的前向输出、输入梯度及所有参数梯度均与原版逐项测试。`--backend pyg` 保留未经该聚合改写的路径。低内存实现只测试一阶反向传播，不宣称高阶梯度或 GPU 数值完全一致。

使用 [CNNLoc 公开仓库](https://github.com/XudongSong/CNNLoc) 的已发布 `UJIIndoorLoc_codes/AllValuationData.csv` 恢复内部开发集，固定提交 `ec8d6001289a1239e21b23b87d52f16e1086d43f`。2,132 条记录全部匹配官方 training，没有匹配官方 validation。用所有 529 列匹配，坐标仅为 CSV 文本舍入允许 1e-6 米精度，重复记录按多重集匹配。补集为 17,805 条拟合数据。公开开发文件是最接近的可恢复划分，但未得到 GConvLoc 作者确认其使用了字节完全一致的子集。

未复制 CNNLoc 的模型源码，未给缺少许可证的第三方源码擅自授权。新增研究代码沿用仓库现有 CC BY-NC-SA 4.0；PyG 保留自己的 MIT 许可；UJI 数据为 CC BY 4.0，来源 [UCI 310](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)，Torres-Sospedra 等，2014，DOI 10.24432/C5MS59。

## 3. 数据协议：先划分，再拟合所有统计量

### 标准定位 S0：对齐 GConvLoc 的任务

官方 training：17,805 拟合 + 作者开发子集 2,132；官方 validation：1,111 条最终测试。两份官方 CSV 均校验固定 SHA256。论文训练数量写 19,938，而实际下载文件为 19,937；本分支保留实际行数，不补造记录。

图的参考节点只来自拟合集。开发和最终查询只能接收参考节点的信息及自身自环；禁止查询→参考节点及查询间传递，防止批量测试时查询相互影响。坐标、设备、用户、时间均不进入图节点输入。参考标签用于训练损失，查询标签仅在评分阶段使用。早停和超参数选择只看内部开发集。

### 跨楼层 D0–D3：明确属于独立扩展

使用官方 training 的原始 `FLOOR=0,1,2,3` 构成四个域；每个域包含多栋建筑的相应楼层。逐一留出整个原始 FLOOR，仅在其余三个域训练。原始 FLOOR=4 不进入这个四域协议。

这不是把 13 个 `(BUILDINGID,FLOOR)` 全部分别作为任务，也不是 5/10/20-shot：本次目标楼层提供 **0 条训练标签**。JPRL 各源域随机 90/10 划分；GConvLoc 保留 CNNLoc 作者开发集在源域中的交集。这两条线的源开发划分不同，不能把两者成绩包装成完全受控的算法对决。

官方 validation 中相应目标楼层只做额外压力测试。该测试还混合时间、设备、用户变化，不是纯空间效应。代码提供 `building-dg` 的整栋建筑留出接口，但本次没有完成跨建筑结果或任何 UE-GLoc 少样本适应结果。

### 为什么没有直接声称复现 JPRL 表 12

实际计数如下；计数过程可运行 `python -m graph_repro.prepare` 重建。

|原始 FLOOR|training|validation|合计|JPRL 正文样本数|
|---|---:|---:|---:|---:|
|0|4369|132|4501|4501|
|1|5002|462|5464|5464|
|2|4416|306|4722|4722|
|3|5048|172|5220|5220|
|4|1102|39|1141|未列入四域|

四个合计与论文完全对应，强烈支持“原文合并了两份官方文件后按层划域”的解释；这仍是依据计数的推断，而非作者明确声明。本分支为了保留官方外部测试隔离，采用 training-only 扩展；不能把我们的结果与原文 0.114 作严格同协议复现判定。

## 4. 参数逐项登记

|设置|取值|依据及边界|
|---|---|---|
|GConvLoc 输入|520 个原始 WAP 列|原文；不重编号 AP，不提供地图或 AP 坐标|
|RSSI|100→-104，拟合集全局 min-max，缺失归零|原文归一化方向；仅源拟合统计是本分支的防泄漏约束|
|图|余弦 23 近邻；参考自身从 k 中剔除，再添加一次自环|k、度量和自环来自原文；并列距离由 sklearn 确定，作者具体 tie-break 未知|
|GConvLoc 网络|GAT 256→128，一头；FC 64→2|原文表 2|
|隐藏/输出激活|ReLU / linear；无 dropout|正文未充分说明，显式重实现选择；未经过最终测试调参|
|GConvLoc 优化|Adam 0.001，full-batch，normalized coordinate MSE|原文表 2 与训练损失|
|本次标准预算|2000 epochs，patience 200，每 10 epochs 验证|原文仅说明早停、没有这些数值；工程预算，最佳点到上限不等于充分收敛|
|云端跨层预算|1000 epochs，patience 150|独立扩展预算；未用目标层选择|
|JPRL 输入/标签|源拟合集逐列 z-score；源坐标 min-max|原文预处理种类；统计只用 source fit。原文未交代缺失 100 的额外处理，本版保留原值再标准化，必须单独评估其影响|
|JPRL 网络|520→260 ReLU→2 Sigmoid|原文第 5.2.4 节|
|JPRL 损失|每域平方误差 + lambda × 公式 6–10 的 L2 估计|按原公式；未替换成 MMD、RCS、HSIC 或对抗损失|
|JPRL 优化|SGD 0.001，momentum 0.9；lambda∈{0.001,0.1,10,1000}|原文；额外 lambda=0 为同骨干 ERM 对照|
|JPRL 批次|每源域 16 条，总 48；可重复采样|原文说明按域组 minibatch，未公布大小；本次固定工程值|
|JPRL 线性求解|epsilon=0.001；float64 solve；batch 样本作核中心|原文仅说明小正 epsilon；具体数值为工程选择|
|JPRL 预算|10000 steps，patience 2000，每 100 steps 验证|工程预算；3000-step 开发探索未评分目标，正式选参均使用加长预算|
|种子|本次各主任务 seed=0|JPRL 原文 5 次重复，本次不能报告 5 种子复现|
|环境|Python 3.13、PyTorch 2.10、PyG 2.6.1、CPU|现代可安装环境；不同于 GConvLoc 原文 Python 3.10 / Torch 1.12.1 / PyG 2.1.0|

JPRL 的每个 lambda 使用相同初始种子、批次序列与源划分。只以源开发集归一化 MAE 选择 lambda 和停止点；ERM 使用自身源开发最优点。目标层完全不参与候选排序。

## 5. 已做的工程核验与仍然存在的限制

11 项测试覆盖：PyG 等价前后向、图边方向、自环唯一性、查询批次不变性、JPRL 单域损失为零及数值梯度、米制与归一化指标、原始大坐标的浮点精度、预处理仅源训练、目标域分离、作者开发行数。测试不等同于算法达到论文性能。

每个实验保存完整 `FROZEN.json`、源码 SHA256 和启动时源码副本、训练曲线、停止点及 `best.pt`。首次评分要求 `--allow-test`，之后禁止同目录开发或再次评分。锁文件是审计约束，不能防止用户人为删除文件或在别处重复调参。

本地早期 `standard_s0` 的自定开发划分、`jprl_f*_s0` 的 3000-step 运行只用于开发，未评分其最终测试。正式目录为 `gconvloc_cnnloc_s0` 和 `jprl10k_f*_s0`。恢复作者划分的文件位置调整没有改变选中的样本、模型或训练数学；实际启动源码保留在各次运行的 `source_snapshot/`。

尚未完成：作者级源码恢复、JPRL 原始池化数据协议的 5 种子结果、跨建筑 few-shot 适应、UE-GLoc 全文级实现、设备/用户/时间完全匹配的空间因果对照、GPU 验证。实测成绩及差距见 [GRAPH_RESULTS](../reports/GRAPH_RESULTS.md)。
