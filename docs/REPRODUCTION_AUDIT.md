# 原文复现审计

## 1. 固定的研究材料

论文：Feng, L., Tung, F., Hajimirsadeghi, H., Bengio, Y., & Ahmed, M. O. (2024). *Memory Efficient Neural Processes via Constant Memory Attention Block*. ICML, PMLR 235, 13365–13386.

正式页：https://proceedings.mlr.press/v235/feng24i.html

公开PDF：https://arxiv.org/pdf/2305.14567 （核验为v3，2024-05-27；正式会议2024）

代码：https://github.com/BorealisAI/constant-memory-anp ，固定提交 `8961cd940153d76918f401acc60aa858101a8949`。`third_party/manifest.json` 逐文件列出 SHA-256。本地保留原目录与许可，不改官方模型公式。测试会检查文件字节是否改变。

## 2. 复用哪些源码

- `regression/models/cmanp.py`：官方CMANP模型；
- `regression/models/cmanp_modules.py`：官方常量记忆注意力、状态和增量更新；
- `regression/models/lbanp.py`、`lbanp_modules.py`、`modules.py`：继承的嵌入、解码和分布预测；
- `regression/data/gp.py`：官方GP采样器；
- `regression/configs/gp/cmanp.yaml`、`gp.py`：结构与训练预算来源。

`compat/attrdict.py` 只实现上述模块用到的字典属性访问，解决旧attrdict与新Python组合的兼容问题；不是修改注意力方法。

官方训练脚本含硬编码CUDA。本项目另写设备可配置的 runner，调用同一模型和采样器；随机种子在构造模型之前设置；输出明确定义的JSON结果；只在可信训练checkpoint中使用 `weights_only=False`。这些是工程差异，不能称位级复现作者运行。

## 3. 关键不一致：GP length 分布

正文 §4.2：`length ~ Uniform[0.6,1.0)`。

源码 `regression/data/gp.py`：`0.1 + (max_length-0.1)*rand`，`max_length=0.6`；`gp.py` 直接实例化 `RBFKernel()`。

因此代码实际是 `Uniform[0.1,0.6)`。本项目默认复现**已发布代码**，另外提供 `--kernel-spec paper` 文字版检查，两版分别建run目录。没有证据认定哪一版产生了作者表4，所以不能承诺把代码跑完就严格复现表4。应保留差异记录，必要时向作者核实，而不是测试后选择看起来更接近表格的版本再隐藏另一版。

## 4. 原文数值目标与本次执行范围

论文表4的1D GP元回归、对数似然越大越好：

|方法|RBF|Matern 5/2|
|---|---:|---:|
|CMANP|1.24 ± 0.01|0.80 ± 0.01|
|CMANP-AND|1.48 ± 0.03|0.96 ± 0.01|

本项目实现与核对的是前一行。后者是另一个自回归联合输出版本，不能把名字混用。表3图像实验明确说明5种子；表4的±统计口径不能擅自解释成95%置信区间。

本次仅执行 **64步、1种子、每核8评估batch** 的原文代码流程检查，未完成100000步×5种子×3000评估batch。短跑输出保存在对话ZIP的 `reports/gp_smoke/`，不能用于与上表作方法优劣结论。

## 5. UJI 相对原文做了哪些改变

|部分|原文GP|本项目UJI|
|---|---|---|
|任务|每次重新采样GP函数|有限的建筑-楼层集合|
|输入/输出|1维x→1维y|520维RSSI→二维平面坐标|
|支持信息|函数观测点|少量实测位置RSSI+坐标|
|查询独立性|GP采样点|按坐标组隔离的扫描|
|预处理|合成数据原尺度|固定RSSI变换、仅源坐标归一化|
|预算|随机3–46个context|5/10/20个不同位置，各1条扫描|
|选择规则|官方固定训练预算|未见开发楼层MDE选步数|
|测试分布|RBF、Matern等|未见楼层、共同群体限制、外部复合变化|
|主指标|1D log likelihood|二维位置等权MDE，NLL为附加|

以上改造是一个**新的UJI评测协议**。原论文没有其成绩，因此不存在一个诚实的“UJI误差与该论文相差x%”数值。

## 6. 如何验收，而不是只看代码没报错

第一层：数据和公式完整性。官方文件哈希、支持/查询位置隔离、缩放还原、排列与增量等价、梯度有限性、核基线数值稳定均通过测试。

第二层：原文计算流程。GP runner在短跑下完成模型构造、训练、三种核评估。需继续执行全预算，报告每个种子及官方表差值；GP length差异必须持续披露。

第三层：UJI端到端流程。完成开发、冻结、十源重训、三个目标、共同群体限制、外部测试、基线与逐行预测。pilot已完成；正式配置未执行。

第四层：方法有效性。CMANP必须在同一预算和划分下证明目标支持对应关系有用。当前pilot未满足这一层，不能把29项工程测试当成适应机制已有效。

## 7. 代码审计边界

CPU已测试，GPU没有本次执行证据。初始pilot运行期间新增了输出的标签消耗说明、源码哈希保护和报告工具，未改变模型/训练数学规则；pilot配置没有当时的源码指纹，因此不能声称pilot提供了不可变的全过程源码快照。交付后的新prepare会把训练相关源码指纹写入配置，并拒绝代码改变后的续跑。

随机种子、版本哈希和输入校验降低重跑歧义，不等于所有系统/硬件组合位级相同。原始数据是公开观察数据，地面真值精度、同一位置的微小坐标误差和采集组织方式仍限制科学解释。
