# 8x8 海上风电场闭环机会维护模型

本仓库保存论文算例所使用的最新模型、输入数据和正式运行结果。模型面向 64 台风机、4 类关键部件的闭环滚动维护问题，同时考虑 RUL 不确定性、天气可达性、连续停机、运维船访问与燃油费用，以及尾流感知场级发电损失。

## 当前版本

- 实验配置：`64turbine_8x8_carroll_dalgic_cl_material5x_cm_partial_progress_20260928`
- 风场规模：8×8，共 64 台 10 MW 风机
- 部件数量：齿轮箱、发电机、叶片、主轴/主轴承总成，共 256 个部件单元
- 仿真长度：20 d，时间步长 1 h
- 作业窗口：08:00–20:00
- 默认并行能力：3 项任务
- 求解器：COPT；代码同时保留 PuLP/CBC 后端
- 正式样本：`1, 26, 51, 76, 101, 126, 151, 176, 201, 226, 251, 276`

## 目录结构

```text
.
├─ cl_stochos_mu.py                 # 候选任务与尾流计算公共模块
├─ method_of_bins.csv               # 10 MW 参数化功率曲线
├─ data/                             # 小时级天气、海况和电价输入
├─ model/                            # 最新 COPT/CBC 模型及验证脚本
├─ results/
│  ├─ main_12samples/               # 12 样本 × 5 策略正式结果
│  └─ sensitivity_sample26/         # sample26 的 K=2、K=4 敏感性结果
├─ analysis/                         # 第五节统计数据与制图脚本
└─ paper_figures/                    # 基于最新结果生成的论文图
```

`results/main_12samples/` 中包含每个样本与策略的初始 RUL、滚动周期汇总、任务执行记录、小时级排程和序列化结果。核心汇总文件为：

```text
five_strategy_routing_objective_summary_material5x_12samples_five_strategies_copt.csv
```

## 环境

推荐使用 Python 3.10 或 3.11。安装基础依赖：

```powershell
python -m pip install -r requirements.txt
```

COPT Python 接口需要按照本机 COPT 安装方式配置，并保证 `import coptpy` 可用。没有 COPT 时，可使用代码中保留的 PuLP/CBC 后端进行小规模检查。

## 数据与参数检查

在仓库根目录执行：

```powershell
python .\model\validate_64turbine_case.py
python .\model\validate_wake_marginal_vectorization.py
```

## 运行单个算例

```powershell
python .\model\case_8x8_multicomponent_hourly_milp.py `
  --solver copt `
  --sample 26 `
  --strategy wake_multi_opportunity `
  --routing-cost-in-objective `
  --output-tag sample26_s5
```

## 运行 12 样本五策略实验

```powershell
python .\model\run_five_strategy_routing_objective.py `
  --solver copt `
  --samples 1 26 51 76 101 126 151 176 201 226 251 276 `
  --output-tag material5x_12samples_five_strategies_copt
```

运行结果默认写入 `model/`，五策略汇总 CSV 会由批量运行脚本自动生成。需要重新绘制原汇总图时可执行：

```powershell
python .\model\summarize_five_strategy_routing_objective.py `
  --summary-csv .\model\five_strategy_routing_objective_summary_material5x_12samples_five_strategies_copt.csv `
  --output-dir .\model\outputs `
  --tag material5x_12samples_five_strategies_copt
```

## 并行维护能力敏感性

正式实验中 `K=3` 的 sample26-S5 结果位于 `results/main_12samples/`。K=2 和 K=4 的补充结果位于 `results/sensitivity_sample26/`。

## 论文图复现

```powershell
python .\analysis\build_section5_figures.py `
  --model-root .\results\main_12samples `
  --output-dir .\analysis_output
```

## 结果边界

- Carroll 等文献的维修材料成本已换算为人民币，并在当前情景中统一乘以 5；该系数为算例设定，不是文献给出的容量换算关系。
- 主轴承采用主轴/主轴承总成代理参数。
- 多机同时停机时，尾流边际损失采用预计算单机边际损失的线性叠加近似。
- 当前结果用于不同策略在相同输入下的相对比较，不应直接解释为特定真实风场的绝对成本预测。

## 文件完整性

`MANIFEST_SHA256.csv` 记录发布包内文件的相对路径、大小和 SHA-256，可用于核对下载后的文件完整性。
