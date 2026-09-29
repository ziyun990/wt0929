# 8x8海上风电场闭环机会维护算例（公开文献参数版）

本目录由“最新模型_8x8_主轴承黄玲玲_20260907”复制后独立修改，原模型和历史结果未被覆盖。

## 1. 参数来源与换算

- 部件维修时间和材料成本：Carroll et al. (2015)，图13-15；
- 风机和风场间距：CL-Windcon D4.5中的DTU 10 MW与NORCOWE参考风场；
- 主轴承：Carroll未单列该部件，采用CL-Windcon的main shaft作为主轴/主轴承总成代理；
- CTV航速、燃油、租金、载员、港场距离和班次：Dalgic et al. (2015)；
- 汇率：1 EUR=7.7248 CNY，1 GBP=9.2118 CNY，仅做币种换算，未做通胀调整；
- 材料成本情景系数：5.0。该系数统一作用于PM和CM材料成本，不作用于人工、船舶、燃油和发电损失成本。

Carroll样本来自约350台2-4 MW海上风机。本算例先将文献材料成本换算为人民币，再统一乘以5.0的情景系数，以表征大型机组材料成本水平变化。该系数属于算例设定，不是Carroll文献给出的容量换算关系；容量迁移不确定性仍应通过敏感性实验评估。

## 2. 主算例设置

- 风机数量：64台，8x8派生阵列；
- 单机容量：10 MW，总装机容量640 MW；
- 叶轮直径178.3 m，轮毂高度119 m；
- 切入、额定和切出风速分别为4、11.4和25 m/s；
- 列间距7D=1248.1 m，行间距8D=1426.4 m；
- 关键部件：齿轮箱、发电机、叶片、主轴/主轴承总成，共256个部件单元；
- 滚动周期20 d，每日求解一个24 h小时级排程；
- 维护作业窗口08:00-20:00，最大并行任务数3；
- 运维基地位于风场西侧，距最近边界37.04 km；
- CTV航速26 kn，载员12人，每日最多访问6台风机；
- 燃油消耗384 L/h，燃油价格8.29062元/L，航行燃油成本3183.59808元/h；
- CTV租金32241.30元/d；
- 技术人员费率3862.40元/(人·d)，按12 h班次折算为321.8667元/(人·h)。

## 3. 两级维修参数

当前模型将Carroll“小修”作为PM代理，将“大修”作为CM代理。该映射用于保持现有两级模型结构，不表示Carroll将小修定义为预防性维护。下表材料费为人民币换算基准值乘以5.0后的模型输入。

| 部件 | PM时间/h | PM基准材料费/元 | PM模型材料费/元 | PM人数 | CM时间/h | CM基准材料费/元 | CM模型材料费/元 | CM人数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 齿轮箱 | 8 | 965.60 | 4828.00 | 3 | 22 | 19312.00 | 96560.00 | 4 |
| 发电机 | 7 | 1235.97 | 6179.85 | 3 | 24 | 27036.80 | 135184.00 | 3 |
| 叶片 | 9 | 1313.22 | 6566.10 | 3 | 21 | 11587.20 | 57936.00 | 4 |
| 主轴/主轴承总成 | 10 | 7724.80 | 38624.00 | 2 | 36 | 108147.20 | 540736.00 | 3 |

维修材料费不含人工、船舶、燃油和发电损失。模型分别核算这些成本。

## 4. 功率曲线与布局说明

`method_of_bins.csv`按照DTU 10 MW参考机组的切入、额定和切出风速构造归一化分箱曲线。切入至额定风速之间采用标准三次功率近似，额定至切出风速取1。该曲线是参数化近似，不是CL-Windcon图8的数字化原始点。

CL-Windcon的NORCOWE参考风场包含80台风机；本算例保留64台8x8规模，仅继承7D列间距和8D行间距，因此属于派生布局而不是原布局复现。

## 5. 输入数据

- `data/openmeteo_liaoning_wind_wave_2025_gfs/`：小时级风速、风向和浪高；
- `data/liaoning_spot_prices_api_2025/`：同期日前和实时电价；
- `method_of_bins.csv`：10 MW参数化归一功率曲线。

## 6. 运行方法

小时级MILP支持COPT与CBC两种后端，默认使用COPT。COPT 8.0.7安装在`C:\conda\envs\stochos`环境中，当前主程序不再导入或调用Gurobi。`hourly_component_scheduler.py`、`STOCHOS.py`和`case_8x8_multicomponent_cl_stochos_mu.py`仅为历史Gurobi实现，不属于当前小时级COPT运行链路。

```powershell
Set-Location 'F:\文献\开题\机会维护0412\最新模型_8x8_Carroll_Dalgic_CL参数_20260927\multi_component_case_64turbine_routing_fuel'
& 'C:\conda\envs\stochos\python.exe' .\validate_64turbine_case.py
& 'C:\conda\envs\stochos\python.exe' .\validate_wake_marginal_vectorization.py
& 'C:\conda\envs\stochos\python.exe' .\case_8x8_multicomponent_hourly_milp.py --solver copt --sample 31 --strategy multi_opportunity --routing-cost-in-objective --output-tag sample31_s4_copt
& 'C:\conda\envs\stochos\python.exe' .\run_five_strategy_routing_objective.py --solver copt --samples 31 --output-tag sample31_carroll_copt
```

S5的尾流边际损失采用向量化单机停机计算：每个风向只构造一次机组间Jensen亏损矩阵，再利用平方和叠加关系批量得到64种单机停机状态。该实现与逐台停机重建风场的原始算法数值等价，但不改变尾流参数、功率曲线或排程目标。可使用`validate_wake_marginal_vectorization.py`执行等价性回归检查。

持续任务推进权重仅对上一滚动周期已经开始但尚未完成的`ongoing`任务生效。新出现的强制性纠正任务虽然允许在资源不足时部分推进，但不再获得持续任务推进权重，以避免其与既有持续任务竞争同一优先权重。

为使工时超过单日窗口的CM任务能够合理启动，新CM的未完成执行模式按“纠正完成优先值/CM总工时”获得较低的逐小时部分推进权重；该权重仅作用于当日不能完成的模式。既有持续任务仍使用50000元/h的较高推进权重，因此模型优先推进已开始任务，再利用剩余能力启动新的跨日CM任务。完整完成模式不叠加部分推进权重，避免重复计入收益。

## 7. 解释边界

完成维修后恢复完整RUL是当前模型假设。由于Carroll的小修和大修未必达到“修复如新”，论文中应明确这一简化，或在后续模型中增加维修恢复系数。大部件更换尚未纳入主模型；若启用，应增加JUV、吊装、动员时间及相应天气约束。
