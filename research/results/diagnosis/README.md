# 诊断输出口径

- `trees_threads4/`：与既有B2/B3统一四线程后的正式开发诊断输出。
- `graphs/`：四种神经修正对照，各三个种子；训练仅使用2010—2015年前向交叉拟合基础预测，验证2016—2018。
- `crossfit_*.csv`：已完成的前向预测检查点，勿无故重跑。
- 本目录根部早期`D*_search.csv`、`D*_predictions.csv`、`D*.pkl`、`seedcheck_*`与`validation_metrics.csv`来自单线程数值环境，仅作排查记录，不能混入正式四线程比较。
- 最终汇总采用`tree_summary_verified.csv`、`graph_summary.csv`和对应逐年明细。

线程问题已解释：同一预处理矩阵及种子下，当前XGBoost运行时改变线程数会改变树；四线程复现原模型预测完全一致。证据见`../../audit/model_reproduction.json`。该问题不涉及原始数据或已生成历史特征，无须重算它们。

诊断集已多次用于参数及检查点选择，所有结果都是开发证据；2019—2020校准期和2021—2023末端测试均未在本轮用于选择或评价。
