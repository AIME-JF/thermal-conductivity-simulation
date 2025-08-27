# 结果文件目录

此目录用于存放预测结果、训练日志和导出文件。

## 文件类型

### 预测结果
- `predictions.csv` - 批量预测结果
- `steady_predictions_*.csv` - 稳态法预测结果
- `quasi_predictions_*.csv` - 准稳态法预测结果

### 训练日志
- `training_log_*.txt` - 模型训练日志
- `training_history.json` - 训练历史记录

### 导出文件
- `*.png` - 导出的图表文件
- `*.pdf` - 导出的实验报告
- `*.csv` - 导出的数据文件

## 文件命名规范

- 时间戳格式：`YYYYMMDD_HHMMSS`
- 任务ID格式：UUID字符串
- 示例：`steady_predictions_20241221_143022.csv`

## 清理策略

- 预测结果文件保留30天
- 训练日志保留90天
- 导出文件保留7天
- 训练历史永久保留（最多1000条记录）