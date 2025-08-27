# 模型文件目录

此目录用于存放预训练的机器学习模型文件。

## 文件说明

### 稳态法模型
- `steady_initial_model.keras` - 稳态法初始模型
- `steady_fine_tuned_model.keras` - 稳态法微调模型
- `steady_scaler_X.pkl` - 稳态法输入特征标准化器
- `steady_scaler_y.pkl` - 稳态法输出标准化器

### 准稳态法模型
- `quasi_multitask_model.keras` - 准稳态法多任务模型
- `quasi_scaler_X.pkl` - 准稳态法输入特征标准化器
- `quasi_scaler_lambda.pkl` - 准稳态法导热系数标准化器
- `quasi_scaler_c.pkl` - 准稳态法比热容标准化器

## 模型格式

- `.keras` 文件：TensorFlow/Keras模型文件
- `.pkl` 文件：scikit-learn标准化器，使用joblib保存

## 使用说明

模型文件由训练脚本生成，用于在线预测服务。如需重新训练模型，请使用管理员面板的训练功能。