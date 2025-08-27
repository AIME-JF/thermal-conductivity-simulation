import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.optimize import minimize
import matplotlib.pyplot as plt  # 确保已导入matplotlib
# ==================== 第一步：基于公式(4)训练初始模型 ====================
# 常量定义（单位：国际标准制）
m_copper = 0.93577  # kg
c_copper = 385.0    # J/(kg·K)
delta_T_over_delta_t = 0.0360 # °C/s
h_b = 0.00847       # m
d_b = 0.12708       # m
h_c = 0.00815       # m
d_c = 0.13122       # m

# 计算公式(4)中的常数C
numerator = d_c + 4 * h_c
denominator = 2 * d_c + 4 * h_c
ratio = numerator / denominator
area_term = 4 / (np.pi * d_b ** 2)
C = m_copper * c_copper * delta_T_over_delta_t * ratio * h_b * area_term

# 生成模拟数据 (T1, T2) → λ
num_samples = 10000
np.random.seed(42)
T1 = np.random.uniform(50, 100, num_samples)
T2 = np.random.uniform(0, 50, num_samples)
invalid = T1 <= T2  # 确保T1 > T2
while invalid.any():
    T1[invalid] = np.random.uniform(50, 100, invalid.sum())
    T2[invalid] = np.random.uniform(0, 50, invalid.sum())
    invalid = T1 <= T2
lambda_values = C / (T1 - T2)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(np.column_stack((T1, T2)))
y = scaler_y.fit_transform(lambda_values.reshape(-1, 1))

# 构建数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义并训练神经网络模型
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(2,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

initial_model = build_model()
initial_model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)
initial_model.save('initial_model.keras')  # 使用新的Keras格式

# ==================== 第二步：实验数据回归与模型微调 ====================
# 实验数据
raw_data = {
    'T3': [52.3,52,50.6,47.7,50.1,46.8,51.4,54,53,57.2,52.6,56.4,58.3,53.9,52.2,
           54.5,52.5,52.2,53.1,52.1,52.9,49.2,50.6,52.4,52.9,51.8,50.9,52,51,53.3,
           50.3,56.1,51.3,53.9,51.1,50.9,58.8,57.3,58.7,56.8,57.3,56.3,62.4,55.7,
           54.5,43,49.3,49,48.3,46.4,46.4,48.6,48,49.6,49.6,59.9,59.1,60.2,60.5,
           59.9,60.7,59.8,60.2,56.8,57.3,62.4,56.2,55.7,54.5,58.8,50.1,49.8,49.8,
           47.4,48,47.8,45.7,58.2,59.8,59.9,59.1,60.2,60.5,59.9,60.7,59.9,56.8,
           60.2,48.7,49.1,49,48.4,48.1,49.3,50.1,48.8,49.3,49,46.5,47.9],
 'lambda': [0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,
            0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,
            0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,
            0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,
            0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,
            0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,
            0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,
            0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,
            0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,
            0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23]
}

# 创建DataFrame并过滤异常值
df = pd.DataFrame({
    'T1': [75] * len(raw_data['T3']),  # 使用列表形式
    'T3': raw_data['T3'],
    'lambda': raw_data['lambda']
})
df = df[df['lambda'] < 0.5]  # 过滤异常值

# 定义修正函数和误差函数
def corrected_T2(T3, a, b):
    return a * T3 + b

def error(params):
    a, b = params
    T2_pred = corrected_T2(df['T3'], a, b)
    lambda_pred = C / (df['T1'] - T2_pred)
    return np.mean((lambda_pred - df['lambda']) ** 2)

# 优化参数a和b（添加bounds限制合理范围）
initial_guess = [1.0, 0.0]
result = minimize(error, initial_guess, method='L-BFGS-B', 
                 bounds=[(0.5, 1.5), (-10, 10)])  # a∈[0.5,1.5), b∈[-10,10]
a_fit, b_fit = result.x
#   print(f"拟合参数: a={a_fit:.4f}, b={b_fit:.4f}")

# 生成微调数据并标准化
T2_prime = corrected_T2(df['T3'], a_fit, b_fit)
X_fine = scaler_X.transform(np.column_stack((df['T1'], T2_prime)))
y_fine = scaler_y.transform(df['lambda'].values.reshape(-1, 1))

# 微调模型
fine_tuned_model = keras.models.load_model('initial_model.keras')
fine_tuned_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # 更低的学习率
                        loss='mse')
fine_tuned_model.fit(X_fine, y_fine, epochs=100, verbose=0)  # 更多epochs
fine_tuned_model.save('fine_tuned_model.keras')

# ==================== 模型预测 ====================
def predict_lambda(T1, T3):
    T2_prime = corrected_T2(np.array([T3]), a_fit, b_fit)[0]
    X_pred = scaler_X.transform(np.array([[T1, T2_prime]]))
    y_pred = fine_tuned_model.predict(X_pred)
    return scaler_y.inverse_transform(y_pred)[0][0]

# 预测


lambda_pred = predict_lambda(75, 52)
print(f"预测导热系数: {lambda_pred:.3f} W/(m·K)")


# -------------------- 模型训练与损失监控 --------------------
# 生成输入数据
X = np.column_stack((df['T1'], a_fit * df['T3'] + b_fit))
y = df['lambda'].values

# 数据归一化
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_norm = (X - X_mean) / X_std

# 构建神经网络模型
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(2,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# 训练配置
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# 训练模型
model = build_model()
history = model.fit(
    X_norm, y,
    epochs=200,
    validation_split=0.2,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# -------------------- 损失可视化与分析 --------------------
# 绘制训练曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Convergence')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE Convergence')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------- 模型评估 --------------------
from sklearn.metrics import r2_score
y_pred = model.predict(X_norm).flatten()
print(f"模型性能评估:")
print(f"- R² Score: {r2_score(y, y_pred):.4f}")
print(f"- MAE: {np.mean(np.abs(y - y_pred)):.4f}")
print(f"- MSE: {np.mean((y - y_pred)**2):.4f}")