import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('consumer_realistic_raw.csv')

# ============================================================
# 原始数据概况
# ============================================================
print("=" * 60)
print("原始数据概况")
print("=" * 60)
print(f"数据形状: {df.shape}")
print(f"\n前5行数据:\n{df.head()}")
print(f"\n数据类型:\n{df.dtypes}")
print(f"\n基本统计:\n{df.describe()}")
print(f"\n缺失值统计:\n{df.isnull().sum()}")
print(f"\n各列缺失值比例:")
for col in df.columns:
    missing_rate = df[col].isnull().mean() * 100
    print(f"  {col}: {missing_rate:.2f}%")

# ============================================================
# 预处理
# ============================================================
df_drop = df.copy()
df_mean = df.copy()
numeric_cols = ['age', 'income', 'consumption_index']

# ============================================================
# 方法一：直接删除缺失行
# ============================================================
print("\n" + "=" * 60)
print("方法一：直接删除缺失行")
print("=" * 60)
print(f"删除前数据量: {len(df_drop)} 行")
print(f"缺失值分布:\n{df_drop.isnull().sum()}")

df_drop = df_drop.dropna()
print(f"删除后数据量: {len(df_drop)} 行")
print(f"损失数据: {1000 - len(df_drop)} 行 ({(1000 - len(df_drop)) / 10:.1f}%)")

# Z-score异常值处理
print("\n异常值检测(Z-score > 3):")
for col in numeric_cols:
    mean = df_drop[col].mean()
    std = df_drop[col].std()
    z_scores = (df_drop[col] - mean) / std
    outliers = np.abs(z_scores) > 3
    print(f"  {col}: 发现 {outliers.sum()} 个异常值")
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    df_drop[col] = df_drop[col].clip(lower_bound, upper_bound)

# 标准化
scaler1 = StandardScaler()
df_drop[numeric_cols] = scaler1.fit_transform(df_drop[numeric_cols])
print(f"\n标准化后统计:\n{df_drop[numeric_cols].describe().round(2)}")

# ============================================================
# 方法二：均值填充
# ============================================================
print("\n" + "=" * 60)
print("方法二：均值填充")
print("=" * 60)
print(f"填充前数据量: {len(df_mean)} 行")
print(f"缺失值分布:\n{df_mean.isnull().sum()}")

# 用均值填充
for col in numeric_cols:
    col_mean = df_mean[col].mean()
    df_mean[col].fillna(col_mean, inplace=True)
    print(f"  {col} 均值: {col_mean:.2f}")

print(f"填充后缺失值: {df_mean.isnull().sum().sum()}")

# Z-score异常值处理
print("\n异常值检测(Z-score > 3):")
for col in numeric_cols:
    mean = df_mean[col].mean()
    std = df_mean[col].std()
    z_scores = (df_mean[col] - mean) / std
    outliers = np.abs(z_scores) > 3
    print(f"  {col}: 发现 {outliers.sum()} 个异常值")
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    df_mean[col] = df_mean[col].clip(lower_bound, upper_bound)

# 标准化
scaler2 = StandardScaler()
df_mean[numeric_cols] = scaler2.fit_transform(df_mean[numeric_cols])
print(f"\n标准化后统计:\n{df_mean[numeric_cols].describe().round(2)}")

# ============================================================
# 两种方法对比
# ============================================================
print("\n" + "=" * 60)
print("两种方法对比")
print("=" * 60)
print(f"删除法保留数据量: {len(df_drop)} 行")
print(f"填充法保留数据量: {len(df_mean)} 行")
print(f"删除法损失: {1000 - len(df_drop)} 行 ({(1000 - len(df_drop)) / 10:.1f}%)")

# ============================================================
# 线性回归建模对比
# ============================================================
print("\n" + "=" * 60)
print("线性回归建模对比")
print("=" * 60)

# 方法一：删除法建模
X_drop = df_drop[['age', 'income']]
y_drop = df_drop['consumption_index']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_drop, y_drop, test_size=0.2, random_state=42)

model1 = LinearRegression()
model1.fit(X_train1, y_train1)
y_pred1 = model1.predict(X_test1)

mse1 = mean_squared_error(y_test1, y_pred1)
r2_1 = r2_score(y_test1, y_pred1)

print(f"\n删除法模型:")
print(f"  MSE: {mse1:.4f}")
print(f"  R²:  {r2_1:.4f}")

# 方法二：均值填充建模
X_mean = df_mean[['age', 'income']]
y_mean = df_mean['consumption_index']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mean, y_mean, test_size=0.2, random_state=42)

model2 = LinearRegression()
model2.fit(X_train2, y_train2)
y_pred2 = model2.predict(X_test2)

mse2 = mean_squared_error(y_test2, y_pred2)
r2_2 = r2_score(y_test2, y_pred2)

print(f"\n均值填充模型:")
print(f"  MSE: {mse2:.4f}")
print(f"  R²:  {r2_2:.4f}")

# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 图1：删除法 - 预测值 vs 真实值
axes[0, 0].scatter(y_test1, y_pred1, alpha=0.5, color='blue')
axes[0, 0].plot([y_test1.min(), y_test1.max()], [y_test1.min(), y_test1.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('真实值')
axes[0, 0].set_ylabel('预测值')
axes[0, 0].set_title(f'删除法: 预测 vs 真实\nR²={r2_1:.4f}, MSE={mse1:.4f}')

# 图2：均值填充 - 预测值 vs 真实值
axes[0, 1].scatter(y_test2, y_pred2, alpha=0.5, color='green')
axes[0, 1].plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('真实值')
axes[0, 1].set_ylabel('预测值')
axes[0, 1].set_title(f'均值填充: 预测 vs 真实\nR²={r2_2:.4f}, MSE={mse2:.4f}')

# 图3：残差分布对比
residuals1 = y_test1 - y_pred1
residuals2 = y_test2 - y_pred2
axes[1, 0].hist(residuals1, bins=20, alpha=0.7, label='删除法', color='blue')
axes[1, 0].hist(residuals2, bins=20, alpha=0.7, label='均值填充', color='green')
axes[1, 0].set_xlabel('残差')
axes[1, 0].set_ylabel('频数')
axes[1, 0].set_title('残差分布对比')
axes[1, 0].legend()

# 图4：模型性能对比柱状图
methods = ['删除法', '均值填充']
mse_values = [mse1, mse2]
r2_values = [r2_1, r2_2]

x = np.arange(len(methods))
width = 0.35

ax2 = axes[1, 1]
bars1 = ax2.bar(x - width/2, mse_values, width, label='MSE', color='coral')
ax2.set_ylabel('MSE', color='coral')
ax2.tick_params(axis='y', labelcolor='coral')

ax2_twin = ax2.twinx()
bars2 = ax2_twin.bar(x + width/2, r2_values, width, label='R²', color='steelblue')
ax2_twin.set_ylabel('R²', color='steelblue')
ax2_twin.tick_params(axis='y', labelcolor='steelblue')

ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.set_title('模型性能对比')

# 添加数值标签
for bar, val in zip(bars1, mse_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}',
             ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, r2_values):
    ax2_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}',
                  ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# ============================================================
# 最终结论
# ============================================================
print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print(f"删除法: 样本量={len(df_drop)}, R²={r2_1:.4f}, MSE={mse1:.4f}")
print(f"均值填充: 样本量={len(df_mean)}, R²={r2_2:.4f}, MSE={mse2:.4f}")

if r2_1 > r2_2:
    print("\n结论：删除法模型表现更好")
else:
    print("\n结论：均值填充模型表现更好")
