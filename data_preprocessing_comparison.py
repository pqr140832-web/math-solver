import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
df = pd.read_csv('user_consumption_data.csv')

print("=" * 60)
print("原始数据概况")
print("=" * 60)
print(f"数据形状: {df.shape}")
print(f"\n缺失值统计:\n{df.isnull().sum()}")
print(f"\n缺失值比例:\n{(df.isnull().mean() * 100).round(2)}%")

# ============================================================
# 方法一：均值填充
# ============================================================
print("\n" + "=" * 60)
print("方法一：均值填充")
print("=" * 60)

df_mean_fill = df.copy()

# 计算各列均值
age_mean = df_mean_fill['age'].mean()
income_mean = df_mean_fill['income'].mean()
consumption_mean = df_mean_fill['consumption_index'].mean()

print(f"填充前均值 - age: {age_mean:.2f}, income: {income_mean:.2f}, consumption: {consumption_mean:.2f}")

# 用均值填充空值
df_mean_fill['age'].fillna(age_mean, inplace=True)
df_mean_fill['income'].fillna(income_mean, inplace=True)
df_mean_fill['consumption_index'].fillna(consumption_mean, inplace=True)

print(f"填充后数据量: {len(df_mean_fill)} 行")
print(f"填充后缺失值: {df_mean_fill.isnull().sum().sum()}")

# Z-score异常值检测与处理
def handle_outliers_zscore(data, columns, threshold=3):
    """用Z-score检测异常值，用边界值替换"""
    df_clean = data.copy()
    for col in columns:
        mean = df_clean[col].mean()
        std = df_clean[col].std()
        z_scores = (df_clean[col] - mean) / std

        # 统计异常值
        outliers = np.abs(z_scores) > threshold
        print(f"  {col}: 发现 {outliers.sum()} 个异常值")

        # 用边界值替换异常值
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

    return df_clean

print("\n异常值检测(Z-score > 3):")
numeric_cols = ['age', 'income', 'consumption_index']
df_mean_fill = handle_outliers_zscore(df_mean_fill, numeric_cols)

# 标准化
scaler1 = StandardScaler()
df_mean_fill[numeric_cols] = scaler1.fit_transform(df_mean_fill[numeric_cols])

print(f"\n标准化后数据统计:\n{df_mean_fill[numeric_cols].describe().round(2)}")

# ============================================================
# 方法二：直接删除缺失行
# ============================================================
print("\n" + "=" * 60)
print("方法二：直接删除缺失行")
print("=" * 60)

df_drop = df.copy()

print(f"删除前数据量: {len(df_drop)} 行")
df_drop = df_drop.dropna()
print(f"删除后数据量: {len(df_drop)} 行")
print(f"删除了 {len(df) - len(df_drop)} 行数据 ({(len(df) - len(df_drop)) / len(df) * 100:.2f}%)")

# Z-score异常值检测与处理
print("\n异常值检测(Z-score > 3):")
df_drop = handle_outliers_zscore(df_drop, numeric_cols)

# 标准化
scaler2 = StandardScaler()
df_drop[numeric_cols] = scaler2.fit_transform(df_drop[numeric_cols])

print(f"\n标准化后数据统计:\n{df_drop[numeric_cols].describe().round(2)}")

# ============================================================
# 模型训练与对比
# ============================================================
print("\n" + "=" * 60)
print("模型训练与效果对比")
print("=" * 60)

def train_and_evaluate(data, method_name):
    """训练线性回归模型并评估"""
    X = data[['age', 'income']]
    y = data['consumption_index']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{method_name}:")
    print(f"  训练集大小: {len(X_train)}")
    print(f"  测试集大小: {len(X_test)}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R² Score: {r2:.4f}")

    return {'mse': mse, 'rmse': rmse, 'r2': r2, 'train_size': len(X_train)}

result1 = train_and_evaluate(df_mean_fill, "方法一（均值填充）")
result2 = train_and_evaluate(df_drop, "方法二（删除缺失）")

# ============================================================
# 结果对比总结
# ============================================================
print("\n" + "=" * 60)
print("两种方法对比总结")
print("=" * 60)

print(f"""
指标              | 均值填充      | 删除缺失
------------------|---------------|---------------
训练样本数        | {result1['train_size']}           | {result2['train_size']}
MSE               | {result1['mse']:.4f}        | {result2['mse']:.4f}
RMSE              | {result1['rmse']:.4f}        | {result2['rmse']:.4f}
R² Score          | {result1['r2']:.4f}        | {result2['r2']:.4f}
""")

print("""
分析结论:
1. 均值填充保留了全部1000个样本，删除法损失了约5%的数据
2. 均值填充可能会降低数据的方差，使分布更集中
3. 删除法保证了数据的真实性，但样本量减少
4. 对于正态分布数据，5%的缺失率下两种方法差异通常不大
5. 如果缺失不是随机的(MNAR)，删除法可能引入偏差

建议:
- 缺失率低(<5%)且随机缺失时，两种方法都可以
- 样本量充足时优先考虑删除法，保证数据质量
- 样本量紧张时使用均值填充，保留更多信息
""")
