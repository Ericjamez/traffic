import pandas as pd
import numpy as np

print("正在分析车流量数据...")
print("=" * 50)

# 读取数据
df = pd.read_csv('static/data/final_traffic_data.csv')

print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print()

# 分析flow_index列
print("flow_index 分析:")
print("-" * 30)
print(f"最小值: {df['flow_index'].min()}")
print(f"最大值: {df['flow_index'].max()}")
print(f"平均值: {df['flow_index'].mean():.2f}")
print(f"中位数: {df['flow_index'].median()}")
print(f"标准差: {df['flow_index'].std():.2f}")
print()

# 查看数据分布
print("flow_index 分布:")
print("-" * 30)
print(df['flow_index'].describe())
print()

# 检查是否有归一化/标准化
print("数据预处理分析:")
print("-" * 30)

# 检查是否已经归一化到0-100范围
if df['flow_index'].max() <= 100:
    print("OK: flow_index 已经归一化到 0-100 范围")
    print(f"  实际范围: {df['flow_index'].min():.2f} - {df['flow_index'].max():.2f}")
else:
    print("NO: flow_index 未归一化到 0-100 范围")
    print(f"  实际范围: {df['flow_index'].min():.2f} - {df['flow_index'].max():.2f}")
    
    # 检查是否是流量指数（相对值）
    if df['flow_index'].max() < 2000:
        print("  可能是流量指数（相对值），不是实际车流量")
        print("  实际车流量可能需要乘以一个系数")
    else:
        print("  可能是实际车流量数值")

print()

# 查看前10行数据
print("前10行数据中的flow_index:")
print("-" * 30)
for i in range(min(10, len(df))):
    row = df.iloc[i]
    print(f"{i+1}: road={row['road_name']}, flow_index={row['flow_index']}, speed={row['avg_speed']}, status={row['congestion_status']}")

print()

# 检查模型训练代码中的处理
print("模型训练中的数据处理:")
print("-" * 30)
print("在 train_model.py 中，模型预测的是 congestion_status (1-4)")
print("flow_index 在趋势图表中显示，但没有用于模型训练")
print("预测结果中的 est_speed 是根据拥堵状态估算的，不是实际车流量")

print()
print("=" * 50)
print("结论:")
print("1. flow_index 是流量指数，不是实际车流量")
print("2. 数值范围大约在 0-2000 之间，表示相对流量")
print("3. 预测模型输出的是拥堵状态和估算车速")
print("4. 实际车流量需要根据流量指数和道路特性计算")