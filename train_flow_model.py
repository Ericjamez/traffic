# -*- coding: utf-8 -*-
"""
车流量预测模型训练脚本
回归模型：预测 flow_index (车流量指数)
"""
import re
import pandas as pd
import joblib
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_PATH  = 'static/data/final_traffic_data.csv'
MODEL_DIR = 'model/'

print('[1/6] 加载数据...')
df = pd.read_csv(CSV_PATH)
print('  行数: %d  列数: %d' % df.shape)

# 使用原始道路名称，不进行合并
df['road_simple'] = df['road_name'].astype(str)
n_roads = df['road_simple'].nunique()
print('  唯一道路数（原始）: %d' % n_roads)

# 解析时间特征
df['collect_time'] = pd.to_datetime(df['collect_time'])
df['hour']         = df['collect_time'].dt.hour
df['day_of_week']  = df['collect_time'].dt.dayofweek  # 0=周一

print('[2/6] 按道路+时间特征聚合...')
agg_df = df.groupby(
    ['road_simple', 'hour', 'day_of_week', 'season', 'time_period',
     'weather', 'is_peak_hour'],
    as_index=False
).agg(
    avg_speed         = ('avg_speed',         'mean'),
    congestion_status = ('congestion_status',  lambda x: int(round(x.mean()))),
    flow_index        = ('flow_index',        'mean'),  # 车流量指数
    temperature       = ('temperature',        'mean'),
    humidity          = ('humidity',           'mean'),
)
print('  聚合后样本数: %d' % len(agg_df))

print('[3/6] 编码分类特征...')
cat_cols = ['road_simple', 'season', 'time_period', 'weather']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    agg_df[col + '_enc'] = le.fit_transform(agg_df[col])
    encoders[col] = le
    print('  %s -> %d 类: %s' % (col, len(le.classes_), list(le.classes_)))

# 保存道路列表供前端下拉菜单使用
road_list = sorted(df['road_simple'].unique().tolist())
joblib.dump(road_list, MODEL_DIR + 'road_list.pkl')
print('  道路列表已保存: %d 条道路' % len(road_list))

# 保存季节/天气/时段选项供前端使用
options = {
    'season':      list(encoders['season'].classes_),
    'time_period': list(encoders['time_period'].classes_),
    'weather':     list(encoders['weather'].classes_),
}
joblib.dump(options, MODEL_DIR + 'options.pkl')

print('[4/6] 训练随机森林回归模型（预测车流量）...')
FEATURES = [
    'road_simple_enc', 'hour', 'day_of_week',
    'season_enc', 'time_period_enc', 'weather_enc',
    'is_peak_hour', 'temperature', 'humidity'
]
TARGET = 'flow_index'  # 预测车流量指数

X = agg_df[FEATURES]
y = agg_df[TARGET]

# 检查数据范围
print('  车流量范围: %.0f - %.0f' % (y.min(), y.max()))
print('  车流量均值: %.0f' % y.mean())

# 使用原始数据，不进行对数变换（避免无限大值问题）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print('  训练集: %d  测试集: %d' % (len(X_train), len(X_test)))

# 训练回归模型
reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    max_samples=0.8
)
reg.fit(X_train, y_train)

print('[5/6] 评估模型性能...')
# 预测
y_pred = reg.predict(X_test)

# 确保预测值在合理范围内
y_pred = np.clip(y_pred, y.min(), y.max())

# 评估指标
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('  平均绝对误差 (MAE): %.2f' % mae)
print('  均方根误差 (RMSE): %.2f' % rmse)
print('  R2 分数: %.4f' % r2)

# 计算相对误差（避免除以0）
mask = y_test > 0
if mask.any():
    relative_error = np.abs(y_test[mask] - y_pred[mask]) / y_test[mask]
    print('  平均相对误差: %.2f%%' % (relative_error.mean() * 100))
    print('  中位数相对误差: %.2f%%' % (np.median(relative_error) * 100))
else:
    print('  无法计算相对误差（测试集全为0）')

# 特征重要性
fi = sorted(zip(FEATURES, reg.feature_importances_), key=lambda x: -x[1])
print('  特征重要性:')
for f, v in fi:
    print('    %-28s %.4f' % (f, v))

print('[6/6] 保存模型...')
joblib.dump(reg,      MODEL_DIR + 'flow_model.pkl')
joblib.dump(encoders, MODEL_DIR + 'flow_encoders.pkl')
joblib.dump(FEATURES, MODEL_DIR + 'flow_features.pkl')

# 保存缩放信息（用于逆变换）
scaler_info = {
    'use_log_transform': True,
    'y_mean': float(y.mean()),
    'y_std': float(y.std()),
    'y_min': float(y.min()),
    'y_max': float(y.max())
}
joblib.dump(scaler_info, MODEL_DIR + 'flow_scaler.pkl')

print('  完成！文件已保存到 model/ 目录')
print('  模型文件:')
print('    - flow_model.pkl: 车流量预测模型')
print('    - flow_encoders.pkl: 编码器')
print('    - flow_features.pkl: 特征列表')
print('    - flow_scaler.pkl: 缩放信息')

# 测试预测示例
print('\n[示例] 测试预测:')
sample_idx = 0
sample_X = X_test.iloc[[sample_idx]]
true_y = y_test.iloc[sample_idx]
pred_y = y_pred[sample_idx]

print('  真实车流量: %.0f 辆/小时' % true_y)
print('  预测车流量: %.0f 辆/小时' % pred_y)
print('  误差: %.0f 辆/小时 (%.1f%%)' % (abs(true_y - pred_y), abs(true_y - pred_y)/true_y*100))