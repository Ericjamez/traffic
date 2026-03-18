# -*- coding: utf-8 -*-
"""
从数据库读取数据训练模型
使用 MySQL 数据库中的 final_traffic_data 表
"""
import re
import pandas as pd
import joblib
import warnings
import numpy as np
import pymysql
from sqlalchemy import create_engine
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'root',
    'database': 'urban_transport',
    'charset': 'utf8mb4'
}

MODEL_DIR = 'model/'

def load_data_from_db():
    """从 MySQL 数据库加载数据"""
    print('[1/6] 从数据库加载数据...')
    
    # 方法1：使用 SQLAlchemy 直接读取到 DataFrame
    try:
        # 创建数据库连接字符串
        conn_str = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset={DB_CONFIG['charset']}"
        engine = create_engine(conn_str)
        
        # 读取数据
        query = "SELECT * FROM final_traffic_data"
        df = pd.read_sql(query, engine)
        print(f'  成功从数据库加载数据: {len(df)} 行, {len(df.columns)} 列')
        
    except Exception as e:
        print(f'  使用 SQLAlchemy 加载失败: {e}')
        print('  尝试使用 pymysql 加载...')
        
        # 方法2：使用 pymysql 连接
        try:
            conn = pymysql.connect(**DB_CONFIG)
            query = "SELECT * FROM final_traffic_data"
            df = pd.read_sql(query, conn)
            conn.close()
            print(f'  成功从数据库加载数据: {len(df)} 行, {len(df.columns)} 列')
        except Exception as e2:
            print(f'  使用 pymysql 加载失败: {e2}')
            raise Exception('无法从数据库加载数据')
    
    return df

def train_congestion_model(df):
    """训练拥堵状态预测模型（分类）"""
    print('[2/6] 训练拥堵状态预测模型...')
    
    # 使用原始道路名称，不进行合并
    df['road_simple'] = df['road_name'].astype(str)
    n_roads = df['road_simple'].nunique()
    print(f'  唯一道路数（原始）: {n_roads}')
    
    # 解析时间特征
    # 注意：数据库中的 collect_time 是 time 类型，需要转换为 datetime
    df['collect_time'] = pd.to_datetime(df['collect_time'], errors='coerce')
    # 如果 collect_time 有问题，使用当前时间作为占位符
    if df['collect_time'].isnull().any():
        print('  警告: 部分 collect_time 解析失败，使用默认时间')
        df['collect_time'] = pd.Timestamp.now()
    
    df['hour'] = df['collect_time'].dt.hour
    df['day_of_week'] = df['collect_time'].dt.dayofweek  # 0=周一
    
    print('[3/6] 按道路+时间特征聚合...')
    agg_df = df.groupby(
        ['road_simple', 'hour', 'day_of_week', 'season', 'time_period',
         'weather', 'is_peak_hour'],
        as_index=False
    ).agg(
        avg_speed = ('avg_speed', 'mean'),
        congestion_status = ('congestion_status', lambda x: int(round(x.mean()))),
        flow_index = ('flow_index', 'mean'),
        temperature = ('temperature', 'mean'),
        humidity = ('humidity', 'mean'),
    )
    print(f'  聚合后样本数: {len(agg_df)}')
    
    print('[4/6] 编码分类特征...')
    cat_cols = ['road_simple', 'season', 'time_period', 'weather']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        agg_df[col + '_enc'] = le.fit_transform(agg_df[col])
        encoders[col] = le
        print(f'  {col} -> {len(le.classes_)} 类')
    
    # 保存道路列表供前端下拉菜单使用
    road_list = sorted(df['road_simple'].unique().tolist())
    joblib.dump(road_list, MODEL_DIR + 'road_list.pkl')
    print(f'  道路列表已保存: {len(road_list)} 条道路')
    
    # 保存季节/天气/时段选项供前端使用
    options = {
        'season': list(encoders['season'].classes_),
        'time_period': list(encoders['time_period'].classes_),
        'weather': list(encoders['weather'].classes_),
    }
    joblib.dump(options, MODEL_DIR + 'options.pkl')
    
    print('[5/6] 训练随机森林分类模型（预测拥堵状态）...')
    FEATURES = [
        'road_simple_enc', 'hour', 'day_of_week',
        'season_enc', 'time_period_enc', 'weather_enc',
        'is_peak_hour', 'temperature', 'humidity'
    ]
    TARGET = 'congestion_status'
    
    X = agg_df[FEATURES]
    y = agg_df[TARGET]
    
    print(f'  拥堵状态分布: {dict(y.value_counts())}')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'  训练集: {len(X_train)}  测试集: {len(X_test)}')
    
    # 训练分类模型
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    
    print('[6/6] 评估模型性能...')
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'  准确率: {accuracy:.4f}')
    print('  分类报告:')
    print(classification_report(y_test, y_pred))
    
    # 特征重要性
    fi = sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1])
    print('  特征重要性:')
    for f, v in fi:
        print(f'    {f:<28s} {v:.4f}')
    
    print('[7/6] 保存模型...')
    joblib.dump(clf, MODEL_DIR + 'traffic_model.pkl')
    joblib.dump(encoders, MODEL_DIR + 'encoders.pkl')
    joblib.dump(FEATURES, MODEL_DIR + 'features.pkl')
    
    print('  拥堵状态预测模型训练完成！')
    return clf, encoders, FEATURES

def train_flow_model(df):
    """训练车流量预测模型（回归）"""
    print('\n[1/6] 训练车流量预测模型...')
    
    # 使用原始道路名称，不进行合并
    df['road_simple'] = df['road_name'].astype(str)
    
    # 解析时间特征
    df['collect_time'] = pd.to_datetime(df['collect_time'], errors='coerce')
    if df['collect_time'].isnull().any():
        df['collect_time'] = pd.Timestamp.now()
    
    df['hour'] = df['collect_time'].dt.hour
    df['day_of_week'] = df['collect_time'].dt.dayofweek  # 0=周一
    
    print('[2/6] 按道路+时间特征聚合...')
    agg_df = df.groupby(
        ['road_simple', 'hour', 'day_of_week', 'season', 'time_period',
         'weather', 'is_peak_hour'],
        as_index=False
    ).agg(
        avg_speed = ('avg_speed', 'mean'),
        congestion_status = ('congestion_status', lambda x: int(round(x.mean()))),
        flow_index = ('flow_index', 'mean'),  # 车流量指数
        temperature = ('temperature', 'mean'),
        humidity = ('humidity', 'mean'),
    )
    print(f'  聚合后样本数: {len(agg_df)}')
    
    print('[3/6] 编码分类特征...')
    cat_cols = ['road_simple', 'season', 'time_period', 'weather']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        agg_df[col + '_enc'] = le.fit_transform(agg_df[col])
        encoders[col] = le
        print(f'  {col} -> {len(le.classes_)} 类')
    
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
    print(f'  车流量范围: {y.min():.0f} - {y.max():.0f}')
    print(f'  车流量均值: {y.mean():.0f}')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f'  训练集: {len(X_train)}  测试集: {len(X_test)}')
    
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
    y_pred = reg.predict(X_test)
    
    # 确保预测值在合理范围内
    y_pred = np.clip(y_pred, y.min(), y.max())
    
    # 评估指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f'  平均绝对误差 (MAE): {mae:.2f}')
    print(f'  均方根误差 (RMSE): {rmse:.2f}')
    print(f'  R2 分数: {r2:.4f}')
    
    # 计算相对误差（避免除以0）
    mask = y_test > 0
    if mask.any():
        relative_error = np.abs(y_test[mask] - y_pred[mask]) / y_test[mask]
        print(f'  平均相对误差: {relative_error.mean() * 100:.2f}%')
        print(f'  中位数相对误差: {np.median(relative_error) * 100:.2f}%')
    else:
        print('  无法计算相对误差（测试集全为0）')
    
    # 特征重要性
    fi = sorted(zip(FEATURES, reg.feature_importances_), key=lambda x: -x[1])
    print('  特征重要性:')
    for f, v in fi:
        print(f'    {f:<28s} {v:.4f}')
    
    print('[6/6] 保存模型...')
    joblib.dump(reg, MODEL_DIR + 'flow_model.pkl')
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
    
    print('  车流量预测模型训练完成！')
    return reg, encoders, FEATURES

def main():
    """主函数"""
    print('=' * 60)
    print('从数据库训练交通预测模型')
    print('=' * 60)
    
    try:
        # 1. 从数据库加载数据
        df = load_data_from_db()
        
        # 2. 训练拥堵状态预测模型
        train_congestion_model(df)
        
        # 3. 训练车流量预测模型
        train_flow_model(df)
        
        print('\n' + '=' * 60)
        print('模型训练完成！文件已保存到 model/ 目录')
        print('模型文件:')
        print('  - traffic_model.pkl: 拥堵状态预测模型')
        print('  - encoders.pkl: 编码器')
        print('  - features.pkl: 特征列表')
        print('  - flow_model.pkl: 车流量预测模型')
        print('  - flow_encoders.pkl: 车流量编码器')
        print('  - flow_features.pkl: 车流量特征列表')
        print('  - flow_scaler.pkl: 缩放信息')
        print('  - road_list.pkl: 道路列表')
        print('  - options.pkl: 选项配置')
        print('=' * 60)
        
    except Exception as e:
        print(f'\n错误: {e}')
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())