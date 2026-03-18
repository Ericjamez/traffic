# -*- coding: utf-8 -*-
"""
Traffic prediction model training script
Random Forest classifier: predict congestion_status (1~4)
"""
import re
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

CSV_PATH  = 'static/data/final_traffic_data.csv'
MODEL_DIR = 'model/'

print('[1/6] Loading data...')
df = pd.read_csv(CSV_PATH)
print('  Rows: %d  Cols: %d' % df.shape)

# 使用原始道路名称，不进行合并
df['road_simple'] = df['road_name'].astype(str)
n_roads = df['road_simple'].nunique()
print('  Unique roads (original): %d' % n_roads)

# Parse time features
df['collect_time'] = pd.to_datetime(df['collect_time'])
df['hour']         = df['collect_time'].dt.hour
df['day_of_week']  = df['collect_time'].dt.dayofweek  # 0=Mon

print('[2/6] Aggregating by road + time features...')
agg_df = df.groupby(
    ['road_simple', 'hour', 'day_of_week', 'season', 'time_period',
     'weather', 'is_peak_hour'],
    as_index=False
).agg(
    avg_speed         = ('avg_speed',         'mean'),
    congestion_status = ('congestion_status',  lambda x: int(round(x.mean()))),
    temperature       = ('temperature',        'mean'),
    humidity          = ('humidity',           'mean'),
)
print('  Aggregated samples: %d' % len(agg_df))

print('[3/6] Encoding categorical features...')
cat_cols = ['road_simple', 'season', 'time_period', 'weather']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    agg_df[col + '_enc'] = le.fit_transform(agg_df[col])
    encoders[col] = le
    print('  %s -> %d classes: %s' % (col, len(le.classes_), list(le.classes_)))

# Save road list for frontend dropdown
road_list = sorted(df['road_simple'].unique().tolist())
joblib.dump(road_list, MODEL_DIR + 'road_list.pkl')
print('  Road list saved: %d roads' % len(road_list))

# Save season/weather/time_period options for frontend
options = {
    'season':      list(encoders['season'].classes_),
    'time_period': list(encoders['time_period'].classes_),
    'weather':     list(encoders['weather'].classes_),
}
joblib.dump(options, MODEL_DIR + 'options.pkl')

print('[4/6] Training Random Forest...')
FEATURES = [
    'road_simple_enc', 'hour', 'day_of_week',
    'season_enc', 'time_period_enc', 'weather_enc',
    'is_peak_hour', 'temperature', 'humidity'
]
TARGET = 'congestion_status'

X = agg_df[FEATURES]
# 反转标签：数据中1=严重拥堵，4=畅通，但代码中1=畅通，4=严重拥堵
# 所以需要将 1→4, 2→3, 3→2, 4→1
y = agg_df[TARGET].clip(1, 4)
y = 5 - y  # 反转：1变成4，2变成3，3变成2，4变成1
print('  Label mapping: 1(严重拥堵)→4, 2(拥堵)→3, 3(缓行)→2, 4(畅通)→1')

# 计算类别权重来处理不平衡数据
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# 使用数据中实际存在的类别
unique_classes = np.unique(y)
print('  Unique classes in data:', unique_classes)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}
print('  Class weights:', class_weight_dict)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print('  Train: %d  Test: %d' % (len(X_train), len(X_test)))

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    class_weight=class_weight_dict
)
clf.fit(X_train, y_train)

print('[5/6] Evaluating...')
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('  Accuracy: %.4f' % acc)
# 注意：标签已经反转，现在 1=畅通, 2=缓行, 3=拥堵, 4=严重拥堵
status_map = {1:'畅通(Free)', 2:'缓行(Slow)', 3:'拥堵(Jam)', 4:'严重拥堵(Heavy)'}
labels_present = sorted(set(y_test) | set(y_pred))
print(classification_report(y_test, y_pred,
      labels=labels_present,
      target_names=[status_map[l] for l in labels_present]))

fi = sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1])
print('  Feature importances:')
for f, v in fi:
    print('    %-28s %.4f' % (f, v))

print('[6/6] Saving model...')
joblib.dump(clf,      MODEL_DIR + 'traffic_model.pkl')
joblib.dump(encoders, MODEL_DIR + 'encoders.pkl')
joblib.dump(FEATURES, MODEL_DIR + 'features.pkl')
print('  Done! Files saved to model/')
