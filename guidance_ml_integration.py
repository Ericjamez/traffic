#!/usr/bin/env python3
"""
疏导功能机器学习集成模块
基于训练数据完善新版guidance功能
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class GuidanceMLModel:
    """疏导机器学习模型"""
    
    def __init__(self, model_path='model/guidance_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.feature_encoders = {}
        self.target_encoder = None
        self.feature_columns = None
        
    def load_training_data(self):
        """加载训练数据"""
        print("加载训练数据...")
        df = pd.read_csv('guidance_training_data.csv')
        print(f"数据形状: {df.shape}")
        return df
    
    def preprocess_features(self, df):
        """预处理特征"""
        print("预处理特征...")
        df_processed = df.copy()
        
        # 特征工程
        df_processed['is_morning_peak'] = df_processed['hour'].apply(lambda x: 1 if 7 <= x <= 9 else 0)
        df_processed['is_evening_peak'] = df_processed['hour'].apply(lambda x: 1 if 17 <= x <= 19 else 0)
        df_processed['is_night'] = df_processed['hour'].apply(lambda x: 1 if 22 <= x or x <= 5 else 0)
        df_processed['speed_flow_ratio'] = df_processed['avg_speed'] / (df_processed['flow_index'] + 0.001)
        df_processed['congestion_severity'] = df_processed['congestion_status'] * df_processed['flow_index']
        
        # 编码分类变量
        encode_cols = ['road_type', 'season', 'weather', 'time_period', 'action_priority']
        
        for col in encode_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
                self.feature_encoders[col] = le
                print(f"  {col} 编码完成，类别数: {len(le.classes_)}")
        
        # 特征列
        self.feature_columns = [
            'congestion_status', 'hour', 'day_of_week', 'temperature', 'humidity',
            'is_peak_hour', 'avg_speed', 'flow_index',
            'is_morning_peak', 'is_evening_peak', 'is_night',
            'speed_flow_ratio', 'congestion_severity'
        ]
        
        # 添加编码后的特征
        for col in encode_cols:
            encoded_col = col + '_encoded'
            if encoded_col in df_processed.columns:
                self.feature_columns.append(encoded_col)
        
        print(f"特征数量: {len(self.feature_columns)}")
        return df_processed
    
    def train_model(self, df_processed):
        """训练模型"""
        print("训练机器学习模型...")
        
        # 准备特征和目标
        X = df_processed[self.feature_columns]
        y = df_processed['action_type']
        
        # 编码目标变量
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        print(f"目标变量类别: {list(self.target_encoder.classes_)}")
        
        # 训练随机森林模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X, y_encoded)
        print("模型训练完成")
        
        # 保存模型
        self.save_model()
        
        return self.model
    
    def save_model(self):
        """保存模型和编码器"""
        print(f"保存模型到: {self.model_path}")
        model_data = {
            'model': self.model,
            'feature_encoders': self.feature_encoders,
            'target_encoder': self.target_encoder,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, self.model_path)
        print("模型保存成功")
    
    def load_model(self):
        """加载模型"""
        print(f"加载模型: {self.model_path}")
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_encoders = model_data['feature_encoders']
            self.target_encoder = model_data['target_encoder']
            self.feature_columns = model_data['feature_columns']
            print("模型加载成功")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def predict_guidance(self, input_data):
        """预测疏导建议"""
        if self.model is None:
            if not self.load_model():
                print("模型未加载，请先训练模型")
                return None
        
        try:
            # 创建特征DataFrame
            features = {}
            
            # 基本特征
            for col in ['congestion_status', 'hour', 'day_of_week', 'temperature', 
                       'humidity', 'is_peak_hour', 'avg_speed', 'flow_index']:
                if col in input_data:
                    features[col] = input_data[col]
                else:
                    # 使用默认值
                    defaults = {
                        'congestion_status': 2,
                        'hour': 8,
                        'day_of_week': 1,
                        'temperature': 20.0,
                        'humidity': 70,
                        'is_peak_hour': 1,
                        'avg_speed': 30.0,
                        'flow_index': 0.7
                    }
                    features[col] = defaults.get(col, 0)
            
            # 计算衍生特征
            hour = features['hour']
            features['is_morning_peak'] = 1 if 7 <= hour <= 9 else 0
            features['is_evening_peak'] = 1 if 17 <= hour <= 19 else 0
            features['is_night'] = 1 if 22 <= hour or hour <= 5 else 0
            features['speed_flow_ratio'] = features['avg_speed'] / (features['flow_index'] + 0.001)
            features['congestion_severity'] = features['congestion_status'] * features['flow_index']
            
            # 编码分类特征
            for col, encoder in self.feature_encoders.items():
                if col in input_data:
                    try:
                        encoded_value = encoder.transform([input_data[col]])[0]
                    except:
                        # 如果值不在训练集中，使用第一个类别
                        encoded_value = 0
                else:
                    # 使用默认值
                    defaults = {
                        'road_type': '主干道',
                        'season': 'spring',
                        'weather': '晴',
                        'time_period': 'early_peak',
                        'action_priority': 'medium'
                    }
                    default_value = defaults.get(col, encoder.classes_[0])
                    encoded_value = encoder.transform([default_value])[0]
                
                features[col + '_encoded'] = encoded_value
            
            # 创建特征向量
            feature_vector = []
            for col in self.feature_columns:
                if col in features:
                    feature_vector.append(features[col])
                else:
                    print(f"警告: 特征 {col} 不存在，使用默认值0")
                    feature_vector.append(0)
            
            # 预测
            X = np.array([feature_vector])
            prediction_encoded = self.model.predict(X)[0]
            prediction_label = self.target_encoder.inverse_transform([prediction_encoded])[0]
            
            # 获取预测概率
            probabilities = self.model.predict_proba(X)[0]
            confidence = max(probabilities) * 100
            
            return {
                'action_type': prediction_label,
                'confidence': round(confidence, 1),
                'all_probabilities': {
                    self.target_encoder.inverse_transform([i])[0]: round(prob, 3)
                    for i, prob in enumerate(probabilities)
                }
            }
            
        except Exception as e:
            print(f"预测失败: {e}")
            return None
    
    def get_action_details(self, action_type):
        """获取措施详细信息"""
        action_details = {
            'signal_adjust': {
                'name': '信号调整',
                'description': '调整交通信号灯配时方案',
                'implementation': '延长绿灯时间10-30秒，优化相位差',
                'expected_effect': '减少延误5-15分钟',
                'cost': '低',
                'time_to_implement': '5分钟'
            },
            'traffic_control': {
                'name': '交通管制',
                'description': '限制入口流量或实施临时管制',
                'implementation': '减少20-30%进入车辆，设置临时路障',
                'expected_effect': '降低拥堵指数0.3-0.6',
                'cost': '中',
                'time_to_implement': '15分钟'
            },
            'alternative_route': {
                'name': '替代路线',
                'description': '引导车辆使用周边替代道路',
                'implementation': '发布绕行路线，调整导航系统',
                'expected_effect': '分流20-40%车流',
                'cost': '低',
                'time_to_implement': '10分钟'
            },
            'forced_diversion': {
                'name': '强制分流',
                'description': '强制车辆绕行，关闭部分入口',
                'implementation': '设置强制绕行标志，关闭拥堵路段入口',
                'expected_effect': '减少40-60%进入车辆',
                'cost': '高',
                'time_to_implement': '20分钟'
            },
            'emergency_plan': {
                'name': '应急预案',
                'description': '启动应急预案，协调多部门',
                'implementation': '多部门协同，启动应急指挥系统',
                'expected_effect': '防止拥堵扩散，快速恢复',
                'cost': '高',
                'time_to_implement': '30分钟'
            },
            'public_notice': {
                'name': '公众通知',
                'description': '发布拥堵预警和出行建议',
                'implementation': '通过媒体、APP推送通知',
                'expected_effect': '减少15-25%私家车出行',
                'cost': '低',
                'time_to_implement': '5分钟'
            },
            'info_publish': {
                'name': '信息发布',
                'description': '发布实时路况和疏导信息',
                'implementation': '通过VMS、广播发布信息',
                'expected_effect': '引导10-20%车流错峰',
                'cost': '低',
                'time_to_implement': '5分钟'
            },
            'police_dispatch': {
                'name': '警力调度',
                'description': '增派交警现场指挥疏导',
                'implementation': '增派2-4名交警到现场',
                'expected_effect': '提升通行效率10-20%',
                'cost': '中',
                'time_to_implement': '15分钟'
            },
            'monitor': {
                'name': '监控观察',
                'description': '保持监控，无需特殊措施',
                'implementation': '正常监控，定期报告',
                'expected_effect': '无',
                'cost': '无',
                'time_to_implement': '0分钟'
            }
        }
        
        return action_details.get(action_type, {
            'name': '未知措施',
            'description': '未知疏导措施',
            'implementation': '待确定',
            'expected_effect': '待评估',
            'cost': '未知',
            'time_to_implement': '未知'
        })

def integrate_with_guidance_new():
    """集成到新版guidance功能"""
    print("=== 集成到新版guidance功能 ===")
    
    # 1. 创建或加载模型
    ml_model = GuidanceMLModel()
    
    if not ml_model.load_model():
        print("模型不存在，开始训练新模型...")
        df = ml_model.load_training_data()
        df_processed = ml_model.preprocess_features(df)
        ml_model.train_model(df_processed)
    
    # 2. 测试预测功能
    print("\n=== 测试预测功能 ===")
    
    test_cases = [
        {
            'road_type': '主干道',
            'congestion_status': 3,
            'hour': 8,
            'day_of_week': 1,
            'season': 'spring',
            'weather': '晴',
            'temperature': 25.5,
            'humidity': 65,
            'time_period': 'early_peak',
            'is_peak_hour': 1,
            'avg_speed': 18.2,
            'flow_index': 0.85
        },
        {
            'road_type': '商业区道路',
            'congestion_status': 4,
            'hour': 18,
            'day_of_week': 4,
            'season': 'autumn',
            'weather': '小雨',
            'temperature': 20.3,
            'humidity': 78,
            'time_period': 'late_peak',
            'is_peak_hour': 1,
            'avg_speed': 8.5,
            'flow_index': 0.92
        },
        {
            'road_type': '交通枢纽道路',
            'congestion_status': 1,
            'hour': 22,
            'day_of_week': 6,
            'season': 'summer',
            'weather': '晴',
            'temperature': 28.5,
            'humidity': 65,
            'time_period': 'night',
            'is_peak_hour': 0,
            'avg_speed': 55.8,
            'flow_index': 0.25
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}:")
        print(f"  拥堵状态: {test_case['congestion_status']}")
        print(f"  时间: {test_case['hour']}:00")
        print(f"  道路类型: {test_case['road_type']}")
        
        prediction = ml_model.predict_guidance(test_case)
        if prediction:
            print(f"  推荐措施: {prediction['action_type']}")
            print(f"  置信度: {prediction['confidence']}%")
            
            # 获取措施详情
            details = ml_model.get_action_details(prediction['action_type'])
            print(f"  措施名称: {details['name']}")
            print(f"  描述: {details['description']}")
            print(f"  预期效果: {details['expected_effect']}")
    
    # 3. 生成集成代码
    print("\n=== 生成集成代码 ===")
    
    integration_code = '''
# 在app.py中添加以下代码

from guidance_ml_integration import GuidanceMLModel

# 初始化模型（在应用启动时）
guidance_ml_model = GuidanceMLModel()
if not guidance_ml_model.load_model():
    # 如果模型不存在，可以在这里训练或使用默认规则
    pass

@app.route('/api/guidance/ml_predict', methods=['POST'])
@login_required
def api_guidance_ml_predict():
    """机器学习疏导建议预测"""
    try:
        data = request.get_json()
        
        # 提取特征
        input_features = {
            'road_type': data.get('road_type', '主干道'),
            'congestion_status': int(data.get('congestion_status', 2)),
            'hour': int(data.get('hour', datetime.datetime.now().hour)),
            'day_of_week': int(data.get('day_of_week', datetime.datetime.now().weekday())),
            'season': data.get('season', _month_to_season(datetime.datetime.now().month)),
            'weather': data.get('weather', '晴'),
            'temperature': float(data.get('temperature', 20.0)),
            'humidity': int(data.get('humidity', 70)),
            'time_period': data.get('time_period', _hour_to_time_period(int(data.get('hour', 8)))),
            'is_peak_hour': 1 if data.get('time_period') in ['early_peak', 'late_peak'] else 0,
            'avg_speed': float(data.get('avg_speed', 30.0)),
            'flow_index': float(data.get('flow_index', 0.7))
        }
        
        # 预测
        prediction = guidance_ml_model.predict_guidance(input_features)
        
        if prediction:
            # 获取措施详情
            details = guidance_ml_model.get_action_details(prediction['action_type'])
            
            return jsonify({
                'success': True,
                'action_type': prediction['action_type'],
                'action_name': details['name'],
                'description': details['description'],
                'implementation': details['implementation'],
                'expected_effect': details['expected_effect'],
                'cost': details['cost'],
                'time_to_implement': details['time_to_implement'],
                'confidence': prediction['confidence'],
                'all_options': prediction['all_probabilities']
            })
        else:
            return jsonify({'success': False, 'msg': '预测失败'})
            
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'msg': str(e), 'trace': traceback.format_exc()})

# 在guidance_new.html中添加机器学习预测按钮
'''
    
    print("集成代码已生成，可以添加到app.py中")
    
    # 4. 创建API测试脚本
    print("\n=== 创建API测试脚本 ===")
    
    api_test_code = '''# guidance_ml_api_test.py
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_ml_prediction():
    """测试机器学习预测API"""
    test_data = {
        "road_type": "主干道",
        "congestion_status": 3,
        "hour": 8,
        "day_of_week": 1,
        "season": "spring",
        "weather": "晴",
        "temperature": 25.5,
        "humidity": 65,
        "time_period": "early_peak",
        "avg_speed": 18.2,
        "flow_index": 0.85
    }
    
    response = requests.post(
        f"{BASE_URL}/api/guidance/ml_predict",
        json=test_data,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print("预测成功:")
            print(f"  推荐措施: {result.get('action_name')}")
            print(f"  描述: {result.get('description')}")
            print(f"  预期效果: {result.get('expected_effect')}")
            print(f"  置信度: {result.get('confidence')}%")
        else:
            print(f"预测失败: {result.get('msg')}")
    else:
        print(f"API请求失败: {response.status_code}")

if __name__ == "__main__":
    test_ml_prediction()'''
    
    print("API测试脚本已生成")
    
    # 5. 创建前端集成代码
    print("\n=== 创建前端集成代码 ===")
    
    frontend_code = '''// 在guidance_new.html中添加机器学习预测功能

// 机器学习预测函数
async function predictMLGuidance() {
    const road = document.getElementById('roadSelect').value;
    const hour = document.getElementById('hourSelect').value;
    const congestion = document.getElementById('congestionLevel').value;
    
    if (!road) {
        showToast('请选择道路', 'warning');
        return;
    }
    
    // 获取道路类型（可以从道路数据中获取）
    const roadType = getRoadType(road);
    
    // 构建请求数据
    const requestData = {
        road_type: roadType,
        congestion_status: parseInt(congestion),
        hour: parseInt(hour),
        day_of_week: new Date().getDay(),
        season: getCurrentSeason(),
        weather: getCurrentWeather(),
        temperature: getCurrentTemperature(),
        humidity: getCurrentHumidity(),
        time_period: getTimePeriod(hour),
        avg_speed: getRoadSpeed(road),
        flow_index: getRoadFlowIndex(road)
    };
    
    try {
        showLoading('正在使用AI分析最佳疏导方案...');
        
        const response = await fetch('/api/guidance/ml_predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            // 显示AI推荐
            displayMLRecommendation(result);
            
            // 添加到历史记录
            addToHistory({
                type: 'ai_recommendation',
                road: road,
                recommendation: result,
                timestamp: new Date().toISOString()
            });
            
            showToast('AI疏导建议已生成', 'success');
        } else {
            showToast(`AI预测失败: ${result.msg}`, 'error');
        }
    } catch (error) {
        console.error('AI预测错误:', error);
        showToast('AI服务暂时不可用', 'error');
    } finally {
        hideLoading();
    }
}

// 显示AI推荐
function displayMLRecommendation(result) {
    const container = document.getElementById('mlRecommendation');
    
    const html = `
        <div class="ai-recommendation-card">
            <div class="ai-header">
                <i class="fas fa-robot"></i>
                <h4>AI智能推荐</h4>
                <span class="confidence-badge">置信度: ${result.confidence}%</span>
            </div>
            <div class="ai-content">
                <h5>${result.action_name}</h5>
                <p>${result.description}</p>
                <div class="ai-details">
                    <div class="detail-item">
                        <span class="label">实施方法:</span>
                        <span>${result.implementation}</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">预期效果:</span>
                        <span class="effect-highlight">${result.expected_effect}</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">实施成本:</span>
                        <span class="cost-${result.cost}">${result.cost}</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">实施时间:</span>
                        <span>${result.time_to_implement}</span>
                    </div>
                </div>
                <div class="ai-actions">
                    <button class="btn btn-primary" onclick="applyMLRecommendation('${result.action_type}')">
                        <i class="fas fa-check"></i> 采用此建议
                    </button>
                    <button class="btn btn-outline-secondary" onclick="showAllOptions(${JSON.stringify(result.all_options)})">
                        <i class="fas fa-list"></i> 查看所有选项
                    </button>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
    container.style.display = 'block';
}

// 在页面中添加AI预测按钮
const aiButtonHtml = `
    <div class="ai-prediction-section">
        <h4><i class="fas fa-brain"></i> AI智能疏导</h4>
        <p>基于机器学习模型分析历史数据，推荐最优疏导方案</p>
        <button class="btn btn-ai" onclick="predictMLGuidance()">
            <i class="fas fa-robot"></i> AI智能分析
        </button>
        <div id="mlRecommendation" style="display: none;"></div>
    </div>
`;

// 将AI按钮添加到页面合适的位置
document.querySelector('.guidance-controls').insertAdjacentHTML('beforeend', aiButtonHtml);'''
    
    print("前端集成代码已生成")
    
    # 6. 保存集成文档
    print("\n=== 保存集成文档 ===")
    
    with open('guidance_ml_integration_guide.md', 'w', encoding='utf-8') as f:
        f.write(f'''# 疏导功能机器学习集成指南

## 概述
基于训练数据 `guidance_training_data.csv`，将机器学习模型集成到新版疏导功能中。

## 文件结构
- `guidance_ml_integration.py` - 机器学习模型类
- `model/guidance_model.pkl` - 训练好的模型文件
- `guidance_training_data.csv` - 训练数据
- `guidance_ml_api_test.py` - API测试脚本

## 集成步骤

### 1. 训练模型
```python
from guidance_ml_integration import GuidanceMLModel

ml_model = GuidanceMLModel()
df = ml_model.load_training_data()
df_processed = ml_model.preprocess_features(df)
ml_model.train_model(df_processed)
```

### 2. 在app.py中添加API路由
```python
{integration_code}
```

### 3. 在前端添加AI功能
将前端代码添加到 `templates/guidance_new.html` 中。

### 4. 测试API
运行测试脚本验证功能：
```bash
python guidance_ml_api_test.py
```

## 功能特点

### 1. 智能预测
- 基于42条历史训练数据
- 考虑18个特征维度
- 输出9种疏导措施推荐
- 提供置信度评分

### 2. 措施详情
每种措施包含：
- 名称和描述
- 实施方法
- 预期效果
- 成本评估
- 实施时间

### 3. 用户体验
- AI分析按钮
- 实时预测结果
- 置信度显示
- 一键采用建议
- 历史记录保存

## 模型性能
- 训练数据: 42条记录
- 特征数量: 18个
- 目标类别: 9种措施
- 算法: 随机森林
- 置信度: 基于预测概率

## 扩展建议

### 1. 数据增强
- 收集更多真实疏导记录
- 增加道路拓扑特征
- 加入实时交通流数据

### 2. 模型优化
- 尝试XGBoost、神经网络
- 加入时间序列分析
- 实现在线学习

### 3. 功能扩展
- A/B测试框架
- 效果反馈收集
- 多模型集成
- 实时优化建议

## 注意事项
1. 当前模型基于有限数据训练，建议持续收集数据优化
2. AI建议仅供参考，实际决策需结合专家经验
3. 定期重新训练模型以适应变化
4. 监控模型性能，及时调整参数

## 技术支持
如有问题，请联系项目负责人。
''')
    
    print("集成指南已保存到: guidance_ml_integration_guide.md")
    
    return ml_model

if __name__ == "__main__":
    # 运行集成测试
    integrate_with_guidance_new()
       