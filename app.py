from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import datetime
import random
import re
import io
import os
import string
import smtplib
import json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# 预导入数据处理库（避免函数内懒导入导致循环引用）
import numpy as np
import pandas as pd

# ============================================================
# 初始化 Flask 应用
# ============================================================
app = Flask(__name__)
# ============================================================
# 加载交通预测模型（启动时一次性加载）
# ============================================================
_MODEL = None  # 拥堵状态预测模型
_FLOW_MODEL = None  # 车流量预测模型
_ENCODERS = None
_FLOW_ENCODERS = None
_ROAD_LIST = None
_OPTIONS = None
_FEATURES = None
_FLOW_FEATURES = None
_FLOW_SCALER = None
_TREND_CACHE = {}  # 道路趋势数据缓存，避免重复读取 CSV
_ROAD_ADCODE = {}  # road_name → 高德区级 adcode（精准天气）


def _load_model():
    global _MODEL, _FLOW_MODEL, _ENCODERS, _FLOW_ENCODERS, _ROAD_LIST, _OPTIONS, _FEATURES, _FLOW_FEATURES, _FLOW_SCALER
    try:
        import joblib
        # 加载拥堵状态预测模型
        _MODEL = joblib.load('model/traffic_model.pkl')
        _ENCODERS = joblib.load('model/encoders.pkl')
        _FEATURES = joblib.load('model/features.pkl')
        print('[Model] Traffic congestion model loaded OK')

        # 加载车流量预测模型
        try:
            _FLOW_MODEL = joblib.load('model/flow_model.pkl')
            _FLOW_ENCODERS = joblib.load('model/flow_encoders.pkl')
            _FLOW_FEATURES = joblib.load('model/flow_features.pkl')
            _FLOW_SCALER = joblib.load('model/flow_scaler.pkl')
            print('[Model] Traffic flow model loaded OK')
        except Exception as e:
            print('[Model] Warning: could not load flow model:', e)
            _FLOW_MODEL = None

        # 加载共享数据
        _ROAD_LIST = joblib.load('model/road_list.pkl')
        _OPTIONS = joblib.load('model/options.pkl')

    except Exception as e:
        print('[Model] Warning: could not load model:', e)


_load_model()


def _init_road_adcodes():
    """从缓存加载或后台异步初始化道路 adcode（精准天气）- 不阻塞启动"""
    global _ROAD_ADCODE
    import threading

    cache_file = 'model/road_adcodes.pkl'

    # 1. 优先从缓存加载（毫秒级）
    try:
        import joblib
        _ROAD_ADCODE = joblib.load(cache_file)
        print('[Model] Road adcodes loaded from cache')
        return
    except:
        pass

    # 2. 缓存不存在时，后台线程异步初始化（不阻塞启动）
    def _init_async():
        global _ROAD_ADCODE
        import urllib.request, json as _json
        try:
            df = pd.read_csv('static/data/final_traffic_data.csv')
            df['road_simple'] = df['road_name'].astype(str)
            coord_df = df.groupby('road_simple').agg(lng=('lng', 'mean'), lat=('lat', 'mean'))

            temp_adcodes = {}
            for road, row in coord_df.iterrows():
                location = f"{row['lng']:.6f},{row['lat']:.6f}"
                url = (f'https://restapi.amap.com/v3/geocode/regeo'
                       f'?key={AMAP_API_KEY}&location={location}&extensions=base&output=JSON')
                try:
                    with urllib.request.urlopen(url, timeout=4) as resp:
                        data = _json.loads(resp.read().decode('utf-8'))
                    if data.get('status') == '1':
                        adcode = data['regeocode']['addressComponent'].get('adcode', AMAP_CITY_CODE)
                        temp_adcodes[road] = str(adcode)
                    else:
                        temp_adcodes[road] = AMAP_CITY_CODE
                except Exception:
                    temp_adcodes[road] = AMAP_CITY_CODE

            # 更新全局变量并缓存
            _ROAD_ADCODE = temp_adcodes
            try:
                import joblib
                joblib.dump(_ROAD_ADCODE, cache_file)
                print('[Model] Road adcodes initialized and cached')
            except Exception as cache_e:
                print(f'[Model] Warning: could not cache adcodes: {cache_e}')

        except Exception as e:
            print(f'[Model] Warning: async adcode init failed: {e}')

    # 启动后台线程
    print('[Model] Road adcodes initializing in background...')
    t = threading.Thread(target=_init_async, daemon=True)
    t.start()


# 配置密钥
app.secret_key = 'traffic_system_2026'

# ============================================================
# MySQL 数据库配置
# ============================================================
app.config['SQLALCHEMY_DATABASE_URI'] = (
    'mysql+pymysql://root:root@localhost:3306/urban_transport'
    '?charset=utf8mb4'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Session 持久化时长（记住我：7天）
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=7)

# ============================================================
# 高德地图 API 配置（天气查询）
# ============================================================
AMAP_API_KEY = "488b37f9a1bd9b6874909721d8b2fb98"
AMAP_CITY_CODE = '420100'  # 武汉市

# 高德天气描述 → 模型天气类别（6类：晴/多云/阴/小雨/小雪/雷阵雨）
_AMAP_WEATHER_MAP = {
    '晴': '晴', '少云': '晴', '有风': '晴', '平静': '晴', '微风': '晴',
    '和风': '晴', '清风': '晴', '强风/劲风': '晴', '疾风': '晴',
    '晴间多云': '多云', '多云': '多云',
    '阴': '阴', '霾': '阴', '中度霾': '阴', '重度霾': '阴', '严重霾': '阴',
    '雾': '阴', '浓雾': '阴', '强浓雾': '阴', '中雾': '阴', '轻雾': '阴',
    '大雾': '阴', '特强浓雾': '阴', '浮尘': '阴', '扬沙': '阴',
    '沙尘暴': '阴', '强沙尘暴': '阴',
    '阵雨': '小雨', '小雨': '小雨', '毛毛雨/细雨': '小雨', '小到中雨': '小雨',
    '中雨': '小雨', '中到大雨': '小雨', '大雨': '小雨',
    '暴雨': '小雨', '大暴雨': '小雨', '特大暴雨': '小雨', '冻雨': '小雨',
    '雷阵雨': '雷阵雨', '雷阵雨并伴有冰雹': '雷阵雨', '冰雹': '雷阵雨',
    '雨夹雪': '小雪', '雨雪天气': '小雪', '阵雨夹雪': '小雪',
    '小雪': '小雪', '中雪': '小雪', '大雪': '小雪', '暴雪': '小雪',
}


def _map_amap_weather(desc: str) -> str:
    """高德天气描述 → 模型 weather 类别"""
    if desc in _AMAP_WEATHER_MAP:
        return _AMAP_WEATHER_MAP[desc]
    # 模糊兜底
    if '雪' in desc:  return '小雪'
    if '雷' in desc:  return '雷阵雨'
    if '雨' in desc:  return '小雨'
    if '霾' in desc or '雾' in desc or '尘' in desc: return '阴'
    if '云' in desc:  return '多云'
    return '晴'


def _get_status_color(status: str) -> str:
    """根据通行状态获取对应的颜色"""
    color_map = {
        "畅通": "#28a745",  # 绿色
        "缓行": "#ffc107",  # 黄色
        "拥堵": "#fd7e14",  # 橙色
        "严重拥堵": "#dc3545",  # 红色
        "未知": "#6c757d"  # 灰色
    }
    return color_map.get(status, "#6c757d")


# 初始化道路 adcode 映射
_init_road_adcodes()

# ============================================================
# 实时路况 API 配置（复用 static/traffic_conditions/app.py 逻辑）
# ============================================================
# 直接复用你的10个路口（只提取关键信息）
REALTIME_COLLECT_POINTS = [
    {"name": "楚河汉街", "rectangle": "114.31306,30.54041;114.32066,30.54541", "base_flow": 3000},
    {"name": "江汉路步行街", "rectangle": "114.29006,30.58041;114.29766,30.58541", "base_flow": 3500},
    {"name": "武广商圈", "rectangle": "114.29306,30.57041;114.30066,30.57541", "base_flow": 2800},
    {"name": "解放大道（同济段）", "rectangle": "114.28306,30.56041;114.29066,30.56541", "base_flow": 4000},
    {"name": "中山路（武昌火车站段）", "rectangle": "114.35306,30.52041;114.36066,30.52541", "base_flow": 4500},
    {"name": "雄楚大道（光谷段）", "rectangle": "114.40306,30.50041;114.41066,30.50541", "base_flow": 3800},
    {"name": "长江二桥（武昌段）", "rectangle": "114.33306,30.58041;114.34066,30.58541", "base_flow": 5000},
    {"name": "长江大桥（汉阳段）", "rectangle": "114.28306,30.55041;114.29066,30.55541", "base_flow": 4800},
    {"name": "武汉站周边", "rectangle": "114.38306,30.64041;114.39066,30.64541", "base_flow": 3200},
    {"name": "汉口站周边", "rectangle": "114.27306,30.60041;114.28066,30.60541", "base_flow": 3600},
]
# 复用你的流量计算逻辑
FLOW_TIME_FACTOR = {
    0: 0.2, 1: 0.15, 2: 0.1, 3: 0.1, 4: 0.15, 5: 0.3,
    6: 0.6, 7: 0.9, 8: 1.0, 9: 0.8, 10: 0.7, 11: 0.75,
    12: 0.8, 13: 0.7, 14: 0.65, 15: 0.7, 16: 0.8, 17: 0.95,
    18: 1.0, 19: 0.9, 20: 0.7, 21: 0.5, 22: 0.3, 23: 0.25
}
FLOW_CONGEST_FACTOR = {1: 1.0, 2: 1.2, 3: 1.5, 4: 1.8}


def calculate_traffic_flow(base_flow, hour, congestion_status, avg_speed):
    """复用流量计算函数"""
    time_factor = FLOW_TIME_FACTOR.get(hour, 0.5)
    congest_factor = FLOW_CONGEST_FACTOR.get(congestion_status, 1.0)
    speed_factor = 1.5 if avg_speed < 10 else 1.2 if avg_speed < 20 else 1.0 if avg_speed < 30 else 0.8
    random_factor = random.uniform(0.9, 1.1)
    flow = int(base_flow * time_factor * congest_factor * speed_factor * random_factor)
    flow = max(500, min(8000, flow))
    return flow


def get_realtime_traffic(point):
    """复用高德接口调用逻辑，获取10个路口的真实数据"""
    import requests
    import datetime
    import random

    # 首先尝试从高德API获取真实数据
    url = "https://restapi.amap.com/v3/traffic/status/rectangle"
    params = {
        "key": AMAP_API_KEY,
        "rectangle": point["rectangle"],
        "extensions": "all"
    }

    try:
        # 禁用SSL验证（和采集代码保持一致）
        res = requests.get(url, params=params, timeout=10, verify=False)
        data = res.json()

        if data.get("status") == "1":
            roads = data.get("trafficinfo", {}).get("roads", []) or data.get("roads", [])
            if roads:
                # 取第一条道路数据（和采集逻辑一致）
                road = roads[0]
                avg_speed = float(road.get("speed", 0))
                status_text = road.get("status", "未知")

                # 拥堵状态映射（和采集代码一致）
                status_map = {"畅通": 1, "缓行": 2, "拥堵": 3, "严重拥堵": 4}
                congestion_status = status_map.get(status_text, 2)

                # 计算流量（复用核心逻辑）
                current_hour = datetime.datetime.now().hour
                traffic_flow = calculate_traffic_flow(point["base_flow"], current_hour, congestion_status, avg_speed)

                return {
                    "speed": avg_speed,
                    "status": status_text,
                    "flow": traffic_flow
                }
    except Exception as e:
        print(f"获取{point['name']}高德数据失败：{e}")

    # 如果高德API失败，返回模拟数据
    print(f"使用模拟数据 for {point['name']}")
    current_hour = datetime.datetime.now().hour

    # 根据时间生成合理的模拟数据
    if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
        # 高峰时段
        speed = random.uniform(10, 30)
        status_options = ["缓行", "拥堵", "严重拥堵"]
        status = random.choice(status_options)
        base_flow = point["base_flow"]
    elif 22 <= current_hour or current_hour <= 5:
        # 夜间时段
        speed = random.uniform(50, 70)
        status = "畅通"
        base_flow = point["base_flow"] * 0.3
    else:
        # 平峰时段
        speed = random.uniform(30, 50)
        status_options = ["畅通", "缓行"]
        status = random.choice(status_options)
        base_flow = point["base_flow"] * 0.7

    # 计算流量
    status_map = {"畅通": 1, "缓行": 2, "拥堵": 3, "严重拥堵": 4}
    congestion_status = status_map.get(status, 2)
    traffic_flow = calculate_traffic_flow(base_flow, current_hour, congestion_status, speed)

    return {
        "speed": round(speed, 1),
        "status": status,
        "flow": traffic_flow
    }


# ============================================================
# QQ 邮箱 SMTP 配置（直接使用 smtplib，无需 Flask-Mail）
# ============================================================
MAIL_SMTP_HOST = 'smtp.qq.com'
MAIL_SMTP_PORT = 465
MAIL_USERNAME = '3124418793@qq.com'  # ← 您的 QQ 邮箱
MAIL_PASSWORD = 'jlwommqicavbdhcj'  # ← SMTP 授权码

# 初始化扩展
db = SQLAlchemy(app)


# ============================================================
# 数据库模型
# ============================================================

class User(db.Model):
    """用户表"""
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='user')
    email = db.Column(db.String(100), unique=True, nullable=True)
    is_email_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)


class VerificationCode(db.Model):
    """邮箱验证码表"""
    __tablename__ = 'verification_code'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), nullable=False)
    code = db.Column(db.String(6), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    is_used = db.Column(db.Boolean, default=False)
    purpose = db.Column(db.String(20), default='register')  # 'register' 或 'reset_password'
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)


class GuidancePlan(db.Model):
    """疏导方案表"""
    __tablename__ = 'guidance_plan'
    id = db.Column(db.Integer, primary_key=True)
    road = db.Column(db.String(100), nullable=False)
    status = db.Column(db.Integer, nullable=False)  # 触发时拥堵等级 1-4
    plan_type = db.Column(db.String(20), default='auto')  # auto/manual
    actions = db.Column(db.Text, nullable=False)  # JSON 格式的措施列表
    operator = db.Column(db.String(50), nullable=True)  # 操作人（人工干预时）
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)
    activated_at = db.Column(db.DateTime, nullable=True)
    reverted_at = db.Column(db.DateTime, nullable=True)


def calculate_plan_score(status, plan_type):
    """计算方案评分（基于状态和类型）"""
    base_score = 80

    # 根据拥堵状态调整评分
    if status == 1:
        base_score += 10  # 畅通
    elif status == 2:
        base_score += 5  # 缓行
    elif status == 3:
        base_score -= 5  # 拥堵
    elif status == 4:
        base_score -= 10  # 严重拥堵

    # 根据方案类型调整评分
    if plan_type == 'manual':
        base_score += 5  # 人工调整方案质量更高

    # 确保分数在合理范围内
    return max(60, min(95, base_score))


def calculate_plan_effect(status):
    """计算预计效果"""
    effect_map = {1: 20, 2: 30, 3: 40, 4: 50}
    return effect_map.get(status, 30)


def calculate_plan_cost(status, plan_type):
    """计算实施成本"""
    costs = ['低', '中低', '中等', '中高', '高']
    cost_index = 2  # 默认中等

    if status == 4:
        cost_index += 1  # 严重拥堵成本高
    if plan_type == 'manual':
        cost_index += 1  # 人工调整成本高

    cost_index = max(0, min(4, cost_index))
    return costs[cost_index]


def calculate_plan_response_time(status):
    """计算响应时间（分钟）"""
    base_time = 15  # 默认15分钟

    # 根据拥堵状态调整响应时间
    if status == 4:
        base_time += 10  # 严重拥堵需要更长时间
    elif status == 3:
        base_time += 5  # 拥堵
    elif status == 2:
        base_time += 2  # 缓行
    elif status == 1:
        base_time -= 3  # 畅通

    # 确保时间在合理范围内
    return max(5, min(45, base_time))


def calculate_plan_scope(status):
    """计算影响范围"""
    scopes = ['极小', '小', '局部', '区域', '全市']
    scope_index = 2  # 默认局部

    if status == 4:
        scope_index += 1  # 严重拥堵影响范围大
    if status == 1:
        scope_index -= 1  # 畅通影响范围小

    scope_index = max(0, min(4, scope_index))
    return scopes[scope_index]


class GuidanceLog(db.Model):
    """疏导操作日志表"""
    __tablename__ = 'guidance_log'
    id = db.Column(db.Integer, primary_key=True)
    plan_id = db.Column(db.Integer, db.ForeignKey('guidance_plan.id'), nullable=True)
    action = db.Column(db.String(50), nullable=False)  # activate/revert/modify
    operator = db.Column(db.String(50), nullable=False)
    note = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)


class TrafficHistory(db.Model):
    """历史路况数据表"""
    __tablename__ = 'traffic_history'
    id = db.Column(db.Integer, primary_key=True)
    road_name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), nullable=False)
    speed = db.Column(db.Float, nullable=False)
    flow = db.Column(db.Integer, nullable=False)
    color = db.Column(db.String(20))
    collect_time = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)


# 自动建表
with app.app_context():
    db.create_all()


# ============================================================
# 工具函数
# ============================================================

def login_required(f):
    """登录保护装饰器"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return wrapper


def send_verify_email(to_email: str, code: str, purpose: str = 'register'):
    """使用 smtplib 直接发送验证码邮件（SSL 465）"""
    msg = MIMEMultipart('alternative')

    if purpose == 'reset_password':
        msg['Subject'] = '【城市交通系统】密码重置验证码'
        title = '密码重置验证码'
        action_text = '重置密码'
    else:
        msg['Subject'] = '【城市交通系统】注册验证码'
        title = '注册验证码'
        action_text = '注册'

    msg['From'] = MAIL_USERNAME
    msg['To'] = to_email

    html_body = f"""
    <div style="font-family:Microsoft YaHei,Arial;max-width:500px;margin:0 auto;
                border:1px solid #e0e0e0;border-radius:10px;overflow:hidden;">
        <div style="background:linear-gradient(135deg,#007bff,#00a8e8);padding:25px;text-align:center;">
            <h2 style="color:#fff;margin:0;">城市交通预测与疏导系统</h2>
        </div>
        <div style="padding:30px;">
            <p style="color:#333;font-size:15px;">您好！您正在进行{action_text}操作，验证码为：</p>
            <div style="background:#f0f7ff;border-radius:8px;padding:20px;text-align:center;margin:20px 0;">
                <span style="font-size:36px;font-weight:700;letter-spacing:10px;color:#007bff;">{code}</span>
            </div>
            <p style="color:#666;font-size:13px;">验证码 <strong>5 分钟内</strong>有效，请勿泄露给他人。</p>
            <p style="color:#999;font-size:12px;margin-top:20px;">如非本人操作，请忽略此邮件。</p>
        </div>
    </div>
    """
    msg.attach(MIMEText(html_body, 'html', 'utf-8'))

    with smtplib.SMTP_SSL(MAIL_SMTP_HOST, MAIL_SMTP_PORT) as server:
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.sendmail(MAIL_USERNAME, [to_email], msg.as_string())


# ============================================================
# 验证码生成
# ============================================================

def generate_captcha_image(text: str) -> bytes:
    """生成图形验证码图片，返回 PNG 字节流"""
    W, H = 120, 42
    # 背景色
    bg_color = (random.randint(230, 255), random.randint(230, 255), random.randint(230, 255))
    img = Image.new('RGB', (W, H), color=bg_color)
    draw = ImageDraw.Draw(img)

    # 随机干扰线
    for _ in range(5):
        x1, y1 = random.randint(0, W), random.randint(0, H)
        x2, y2 = random.randint(0, W), random.randint(0, H)
        draw.line([(x1, y1), (x2, y2)], fill=(
            random.randint(150, 220), random.randint(150, 220), random.randint(150, 220)), width=1)

    # 随机噪点
    for _ in range(60):
        x, y = random.randint(0, W - 1), random.randint(0, H - 1)
        draw.point((x, y), fill=(
            random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)))

    # 尝试加载系统字体，失败则用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 26)
        except Exception:
            font = ImageFont.load_default()

    # 逐字绘制（略微随机偏移和旋转）
    char_w = W // len(text)
    for i, ch in enumerate(text):
        color = (random.randint(20, 120), random.randint(20, 120), random.randint(20, 120))
        x = i * char_w + random.randint(2, 8)
        y = random.randint(4, 12)
        # 在临时图像上旋转字符
        char_img = Image.new('RGBA', (char_w, H), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((2, y - 4), ch, font=font, fill=color)
        char_img = char_img.rotate(random.randint(-18, 18), expand=False)
        img.paste(char_img, (x, 0), char_img)

    # 轻微模糊，增加识别难度
    img = img.filter(ImageFilter.SMOOTH)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


@app.route('/captcha')
def captcha():
    """生成验证码图片接口"""
    chars = string.ascii_uppercase + string.digits
    # 去掉易混淆字符
    chars = chars.replace('O', '').replace('0', '').replace('I', '').replace('1', '')
    code = ''.join(random.choices(chars, k=4))
    session['captcha'] = code.upper()

    img_bytes = generate_captcha_image(code)
    resp = make_response(img_bytes)
    resp.headers['Content-Type'] = 'image/png'
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp


# ============================================================
# 路由
# ============================================================

@app.route('/send_email_code', methods=['POST'])
def send_email_code():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'msg': '参数错误'})

        email = data.get('email', '').strip()
        purpose = data.get('purpose', 'register')  # 'register' 或 'reset_password'

        if not email or not re.match(r'^[\w.-]+@[\w.-]+\.\w+$', email):
            return jsonify({'success': False, 'msg': '邮箱格式不正确'})

        # 根据目的进行不同的检查
        if purpose == 'register':
            if User.query.filter_by(email=email).first():
                return jsonify({'success': False, 'msg': '该邮箱已被注册'})
        elif purpose == 'reset_password':
            if not User.query.filter_by(email=email).first():
                return jsonify({'success': False, 'msg': '该邮箱未注册'})
        else:
            return jsonify({'success': False, 'msg': '无效的目的参数'})

        now = datetime.datetime.now()
        recent = VerificationCode.query.filter_by(email=email, purpose=purpose, is_used=False).filter(
            VerificationCode.expires_at > now
        ).order_by(VerificationCode.created_at.desc()).first()

        if recent:
            interval = int((now - recent.created_at).total_seconds())
            if interval < 60:
                return jsonify({'success': False, 'msg': f'请 {60 - interval} 秒后再试'})

        code = str(random.randint(100000, 999999))
        expires_at = now + datetime.timedelta(minutes=5)

        vc = VerificationCode(email=email, code=code, expires_at=expires_at, purpose=purpose)
        db.session.add(vc)
        db.session.commit()

        send_verify_email(email, code, purpose)
        return jsonify({'success': True, 'msg': f'验证码已发送至 {email}'})

    except Exception as e:
        import traceback
        db.session.rollback()
        err_detail = traceback.format_exc()
        print("发送验证码错误:", err_detail)
        return jsonify({'success': False, 'msg': f'发送失败：{str(e)}'})


@app.route('/register', methods=['GET', 'POST'])
def register():
    """注册"""
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        email_code = request.form.get('email_code', '').strip()
        password = request.form.get('password', '')
        confirm_pwd = request.form.get('confirm_pwd', '')

        # —— 用户名校验 ——
        if not username:
            flash('用户名不能为空！', 'error')
            return render_template('register.html')
        if len(username) < 3 or len(username) > 20:
            flash('用户名长度须在 3~20 个字符之间！', 'error')
            return render_template('register.html', username=username, email=email)
        if not re.match(r'^[a-zA-Z0-9_\u4e00-\u9fa5]+$', username):
            flash('用户名只能包含字母、数字、下划线或中文！', 'error')
            return render_template('register.html', username=username, email=email)

        # —— 邮箱校验 ——
        if not email or not re.match(r'^[\w.-]+@[\w.-]+\.\w+$', email):
            flash('请输入正确的邮箱地址！', 'error')
            return render_template('register.html', username=username)
        if User.query.filter_by(email=email).first():
            flash('该邮箱已被注册，请更换！', 'error')
            return render_template('register.html', username=username)

        # —— 验证码校验 ——
        if not email_code:
            flash('请输入邮箱验证码！', 'error')
            return render_template('register.html', username=username, email=email)

        vc = VerificationCode.query.filter_by(
            email=email, code=email_code, is_used=False
        ).filter(
            VerificationCode.expires_at > datetime.datetime.now()
        ).order_by(VerificationCode.created_at.desc()).first()

        if not vc:
            flash('验证码错误或已过期，请重新发送！', 'error')
            return render_template('register.html', username=username, email=email)

        # —— 密码校验 ——
        if len(password) < 6:
            flash('密码长度至少 6 位！', 'error')
            return render_template('register.html', username=username, email=email)
        if password != confirm_pwd:
            flash('两次输入的密码不一致！', 'error')
            return render_template('register.html', username=username, email=email)

        # —— 用户名重复校验 ——
        if User.query.filter_by(username=username).first():
            flash('用户名已被注册，请更换！', 'error')
            return render_template('register.html', email=email)

        # —— 创建用户 ——
        vc.is_used = True  # 标记验证码已使用
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(
            username=username,
            password_hash=password_hash,
            email=email,
            is_email_verified=True
        )
        db.session.add(new_user)
        db.session.commit()

        flash('注册成功！请登录', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录（支持用户名或邮箱登录）"""
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username_or_email = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        captcha_input = request.form.get('captcha', '').strip().upper()
        remember_me = request.form.get('remember_me')

        # —— 验证码校验 ——
        captcha_correct = session.pop('captcha', '')
        if not captcha_input or captcha_input != captcha_correct:
            flash('验证码错误，请重新输入！', 'error')
            return render_template('login.html', username=username_or_email)

        if not username_or_email or not password:
            flash('用户名/邮箱和密码不能为空！', 'error')
            return render_template('login.html', username=username_or_email)

        # —— 智能识别用户名或邮箱 ——
        if '@' in username_or_email:
            # 按邮箱查找用户
            user = User.query.filter_by(email=username_or_email).first()
            if not user:
                flash('邮箱不存在！', 'error')
                return render_template('login.html', username=username_or_email)
        else:
            # 按用户名查找用户
            user = User.query.filter_by(username=username_or_email).first()
            if not user:
                flash('用户名不存在！', 'error')
                return render_template('login.html', username=username_or_email)

        if not check_password_hash(user.password_hash, password):
            flash('密码错误！', 'error')
            return render_template('login.html', username=username_or_email)

        session.permanent = bool(remember_me)
        session['user_id'] = user.id
        session['username'] = user.username
        session['role'] = user.role

        flash(f'欢迎回来，{user.username}！', 'success')
        return redirect(url_for('index'))

    return render_template('login.html')


# ============================================================
# 预测辅助函数
# ============================================================

def _hour_to_time_period(hour: int) -> str:
    if 6 <= hour <= 9:   return 'early_peak'
    if 17 <= hour <= 19: return 'late_peak'
    if 10 <= hour <= 16: return 'flat'
    return 'night'


def _month_to_season(month: int) -> str:
    if month in (3, 4, 5):    return 'spring'
    if month in (6, 7, 8):    return 'summer'
    if month in (9, 10, 11):  return 'autumn'
    return 'winter'


def _get_guidance(status: int, road: str) -> dict:
    STATUS_INFO = {
        1: {'label': '畅通', 'color': '#28a745', 'icon': 'fa-circle-check',
            'tip': '路况良好，可正常行驶，预计无延误。'},
        2: {'label': '缓行', 'color': '#ffc107', 'icon': 'fa-triangle-exclamation',
            'tip': f'【{road}】当前缓行，建议错峰出行或提前 15~30 分钟出发。'},
        3: {'label': '拥堵', 'color': '#fd7e14', 'icon': 'fa-circle-exclamation',
            'tip': f'【{road}】当前拥堵，建议选择周边替代路线，预计延误 20~40 分钟。'},
        4: {'label': '严重拥堵', 'color': '#dc3545', 'icon': 'fa-ban',
            'tip': f'【{road}】严重拥堵！强烈建议绕行，预计延误 40 分钟以上。'},
    }
    return STATUS_INFO.get(status, STATUS_INFO[1])


def _generate_guidance_actions(status: int, road: str, hour: int) -> list:
    """生成结构化疏导措施列表"""
    import json

    # 加载路网拓扑
    try:
        import json as _json
        with open('static/data/road_network.json', 'r', encoding='utf-8') as f:
            road_network = _json.load(f)
    except:
        road_network = {}

    road_info = road_network.get(road, {})
    adjacent_roads = road_info.get('adjacent', [])

    actions = []

    if status == 1:  # 畅通
        actions.append({
            'action_type': 'monitor',
            'target_road': road,
            'detail': '保持正常监控，无需特殊措施',
            'priority': 'low',
            'estimated_effect': '无'
        })

    elif status == 2:  # 缓行
        actions.append({
            'action_type': 'signal_adjust',
            'target_road': road,
            'detail': f'延长 {road} 信号灯绿灯时长 10 秒',
            'priority': 'medium',
            'estimated_effect': '减少延误 5-10 分钟'
        })
        actions.append({
            'action_type': 'info_publish',
            'target_road': road,
            'detail': f'发布 {road} 缓行提示，建议错峰出行',
            'priority': 'low',
            'estimated_effect': '引导 15% 车流错峰'
        })

    elif status == 3:  # 拥堵
        # 替代路线建议
        alt_routes = adjacent_roads[:2] if adjacent_roads else []
        alt_text = ' → '.join(alt_routes) if alt_routes else '周边平行道路'

        actions.append({
            'action_type': 'traffic_control',
            'target_road': road,
            'detail': f'限制 {road} 入口流量，减少 20% 进入车辆',
            'priority': 'high',
            'estimated_effect': '降低拥堵指数 0.5'
        })
        actions.append({
            'action_type': 'alternative_route',
            'target_road': road,
            'detail': f'启用替代路线：{alt_text}',
            'priority': 'high',
            'estimated_effect': '分流 30% 车流'
        })
        actions.append({
            'action_type': 'police_dispatch',
            'target_road': road,
            'detail': f'增派 2 名交警到 {road} 现场疏导',
            'priority': 'medium',
            'estimated_effect': '提升通行效率 15%'
        })

    elif status == 4:  # 严重拥堵
        actions.append({
            'action_type': 'forced_diversion',
            'target_road': road,
            'detail': f'强制 {road} 车辆绕行，关闭部分入口',
            'priority': 'critical',
            'estimated_effect': '减少 50% 进入车辆'
        })
        actions.append({
            'action_type': 'emergency_plan',
            'target_road': road,
            'detail': '启动应急预案，协调周边区域协同控制',
            'priority': 'critical',
            'estimated_effect': '防止拥堵扩散'
        })
        actions.append({
            'action_type': 'public_notice',
            'target_road': road,
            'detail': f'发布 {road} 严重拥堵预警，建议公共交通出行',
            'priority': 'high',
            'estimated_effect': '减少 25% 私家车出行'
        })

    # 根据时间调整措施
    if 6 <= hour <= 9:  # 早高峰
        for action in actions:
            if action['action_type'] in ['signal_adjust', 'traffic_control']:
                action['detail'] += '（早高峰加强版）'
                action['estimated_effect'] = '效果提升 20%'

    return actions


def _log_guidance_action(plan_id: int, action: str, operator: str, note: str = None):
    """记录疏导操作日志"""
    log = GuidanceLog(
        plan_id=plan_id,
        action=action,
        operator=operator,
        note=note
    )
    db.session.add(log)
    db.session.commit()


def get_available_dates_from_db():
    """从数据库获取可用的日期列表"""
    try:
        # 查询所有不同的日期
        from sqlalchemy import func
        dates = db.session.query(
            func.date(TrafficHistory.collect_time).label('date')
        ).distinct().order_by('date').all()

        # 转换为字符串列表
        available_dates = [date[0].strftime('%Y-%m-%d') for date in dates if date[0]]
        return available_dates
    except Exception as e:
        print(f"[get_available_dates_from_db] 获取日期列表失败: {e}")
        return []


@app.route('/predict')
@login_required
def predict():
    road_list = _ROAD_LIST or []
    options = _OPTIONS or {}
    return render_template('predict.html',
                           road_list=road_list,
                           options=options)


@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    if _MODEL is None:
        return jsonify({'success': False, 'msg': '模型未加载，请先运行 train_model.py'})
    try:
        data = request.get_json()

        road = data.get('road', '')
        hour = int(data.get('hour', datetime.datetime.now().hour))
        day_of_week = int(data.get('day_of_week', datetime.datetime.now().weekday()))
        weather = data.get('weather', '晴')
        temperature = float(data.get('temperature', 20))
        humidity = int(data.get('humidity', 70))
        season = data.get('season', _month_to_season(datetime.datetime.now().month))
        time_period = data.get('time_period', _hour_to_time_period(hour))
        is_peak_hour = 1 if time_period in ('early_peak', 'late_peak') else 0

        # Encode categorical for congestion model
        enc = _ENCODERS

        def safe_encode(le, val):
            classes = list(le.classes_)
            if val in classes:
                return le.transform([val])[0]
            return 0  # fallback

        road_enc = safe_encode(enc['road_simple'], road)
        season_enc = safe_encode(enc['season'], season)
        time_period_enc = safe_encode(enc['time_period'], time_period)
        weather_enc = safe_encode(enc['weather'], weather)

        X = np.array([[road_enc, hour, day_of_week,
                       season_enc, time_period_enc, weather_enc,
                       is_peak_hour, temperature, humidity]])

        # Predict congestion status
        pred_status = int(_MODEL.predict(X)[0])
        proba = _MODEL.predict_proba(X)[0]
        confidence = round(float(max(proba)) * 100, 1)

        # Estimate speed from historical agg data (simple rule)
        speed_map = {1: random.uniform(45, 60), 2: random.uniform(25, 44),
                     3: random.uniform(12, 24), 4: random.uniform(5, 11)}
        est_speed = round(speed_map[pred_status], 1)

        guidance = _get_guidance(pred_status, road)

        # Predict traffic flow if flow model is available
        predicted_flow = None
        flow_confidence = None
        if _FLOW_MODEL is not None:
            try:
                # Encode categorical for flow model
                flow_enc = _FLOW_ENCODERS
                flow_road_enc = safe_encode(flow_enc['road_simple'], road)
                flow_season_enc = safe_encode(flow_enc['season'], season)
                flow_time_period_enc = safe_encode(flow_enc['time_period'], time_period)
                flow_weather_enc = safe_encode(flow_enc['weather'], weather)

                # Prepare features for flow prediction
                flow_features = []
                for feat in _FLOW_FEATURES:
                    if feat == 'road_simple_enc':
                        flow_features.append(flow_road_enc)
                    elif feat == 'hour':
                        flow_features.append(hour)
                    elif feat == 'day_of_week':
                        flow_features.append(day_of_week)
                    elif feat == 'season_enc':
                        flow_features.append(flow_season_enc)
                    elif feat == 'time_period_enc':
                        flow_features.append(flow_time_period_enc)
                    elif feat == 'weather_enc':
                        flow_features.append(flow_weather_enc)
                    elif feat == 'is_peak_hour':
                        flow_features.append(is_peak_hour)
                    elif feat == 'temperature':
                        flow_features.append(temperature)
                    elif feat == 'humidity':
                        flow_features.append(humidity)
                    # Note: congestion_status is not in the original features list

                X_flow = np.array([flow_features])

                # Predict flow (no scaling needed as features are already in correct range)
                predicted_flow_raw = _FLOW_MODEL.predict(X_flow)[0]
                predicted_flow = int(max(0, predicted_flow_raw))  # Ensure non-negative

                # Calculate flow level and color
                if predicted_flow < 1000:
                    flow_level = "低"
                    flow_color = "#28a745"  # 绿色
                elif predicted_flow < 3000:
                    flow_level = "中"
                    flow_color = "#ffc107"  # 黄色
                else:
                    flow_level = "高"
                    flow_color = "#dc3545"  # 红色

                # Format flow display
                flow_formatted = f"{predicted_flow:,} 辆/小时"

                # Calculate flow percentage (relative to max 8000)
                flow_percentage = min(100, int((predicted_flow / 8000) * 100))

                flow_confidence = round(random.uniform(75, 95), 1)  # Simulated confidence

            except Exception as flow_e:
                print(f"[Flow Prediction] Error: {flow_e}")
                predicted_flow = None

        result = {
            'success': True,
            'status': pred_status,
            'label': guidance['label'],
            'color': guidance['color'],
            'icon': guidance['icon'],
            'tip': guidance['tip'],
            'est_speed': est_speed,
            'confidence': confidence,
        }

        # Add flow prediction if available
        if predicted_flow is not None:
            result['predicted_flow'] = predicted_flow
            result['flow_formatted'] = flow_formatted
            result['flow_level'] = flow_level
            result['flow_color'] = flow_color
            result['flow_percentage'] = flow_percentage
            result['flow_confidence'] = flow_confidence

        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'msg': str(e), 'trace': traceback.format_exc()})


@app.route('/api/roads')
@login_required
def api_roads():
    return jsonify({'roads': _ROAD_LIST or []})


@app.route('/api/traffic_flow_history')
@login_required
def api_traffic_flow_history():
    """根据精确时间筛选历史车流量数据 - 从数据库读取，支持完整时间点筛选"""
    try:
        date_str = request.args.get('date', '')  # 格式: 2025-03-01
        hour = request.args.get('hour', '')  # 0-23, 空表示全天
        minute = request.args.get('minute', '')  # 0-59, 空表示整点
        second = request.args.get('second', '')  # 0-59, 空表示整分
        exact_time_str = request.args.get('exact_time', '')  # 新增：完整时间点，格式: 2026-03-16 14:19:20

        # 构建查询
        query = TrafficHistory.query

        # 优先使用完整时间点筛选
        if exact_time_str:
            try:
                # 解析完整时间点
                exact_time = datetime.datetime.strptime(exact_time_str, '%Y-%m-%d %H:%M:%S')

                # 查询该精确时间点的数据（前后30秒范围内）
                start_time = exact_time - datetime.timedelta(seconds=30)
                end_time = exact_time + datetime.timedelta(seconds=30)
                query = query.filter(
                    TrafficHistory.collect_time >= start_time,
                    TrafficHistory.collect_time <= end_time
                )

                # 设置其他参数为从完整时间点解析的值
                date_str = exact_time.strftime('%Y-%m-%d')
                hour = str(exact_time.hour)
                minute = str(exact_time.minute)
                second = str(exact_time.second)

            except ValueError:
                return jsonify({'success': False, 'msg': '时间格式错误，请使用YYYY-MM-DD HH:MM:SS格式'})
        elif date_str:
            # 将日期字符串转换为日期范围
            try:
                date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

                # 如果指定了小时、分钟、秒，则构建精确时间
                if hour != '' and hour is not None:
                    try:
                        hour_int = int(hour)
                        minute_int = int(minute) if minute != '' and minute is not None else 0
                        second_int = int(second) if second != '' and second is not None else 0

                        # 构建精确时间点
                        exact_time = datetime.datetime.combine(
                            date_obj,
                            datetime.time(hour_int, minute_int, second_int)
                        )

                        # 查询该精确时间点的数据（前后30秒范围内）
                        start_time = exact_time - datetime.timedelta(seconds=30)
                        end_time = exact_time + datetime.timedelta(seconds=30)
                        query = query.filter(
                            TrafficHistory.collect_time >= start_time,
                            TrafficHistory.collect_time <= end_time
                        )
                    except ValueError:
                        # 如果时间参数无效，回退到按小时筛选
                        hour_int = int(hour)
                        query = query.filter(db.extract('hour', TrafficHistory.collect_time) == hour_int)
                else:
                    # 只按日期筛选（全天）
                    start_datetime = datetime.datetime.combine(date_obj, datetime.time.min)
                    end_datetime = datetime.datetime.combine(date_obj, datetime.time.max)
                    query = query.filter(
                        TrafficHistory.collect_time >= start_datetime,
                        TrafficHistory.collect_time <= end_datetime
                    )
            except ValueError:
                return jsonify({'success': False, 'msg': '日期格式错误，请使用YYYY-MM-DD格式'})
        elif hour != '' and hour is not None:
            # 只按小时筛选（不指定日期）
            try:
                hour_int = int(hour)
                query = query.filter(db.extract('hour', TrafficHistory.collect_time) == hour_int)
            except ValueError:
                pass

        # 执行查询
        records = query.all()

        # 如果没有数据
        if not records:
            # 获取可用的日期列表
            available_dates = get_available_dates_from_db()
            return jsonify({
                'success': True,
                'data': [],
                'available_dates': available_dates,
                'selected_date': date_str,
                'selected_hour': hour,
                'selected_minute': minute,
                'selected_second': second,
                'selected_exact_time': exact_time_str,
                'message': '该时间段没有数据'
            })

        # 按道路聚合计算平均流量
        road_data = {}
        for record in records:
            road_name = record.road_name
            if road_name not in road_data:
                road_data[road_name] = {
                    'flows': [],
                    'speeds': [],
                    'status_values': [],
                    'times': []  # 记录时间，用于显示精确时间
                }

            road_data[road_name]['flows'].append(record.flow)
            road_data[road_name]['speeds'].append(record.speed)
            # 将状态转换为数值
            status_map = {"畅通": 1, "缓行": 2, "拥堵": 3, "严重拥堵": 4}
            road_data[road_name]['status_values'].append(status_map.get(record.status, 2))
            road_data[road_name]['times'].append(record.collect_time)

        traffic_flow_data = []
        for road_name, data in road_data.items():
            # 计算平均值
            avg_flow = int(sum(data['flows']) / len(data['flows']))
            avg_speed = sum(data['speeds']) / len(data['speeds'])
            avg_status = sum(data['status_values']) / len(data['status_values'])

            # 获取最接近的时间
            if data['times']:
                # 找到最接近筛选时间的数据
                if exact_time_str:
                    try:
                        target_time = datetime.datetime.strptime(exact_time_str, '%Y-%m-%d %H:%M:%S')
                        closest_time = min(data['times'], key=lambda x: abs(x - target_time))
                        time_str = closest_time.strftime("%H:%M:%S")
                    except:
                        time_str = data['times'][0].strftime("%H:%M:%S")
                elif hour != '' and hour is not None and minute != '' and minute is not None:
                    try:
                        target_time = datetime.datetime.combine(
                            datetime.datetime.strptime(date_str, '%Y-%m-%d').date(),
                            datetime.time(int(hour), int(minute),
                                          int(second) if second != '' and second is not None else 0)
                        )
                        closest_time = min(data['times'], key=lambda x: abs(x - target_time))
                        time_str = closest_time.strftime("%H:%M:%S")
                    except:
                        time_str = data['times'][0].strftime("%H:%M:%S")
                else:
                    time_str = data['times'][0].strftime("%H:%M:%S")
            else:
                time_str = ""

            # 计算流量等级和颜色
            if avg_flow < 1000:
                flow_level = "低"
                flow_color = "#28a745"
            elif avg_flow < 3000:
                flow_level = "中"
                flow_color = "#ffc107"
            else:
                flow_level = "高"
                flow_color = "#dc3545"

            flow_formatted = f"{avg_flow:,} 辆/小时"
            flow_percentage = min(100, int((avg_flow / 8000) * 100))

            # 将数值状态转换回文本
            status_text_map = {1: "畅通", 2: "缓行", 3: "拥堵", 4: "严重拥堵"}
            status_text = status_text_map.get(round(avg_status), "缓行")

            traffic_flow_data.append({
                "road_name": road_name,
                "flow": avg_flow,
                "flow_formatted": flow_formatted,
                "flow_level": flow_level,
                "flow_color": flow_color,
                "flow_percentage": flow_percentage,
                "avg_speed": round(avg_speed, 1),
                "congestion_status": round(avg_status, 1),
                "status": status_text,
                "time": time_str  # 添加精确时间显示
            })

        # 按流量降序排序
        traffic_flow_data.sort(key=lambda x: x['flow'], reverse=True)

        # 获取可用的日期列表
        available_dates = get_available_dates_from_db()

        return jsonify({
            'success': True,
            'data': traffic_flow_data,
            'available_dates': available_dates,
            'selected_date': date_str,
            'selected_hour': hour,
            'selected_minute': minute,
            'selected_second': second,
            'selected_exact_time': exact_time_str,
            'record_count': len(records),
            'road_count': len(traffic_flow_data),
            'time_precision': 'exact' if exact_time_str or (
                        hour != '' and minute != '' and second != '') else 'hour' if hour != '' else 'day'
        })

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'msg': str(e),
            'trace': traceback.format_exc()
        })


@app.route('/api/traffic_available_times')
@login_required
def api_traffic_available_times():
    """获取历史数据中可用的完整时间点 - 从数据库读取"""
    try:
        from sqlalchemy import func

        # 获取所有可用的完整时间点（过滤NULL值）
        time_points = db.session.query(
            TrafficHistory.collect_time
        ).filter(TrafficHistory.collect_time.isnot(None)).distinct().order_by(TrafficHistory.collect_time).all()

        # 格式化为字符串列表，处理不同类型的时间对象
        available_times = []
        for tp in time_points:
            if tp[0]:
                try:
                    # 处理不同类型的日期时间对象
                    if hasattr(tp[0], 'strftime'):
                        # datetime对象
                        formatted = tp[0].strftime('%Y-%m-%d %H:%M:%S')
                    elif isinstance(tp[0], str):
                        # 字符串对象，尝试解析
                        import datetime
                        # 尝试多种格式
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d']:
                            try:
                                dt = datetime.datetime.strptime(tp[0], fmt)
                                formatted = dt.strftime('%Y-%m-%d %H:%M:%S')
                                break
                            except:
                                continue
                        else:
                            # 如果所有格式都失败，跳过
                            continue
                    else:
                        # 其他类型，尝试转换为字符串
                        formatted = str(tp[0])
                        # 尝试解析为datetime
                        try:
                            import datetime
                            dt = datetime.datetime.fromisoformat(formatted.replace('Z', '+00:00'))
                            formatted = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            # 如果转换失败，使用原始字符串
                            pass

                    available_times.append(formatted)
                except Exception as e:
                    print(f"[api_traffic_available_times] 格式化时间点错误: {e}, 时间: {tp[0]}")
                    continue

        # 去重并排序
        available_times = sorted(list(set(available_times)))

        # 按日期分组，用于兼容旧接口
        available_dates = sorted(list(set([t.split(' ')[0] for t in available_times if ' ' in t])))

        # 获取每个日期可用的小时（用于兼容旧接口）
        date_hours = {}
        for date_str in available_dates:
            # 过滤出该日期的时间点
            date_times = [t for t in available_times if t.startswith(date_str)]
            # 提取小时
            hours = []
            for t in date_times:
                try:
                    hour = int(t.split(' ')[1].split(':')[0])
                    hours.append(hour)
                except:
                    continue
            hours = sorted(list(set(hours)))
            date_hours[date_str] = hours

        # 获取最新日期和其可用小时（用于默认值）
        latest_date = available_dates[-1] if available_dates else ''
        latest_hours = date_hours.get(latest_date, [])

        return jsonify({
            'success': True,
            'available_times': available_times,  # 新增：完整时间点列表
            'available_dates': available_dates,
            'date_hours': date_hours,
            'latest_date': latest_date,
            'latest_hours': latest_hours,
            'date_count': len(available_dates),
            'time_count': len(available_times)
        })

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'msg': str(e),
            'trace': traceback.format_exc()
        })


@app.route('/api/trend')
@login_required
def api_trend():
    """返回某条路 24 小时历史数据（流量指数 + 拥堵等级 + 车速，用于图表）- 带内存缓存"""
    road = request.args.get('road', '')
    if not road:
        return jsonify({'success': False, 'msg': '未指定道路'})

    # 命中缓存直接返回
    if road in _TREND_CACHE:
        return jsonify(_TREND_CACHE[road])

    try:
        df = pd.read_csv('static/data/final_traffic_data.csv')
        # 使用原始道路名称，不进行合并
        df['road_simple'] = df['road_name'].astype(str)
        df['collect_time'] = pd.to_datetime(df['collect_time'])
        df['hour'] = df['collect_time'].dt.hour
        hours = list(range(24))

        # 获取所有道路名（保持一致顺序）
        all_roads = sorted(df['road_simple'].unique().tolist())

        # 预计算热力图数据（所有道路 × 24小时）
        heatmap_data = []
        for road_idx, r in enumerate(all_roads):
            sub = df[df['road_simple'] == r]
            cong_grp = sub.groupby('hour')['congestion_status'].mean().round(2)
            for h in hours:
                val = round(float(cong_grp.get(h, 1)), 2)
                heatmap_data.append([h, road_idx, val])

        # 预计算所有道路的趋势并缓存
        for r in df['road_simple'].unique():
            sub = df[df['road_simple'] == r]
            flow_grp = sub.groupby('hour')['flow_index'].mean().round(3)
            cong_grp = sub.groupby('hour')['congestion_status'].mean().round(2)
            speed_grp = sub.groupby('hour')['avg_speed'].mean().round(1)
            _TREND_CACHE[r] = {
                'success': True,
                'hours': hours,
                'flows': [round(float(flow_grp.get(h, 0)), 3) for h in hours],
                'congestions': [round(float(cong_grp.get(h, 1)), 2) for h in hours],
                'speeds': [round(float(speed_grp.get(h, 0)), 1) for h in hours],
                'heatmap_data': heatmap_data,
                'road_names': all_roads,
            }

        if road in _TREND_CACHE:
            return jsonify(_TREND_CACHE[road])
        return jsonify({'success': False, 'msg': f'未找到道路: {road}'})
    except Exception as e:
        return jsonify({'success': False, 'msg': str(e)})


@app.route('/api/weather')
@login_required
def api_weather():
    """调用高德天气 API，根据预测时间返回对应天气数据"""
    import urllib.request
    import json as _json

    dt_str = request.args.get('dt', '')
    road = request.args.get('road', '')  # 可选：路段名（用于精准天气）
    if not dt_str:
        return jsonify({'success': False, 'msg': '未指定时间'})

    try:
        target_dt = datetime.datetime.fromisoformat(dt_str)
    except Exception:
        return jsonify({'success': False, 'msg': '时间格式错误'})

    now = datetime.datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    target_date = target_dt.strftime('%Y-%m-%d')
    target_hour = target_dt.hour

    # 优先使用路段级 adcode，回退到武汉市
    city_code = _ROAD_ADCODE.get(road, AMAP_CITY_CODE) if road else AMAP_CITY_CODE

    try:
        # ── 1. 获取实况天气（含湿度，仅今天有效）
        base_url = (
            f'https://restapi.amap.com/v3/weather/weatherInfo'
            f'?key={AMAP_API_KEY}&city={city_code}'
            f'&extensions=base&output=JSON'
        )
        with urllib.request.urlopen(base_url, timeout=5) as resp:
            live_data = _json.loads(resp.read().decode('utf-8'))

        live_weather = '晴'
        live_temp = 20.0
        live_humidity = 70

        if live_data.get('status') == '1' and live_data.get('lives'):
            live = live_data['lives'][0]
            live_weather = _map_amap_weather(live.get('weather', '晴'))
            live_temp = float(live.get('temperature_float', live.get('temperature', 20)))
            live_humidity = int(float(live.get('humidity_float', live.get('humidity', 70))))

        # ── 2. 获取预报天气（未来3天）
        fore_url = (
            f'https://restapi.amap.com/v3/weather/weatherInfo'
            f'?key={AMAP_API_KEY}&city={AMAP_CITY_CODE}'
            f'&extensions=all&output=JSON'
        )
        with urllib.request.urlopen(fore_url, timeout=5) as resp:
            fore_data = _json.loads(resp.read().decode('utf-8'))

        # ── 3. 根据目标日期选择数据
        if target_date == today_str:
            # 今天：直接使用实况天气
            return jsonify({
                'success': True,
                'weather': live_weather,
                'temperature': round(live_temp, 1),
                'humidity': live_humidity,
                'source': '实况',
            })

        # 今天以外（明天）：从预报中匹配
        weather_out = live_weather  # 兜底
        temp_out = live_temp
        humidity_out = live_humidity

        if fore_data.get('status') == '1' and fore_data.get('forecasts'):
            casts = fore_data['forecasts'][0].get('casts', [])
            for cast in casts:
                if cast.get('date') == target_date:
                    # 白天 6-18 时用白天天气，否则用夜间天气
                    if 6 <= target_hour < 18:
                        w_desc = cast.get('dayweather', '晴')
                        t_val = float(cast.get('daytemp_float', cast.get('daytemp', 20)))
                    else:
                        w_desc = cast.get('nightweather', '晴')
                        t_val = float(cast.get('nighttemp_float', cast.get('nighttemp', 15)))
                    weather_out = _map_amap_weather(w_desc)
                    temp_out = t_val
                    # 预报无湿度，使用实况湿度估算
                    humidity_out = live_humidity
                    break

        return jsonify({
            'success': True,
            'weather': weather_out,
            'temperature': round(temp_out, 1),
            'humidity': humidity_out,
            'source': '预报',
        })

    except Exception as e:
        return jsonify({'success': False, 'msg': f'天气获取失败：{str(e)}'})


@app.route('/logout')
def logout():
    session.clear()
    flash('已成功退出登录！', 'success')
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    # 获取最近一次的历史数据（每个道路的最新记录）
    try:
        from sqlalchemy import func

        # 获取每个道路的最新采集时间
        latest_records = db.session.query(
            TrafficHistory.road_name,
            func.max(TrafficHistory.collect_time).label('latest_time')
        ).group_by(TrafficHistory.road_name).subquery()

        # 查询这些最新记录
        history_data = TrafficHistory.query.join(
            latest_records,
            (TrafficHistory.road_name == latest_records.c.road_name) &
            (TrafficHistory.collect_time == latest_records.c.latest_time)
        ).order_by(TrafficHistory.road_name).all()

        # 转换为前端需要的格式 - 车流量可视化
        traffic_flow_data = []
        for record in history_data:
            # 计算流量等级和颜色
            flow = record.flow
            if flow < 1000:
                flow_level = "低"
                flow_color = "#28a745"  # 绿色
            elif flow < 3000:
                flow_level = "中"
                flow_color = "#ffc107"  # 黄色
            else:
                flow_level = "高"
                flow_color = "#dc3545"  # 红色

            # 格式化流量显示
            flow_formatted = f"{flow:,} 辆/小时"

            # 计算流量百分比（相对于最大流量8000）
            flow_percentage = min(100, int((flow / 8000) * 100))

            traffic_flow_data.append({
                "road_name": record.road_name,
                "flow": flow,
                "flow_formatted": flow_formatted,
                "flow_level": flow_level,
                "flow_color": flow_color,
                "flow_percentage": flow_percentage
            })

        # 如果没有历史数据，使用示例数据
        if not traffic_flow_data:
            traffic_flow_data = [
                {"road_name": "楚河汉街", "flow": 2500, "flow_formatted": "2,500 辆/小时", "flow_level": "中",
                 "flow_color": "#ffc107", "flow_percentage": 31},
                {"road_name": "江汉路步行街", "flow": 3500, "flow_formatted": "3,500 辆/小时", "flow_level": "高",
                 "flow_color": "#dc3545", "flow_percentage": 44},
                {"road_name": "武广商圈", "flow": 1800, "flow_formatted": "1,800 辆/小时", "flow_level": "中",
                 "flow_color": "#ffc107", "flow_percentage": 23},
                {"road_name": "解放大道（同济段）", "flow": 4200, "flow_formatted": "4,200 辆/小时", "flow_level": "高",
                 "flow_color": "#dc3545", "flow_percentage": 53}
            ]

        # 获取数据采集时间（使用最新记录的时间）
        if history_data:
            latest_time = max(record.collect_time for record in history_data)
            current_time = latest_time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        print(f"[Index] 获取历史数据失败: {e}")
        # 出错时使用示例数据
        traffic_flow_data = [
            {"road_name": "楚河汉街", "flow": 2500, "flow_formatted": "2,500 辆/小时", "flow_level": "中",
             "flow_color": "#ffc107", "flow_percentage": 31},
            {"road_name": "江汉路步行街", "flow": 3500, "flow_formatted": "3,500 辆/小时", "flow_level": "高",
             "flow_color": "#dc3545", "flow_percentage": 44},
            {"road_name": "武广商圈", "flow": 1800, "flow_formatted": "1,800 辆/小时", "flow_level": "中",
             "flow_color": "#ffc107", "flow_percentage": 23},
            {"road_name": "解放大道（同济段）", "flow": 4200, "flow_formatted": "4,200 辆/小时", "flow_level": "高",
             "flow_color": "#dc3545", "flow_percentage": 53}
        ]
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return render_template('index.html',
                           username=session['username'],
                           role=session['role'],
                           traffic_flow_data=traffic_flow_data,
                           current_time=current_time)


# ============================================================
# 疏导模块 API
# ============================================================

# 导入机器学习疏导模型
try:
    from guidance_ml_integration import GuidanceMLModel

    guidance_ml_model = GuidanceMLModel()
    if not guidance_ml_model.load_model():
        print('[GuidanceML] 机器学习模型不存在，将在首次使用时训练')
except Exception as e:
    print(f'[GuidanceML] 机器学习模型加载失败: {e}')
    guidance_ml_model = None


@app.route('/guidance')
@login_required
def guidance():
    """疏导控制台页面"""
    road_list = _ROAD_LIST or []
    return render_template('guidance.html', road_list=road_list)


@app.route('/guidance_new')
@login_required
def guidance_new():
    """新版疏导控制台页面"""
    road_list = _ROAD_LIST or []
    return render_template('guidance_new.html', road_list=road_list)


@app.route('/api/guidance/ml_predict', methods=['POST'])
@login_required
def api_guidance_ml_predict():
    """机器学习疏导建议预测"""
    try:
        if guidance_ml_model is None:
            return jsonify({'success': False, 'msg': '机器学习模型未加载'})

        data = request.get_json()

        # 提取特征
        hour = int(data.get('hour', datetime.datetime.now().hour))
        day_of_week = int(data.get('day_of_week', datetime.datetime.now().weekday()))

        input_features = {
            'road_type': data.get('road_type', '主干道'),
            'congestion_status': int(data.get('congestion_status', 2)),
            'hour': hour,
            'day_of_week': day_of_week,
            'season': data.get('season', _month_to_season(datetime.datetime.now().month)),
            'weather': data.get('weather', '晴'),
            'temperature': float(data.get('temperature', 20.0)),
            'humidity': int(data.get('humidity', 70)),
            'time_period': data.get('time_period', _hour_to_time_period(hour)),
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


@app.route('/api/guidance/generate', methods=['POST'])
@login_required
def api_guidance_generate():
    """生成疏导建议"""
    try:
        data = request.get_json()
        road = data.get('road', '')
        status = int(data.get('status', 1))
        hour = int(data.get('hour', datetime.datetime.now().hour))

        if not road:
            return jsonify({'success': False, 'msg': '请选择道路'})

        # 生成结构化措施
        actions = _generate_guidance_actions(status, road, hour)

        # 创建疏导方案记录
        import json
        plan = GuidancePlan(
            road=road,
            status=status,
            plan_type='auto',
            actions=json.dumps(actions, ensure_ascii=False),
            operator=session.get('username', 'system')
        )
        db.session.add(plan)
        db.session.commit()

        # 记录日志
        _log_guidance_action(plan.id, 'generate', session.get('username', 'system'),
                             f'自动生成疏导方案，状态{status}')

        return jsonify({
            'success': True,
            'plan_id': plan.id,
            'road': road,
            'status': status,
            'actions': actions,
            'generated_at': plan.created_at.isoformat()
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'msg': str(e), 'trace': traceback.format_exc()})


@app.route('/api/guidance/region', methods=['GET'])
@login_required
def api_guidance_region():
    """区域协同控制分析 - 增强版"""
    road = request.args.get('road', '')
    if not road:
        return jsonify({'success': False, 'msg': '请选择道路'})

    try:
        import json as _json
        with open('static/data/road_network.json', 'r', encoding='utf-8') as f:
            road_network = _json.load(f)
    except:
        return jsonify({'success': False, 'msg': '路网数据加载失败'})

    road_info = road_network.get(road, {})
    adjacent_roads = road_info.get('adjacent', [])

    # 分析相邻路段容量
    region_analysis = []
    for adj_road in adjacent_roads:
        adj_info = road_network.get(adj_road, {})
        capacity = adj_info.get('capacity', 2000)
        road_type = adj_info.get('type', '未知')

        # 用最近1小时真实数据估算负载
        _status_to_int = {"畅通": 1, "缓行": 2, "拥堵": 3, "严重拥堵": 4}
        _since = datetime.datetime.now() - datetime.timedelta(hours=1)
        _recent = TrafficHistory.query.filter(
            TrafficHistory.road_name == adj_road,
            TrafficHistory.collect_time >= _since
        ).all()
        if _recent:
            _avg = sum(_status_to_int.get(r.status, 2) for r in _recent) / len(_recent)
            current_load = int((_avg - 1) / 3 * 100)
            current_load = max(10, min(98, current_load))
        else:
            current_load = random.randint(60, 95)  # 无数据时保底
        remaining_capacity = capacity * (100 - current_load) / 100

        region_analysis.append({
            'road': adj_road,
            'type': road_type,
            'capacity': capacity,
            'current_load_percent': current_load,
            'remaining_capacity': int(remaining_capacity),
            'can_accept_diversion': remaining_capacity > 200  # 能接受至少200辆车/小时
        })

    # 计算协同策略
    total_remaining = sum(item['remaining_capacity'] for item in region_analysis)
    diversion_strategy = []

    if total_remaining > 0:
        for item in region_analysis:
            if item['can_accept_diversion']:
                share_percent = item['remaining_capacity'] / total_remaining
                diversion_strategy.append({
                    'road': item['road'],
                    'suggested_diversion': int(500 * share_percent),  # 假设需要分流500辆车
                    'percent': round(share_percent * 100, 1)
                })

    # 计算区域整体指标
    total_capacity = sum(item['capacity'] for item in region_analysis)
    avg_load = sum(item['current_load_percent'] for item in region_analysis) / len(
        region_analysis) if region_analysis else 0
    region_efficiency = 100 - avg_load  # 区域效率 = 100 - 平均负载

    return jsonify({
        'success': True,
        'target_road': road,
        'adjacent_roads': adjacent_roads,
        'region_analysis': region_analysis,
        'diversion_strategy': diversion_strategy,
        'region_metrics': {
            'total_capacity': total_capacity,
            'avg_load_percent': round(avg_load, 1),
            'region_efficiency': round(region_efficiency, 1),
            'total_remaining_capacity': int(total_remaining),
            'can_accept_diversion': total_remaining > 0
        },
        'recommendation': '建议按容量比例分配疏导流量' if diversion_strategy else '周边路段容量不足，需启动应急预案'
    })


@app.route('/api/region/heatmap', methods=['GET'])
@login_required
def api_region_heatmap():
    """区域热力图数据"""
    road = request.args.get('road', '')
    if not road:
        return jsonify({'success': False, 'msg': '请选择道路'})

    try:
        import json as _json
        with open('static/data/road_network.json', 'r', encoding='utf-8') as f:
            road_network = _json.load(f)
    except:
        return jsonify({'success': False, 'msg': '路网数据加载失败'})

    road_info = road_network.get(road, {})
    adjacent_roads = road_info.get('adjacent', [])

    # 获取所有相关道路（目标道路+相邻道路）
    all_roads = [road] + adjacent_roads

    # 获取最近1小时的数据
    _status_to_int = {"畅通": 1, "缓行": 2, "拥堵": 3, "严重拥堵": 4}
    _since = datetime.datetime.now() - datetime.timedelta(hours=1)

    heatmap_data = []
    for r in all_roads:
        # 获取道路信息
        r_info = road_network.get(r, {})
        capacity = r_info.get('capacity', 2000)

        # 获取最近数据
        _recent = TrafficHistory.query.filter(
            TrafficHistory.road_name == r,
            TrafficHistory.collect_time >= _since
        ).all()

        if _recent:
            # 计算平均状态
            _avg = sum(_status_to_int.get(rec.status, 2) for rec in _recent) / len(_recent)
            current_load = int((_avg - 1) / 3 * 100)
            current_load = max(10, min(98, current_load))

            # 计算流量
            avg_flow = sum(rec.flow for rec in _recent) / len(_recent) if _recent else 0
            flow_percentage = min(100, int((avg_flow / capacity) * 100))

            # 计算速度
            avg_speed = sum(rec.speed for rec in _recent) / len(_recent) if _recent else 30

        else:
            # 无数据时使用模拟数据
            current_load = random.randint(60, 95)
            flow_percentage = random.randint(50, 90)
            avg_speed = random.uniform(20, 50)

        # 确定颜色等级（基于拥堵程度）
        if current_load < 40:
            color_level = 1  # 绿色
        elif current_load < 70:
            color_level = 2  # 黄色
        elif current_load < 90:
            color_level = 3  # 橙色
        else:
            color_level = 4  # 红色

        heatmap_data.append({
            'road': r,
            'load_percent': current_load,
            'flow_percent': flow_percentage,
            'avg_speed': round(avg_speed, 1),
            'color_level': color_level,
            'is_target': r == road,
            'capacity': capacity
        })

    return jsonify({
        'success': True,
        'target_road': road,
        'heatmap_data': heatmap_data,
        'timestamp': datetime.datetime.now().isoformat()
    })


@app.route('/api/region/topology', methods=['GET'])
@login_required
def api_region_topology():
    """路网拓扑数据"""
    road = request.args.get('road', '')
    if not road:
        return jsonify({'success': False, 'msg': '请选择道路'})

    try:
        import json as _json
        with open('static/data/road_network.json', 'r', encoding='utf-8') as f:
            road_network = _json.load(f)
    except:
        return jsonify({'success': False, 'msg': '路网数据加载失败'})

    # 获取目标道路及其相邻道路
    target_info = road_network.get(road, {})
    adjacent_roads = target_info.get('adjacent', [])

    # 构建拓扑节点
    nodes = []
    links = []

    # 添加目标节点
    nodes.append({
        'id': road,
        'name': road,
        'type': target_info.get('type', '未知'),
        'capacity': target_info.get('capacity', 2000),
        'is_target': True,
        'symbolSize': 50
    })

    # 添加相邻节点
    for i, adj_road in enumerate(adjacent_roads):
        adj_info = road_network.get(adj_road, {})
        nodes.append({
            'id': adj_road,
            'name': adj_road,
            'type': adj_info.get('type', '未知'),
            'capacity': adj_info.get('capacity', 2000),
            'is_target': False,
            'symbolSize': 40
        })

        # 添加连接
        links.append({
            'source': road,
            'target': adj_road,
            'value': 1,
            'lineStyle': {
                'width': 3,
                'curveness': 0.2
            }
        })

    # 添加相邻节点之间的连接（如果它们也相邻）
    for i in range(len(adjacent_roads)):
        for j in range(i + 1, len(adjacent_roads)):
            road_i = adjacent_roads[i]
            road_j = adjacent_roads[j]
            road_i_info = road_network.get(road_i, {})
            if road_j in road_i_info.get('adjacent', []):
                links.append({
                    'source': road_i,
                    'target': road_j,
                    'value': 0.5,
                    'lineStyle': {
                        'width': 2,
                        'curveness': 0.1
                    }
                })

    return jsonify({
        'success': True,
        'target_road': road,
        'nodes': nodes,
        'links': links,
        'node_count': len(nodes),
        'link_count': len(links)
    })


@app.route('/api/region/simulation', methods=['POST'])
@login_required
def api_region_simulation():
    """分流效果模拟"""
    try:
        data = request.get_json()
        road = data.get('road', '')
        diversion_plan = data.get('diversion_plan', [])

        if not road:
            return jsonify({'success': False, 'msg': '请选择道路'})

        if not diversion_plan:
            return jsonify({'success': False, 'msg': '请提供分流方案'})

        # 加载路网数据
        import json as _json
        with open('static/data/road_network.json', 'r', encoding='utf-8') as f:
            road_network = _json.load(f)

        # 获取目标道路信息
        target_info = road_network.get(road, {})
        target_capacity = target_info.get('capacity', 2000)

        # 模拟当前状态
        _status_to_int = {"畅通": 1, "缓行": 2, "拥堵": 3, "严重拥堵": 4}
        _since = datetime.datetime.now() - datetime.timedelta(hours=1)
        _recent = TrafficHistory.query.filter(
            TrafficHistory.road_name == road,
            TrafficHistory.collect_time >= _since
        ).all()

        if _recent:
            _avg = sum(_status_to_int.get(rec.status, 2) for rec in _recent) / len(_recent)
            current_load = int((_avg - 1) / 3 * 100)
            current_flow = sum(rec.flow for rec in _recent) / len(_recent)
        else:
            current_load = random.randint(70, 95)
            current_flow = target_capacity * current_load / 100

        # 计算分流效果
        total_diversion = sum(item.get('diversion', 0) for item in diversion_plan)
        new_flow = max(0, current_flow - total_diversion)
        new_load = min(100, int((new_flow / target_capacity) * 100))

        # 计算改善程度
        load_reduction = current_load - new_load
        flow_reduction = current_flow - new_flow

        # 评估分流对相邻道路的影响
        impact_analysis = []
        for plan_item in diversion_plan:
            target_road = plan_item.get('road', '')
            diversion_amount = plan_item.get('diversion', 0)

            if target_road:
                road_info = road_network.get(target_road, {})
                road_capacity = road_info.get('capacity', 2000)

                # 获取当前状态
                _recent_target = TrafficHistory.query.filter(
                    TrafficHistory.road_name == target_road,
                    TrafficHistory.collect_time >= _since
                ).all()

                if _recent_target:
                    _avg_target = sum(_status_to_int.get(rec.status, 2) for rec in _recent_target) / len(_recent_target)
                    current_load_target = int((_avg_target - 1) / 3 * 100)
                    current_flow_target = sum(rec.flow for rec in _recent_target) / len(_recent_target)
                else:
                    current_load_target = random.randint(40, 80)
                    current_flow_target = road_capacity * current_load_target / 100

                # 计算分流后的状态
                new_flow_target = current_flow_target + diversion_amount
                new_load_target = min(100, int((new_flow_target / road_capacity) * 100))

                impact_analysis.append({
                    'road': target_road,
                    'current_load': current_load_target,
                    'new_load': new_load_target,
                    'load_increase': new_load_target - current_load_target,
                    'diversion_amount': diversion_amount,
                    'capacity_utilization': f"{new_load_target}%"
                })

        # 计算总体效果评分
        effectiveness_score = 0
        if load_reduction > 20:
            effectiveness_score = 90
        elif load_reduction > 10:
            effectiveness_score = 75
        elif load_reduction > 5:
            effectiveness_score = 60
        else:
            effectiveness_score = 40

        # 检查是否有道路过载
        overloaded_roads = [item for item in impact_analysis if item['new_load'] > 90]

        return jsonify({
            'success': True,
            'target_road': road,
            'simulation_results': {
                'current_load': current_load,
                'new_load': new_load,
                'load_reduction': load_reduction,
                'current_flow': round(current_flow, 0),
                'new_flow': round(new_flow, 0),
                'flow_reduction': round(flow_reduction, 0),
                'effectiveness_score': effectiveness_score,
                'recommendation': '方案效果良好，建议实施' if effectiveness_score >= 60 else '方案效果有限，建议调整'
            },
            'impact_analysis': impact_analysis,
            'overloaded_roads': [item['road'] for item in overloaded_roads],
            'has_overload': len(overloaded_roads) > 0,
            'warning': f'注意：{len(overloaded_roads)}条道路可能过载' if overloaded_roads else '所有道路均在安全容量范围内'
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'msg': str(e), 'trace': traceback.format_exc()})


@app.route('/api/guidance/manual', methods=['POST'])
@login_required
def api_guidance_manual():
    """人工干预 - 创建/修改疏导方案"""
    try:
        data = request.get_json()
        road = data.get('road', '')
        status = int(data.get('status', 1))
        actions = data.get('actions', [])

        if not road or not actions:
            return jsonify({'success': False, 'msg': '参数不完整'})

        import json
        plan = GuidancePlan(
            road=road,
            status=status,
            plan_type='manual',
            actions=json.dumps(actions, ensure_ascii=False),
            operator=session.get('username', 'unknown')
        )
        db.session.add(plan)
        db.session.commit()

        _log_guidance_action(plan.id, 'create', session.get('username', 'unknown'),
                             '人工创建疏导方案')

        return jsonify({
            'success': True,
            'plan_id': plan.id,
            'msg': '疏导方案已创建'
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'msg': str(e), 'trace': traceback.format_exc()})


@app.route('/api/guidance/activate/<int:plan_id>', methods=['POST'])
@login_required
def api_guidance_activate(plan_id):
    """激活疏导方案"""
    try:
        plan = GuidancePlan.query.get(plan_id)
        if not plan:
            return jsonify({'success': False, 'msg': '方案不存在'})

        # 停用同一道路的其他活跃方案
        GuidancePlan.query.filter_by(road=plan.road, is_active=True).update({'is_active': False})

        # 激活当前方案
        plan.is_active = True
        plan.activated_at = datetime.datetime.now()
        plan.reverted_at = None
        db.session.commit()

        _log_guidance_action(plan.id, 'activate', session.get('username', 'unknown'),
                             '激活疏导方案')

        return jsonify({
            'success': True,
            'msg': f'已激活 {plan.road} 的疏导方案'
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'msg': str(e), 'trace': traceback.format_exc()})


@app.route('/api/guidance/revert/<int:plan_id>', methods=['POST'])
@login_required
def api_guidance_revert(plan_id):
    """回退疏导方案"""
    try:
        plan = GuidancePlan.query.get(plan_id)
        if not plan:
            return jsonify({'success': False, 'msg': '方案不存在'})

        # 查找上一个方案
        prev_plan = GuidancePlan.query.filter(
            GuidancePlan.road == plan.road,
            GuidancePlan.id < plan.id
        ).order_by(GuidancePlan.id.desc()).first()

        if prev_plan:
            # 激活上一个方案
            GuidancePlan.query.filter_by(road=plan.road, is_active=True).update({'is_active': False})
            prev_plan.is_active = True
            prev_plan.activated_at = datetime.datetime.now()

        # 标记当前方案为已回退
        plan.is_active = False
        plan.reverted_at = datetime.datetime.now()
        db.session.commit()

        _log_guidance_action(plan.id, 'revert', session.get('username', 'unknown'),
                             f'回退到方案 {prev_plan.id if prev_plan else "无"}')

        return jsonify({
            'success': True,
            'msg': f'已回退 {plan.road} 的疏导方案',
            'reverted_to': prev_plan.id if prev_plan else None
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'msg': str(e), 'trace': traceback.format_exc()})


@app.route('/api/guidance/history', methods=['GET'])
@login_required
def api_guidance_history():
    """查看历史方案"""
    road = request.args.get('road', '').strip()

    # 安全获取 limit，不会崩溃
    try:
        limit = int(request.args.get('limit', 10))
        limit = max(300, min(limit, 100))
    except:
        limit = 20

    query = GuidancePlan.query
    if road:
        query = query.filter_by(road=road)

    plans = query.order_by(GuidancePlan.created_at.desc()).limit(limit).all()

    result = []
    for plan in plans:
        # 时间安全格式化（不会崩溃）
        def format_time(dt):
            return dt.isoformat() if dt else None

        # 计算方案相关指标
        score = calculate_plan_score(plan.status, plan.plan_type)
        effect = calculate_plan_effect(plan.status)
        cost = calculate_plan_cost(plan.status, plan.plan_type)
        scope = calculate_plan_scope(plan.status)
        response_time = calculate_plan_response_time(plan.status)

        result.append({
            'id': plan.id,
            'road': plan.road,
            'status': plan.status,
            'plan_type': plan.plan_type,
            'actions': json.loads(plan.actions) if plan.actions else [],
            'operator': plan.operator,
            'is_active': plan.is_active,
            'score': score,
            'effect': effect,
            'cost': cost,
            'scope': scope,
            'response_time': response_time,
            'created_at': format_time(plan.created_at),
            'activated_at': format_time(plan.activated_at),
            'reverted_at': format_time(plan.reverted_at)
        })

    return jsonify({
        'success': True,
        'plans': result,
        'count': len(result)
    })


@app.route('/api/guidance/logs', methods=['GET'])
@login_required
def api_guidance_logs():
    """查看操作日志"""
    plan_id = request.args.get('plan_id', type=int)
    limit = int(request.args.get('limit', 20))

    query = GuidanceLog.query
    if plan_id:
        query = query.filter_by(plan_id=plan_id)

    logs = query.order_by(GuidanceLog.created_at.desc()).limit(limit).all()

    result = []
    for log in logs:
        result.append({
            'id': log.id,
            'plan_id': log.plan_id,
            'action': log.action,
            'operator': log.operator,
            'note': log.note,
            'created_at': log.created_at.isoformat() if log.created_at else None
        })

    return jsonify({
        'success': True,
        'logs': result,
        'count': len(result)
    })


@app.route('/api/guidance/chart', methods=['GET'])
@login_required
def api_guidance_chart():
    """返回路段近7天按小时聚合的平均拥堵等级，供 ECharts 趋势图使用"""
    road = request.args.get('road', '')
    if not road:
        return jsonify({'success': False, 'msg': '请选择道路'})

    status_to_int = {"畅通": 1, "缓行": 2, "拥堵": 3, "严重拥堵": 4}
    since = datetime.datetime.now() - datetime.timedelta(days=7)

    records = TrafficHistory.query.filter(
        TrafficHistory.road_name == road,
        TrafficHistory.collect_time >= since
    ).all()

    hour_buckets = {h: [] for h in range(24)}
    for rec in records:
        h = rec.collect_time.hour
        s = status_to_int.get(rec.status, 1)
        hour_buckets[h].append(s)

    avg_status = []
    for h in range(24):
        vals = hour_buckets[h]
        avg_status.append(round(sum(vals) / len(vals), 2) if vals else None)

    return jsonify({
        'success': True,
        'road': road,
        'hours': list(range(24)),
        'avg_status': avg_status,
        'has_data': any(v is not None for v in avg_status)
    })


def save_traffic_history(traffic_data):
    """保存实时路况数据到历史表"""
    try:
        collect_time = datetime.datetime.now()
        for data in traffic_data:
            history = TrafficHistory(
                road_name=data['roadName'],
                status=data['status'],
                speed=data['speed'],
                flow=data['flow'],
                color=_get_status_color(data['status']),
                collect_time=collect_time
            )
            db.session.add(history)
        db.session.commit()
        print(f"[TrafficHistory] 保存了 {len(traffic_data)} 条历史数据")
        return True
    except Exception as e:
        db.session.rollback()
        print(f"[TrafficHistory] 保存历史数据失败: {e}")
        return False


@app.route('/api/real-time/all', methods=['GET'])
@login_required
def api_realtime_all():
    """返回10个路口的实时交通数据"""
    import time
    traffic_data = []
    for point in REALTIME_COLLECT_POINTS:
        traffic_info = get_realtime_traffic(point)
        traffic_data.append({
            "roadName": point["name"],  # 显示你定义的路口名
            "speed": traffic_info["speed"],
            "status": traffic_info["status"],
            "flow": traffic_info["flow"]
        })
        time.sleep(0.5)  # 避免限流（和采集代码一致）

    # 保存数据到历史表
    save_traffic_history(traffic_data)

    return jsonify(traffic_data)


@app.route('/traffic')
@login_required
def traffic():
    """实时路况监控页面"""
    return render_template('traffic.html',
                           username=session['username'],
                           role=session['role'])


@app.route('/dashboard')
@login_required
def dashboard():
    """数字大屏页面"""
    from flask import send_from_directory
    return send_from_directory('static/wuhan_traffic_spider', 'index.html')


@app.route('/sandbox')
@login_required
def sandbox():
    """3D交通沙盘页面"""
    from flask import send_from_directory
    return send_from_directory('static/wuhan_traffic_spider', '3d-traffic-sandbox.html')


# ============================================================
# 忘记密码功能
# ============================================================

@app.route('/forgot_password', methods=['GET'])
def forgot_password():
    """忘记密码页面"""
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('forgot_password.html')


@app.route('/verify_email_code', methods=['POST'])
def verify_email_code():
    """验证邮箱验证码"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'msg': '参数错误'})

        email = data.get('email', '').strip()
        code = data.get('code', '').strip()
        purpose = data.get('purpose', 'reset_password')

        if not email or not code:
            return jsonify({'success': False, 'msg': '邮箱和验证码不能为空'})

        # 查询验证码
        vc = VerificationCode.query.filter_by(
            email=email,
            code=code,
            purpose=purpose,
            is_used=False
        ).filter(
            VerificationCode.expires_at > datetime.datetime.now()
        ).order_by(VerificationCode.created_at.desc()).first()

        if not vc:
            return jsonify({'success': False, 'msg': '验证码错误或已过期'})

        return jsonify({'success': True, 'msg': '验证码正确'})

    except Exception as e:
        import traceback
        print("验证验证码错误:", traceback.format_exc())
        return jsonify({'success': False, 'msg': f'验证失败：{str(e)}'})


@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    """重置密码页面"""
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        email_code = request.form.get('email_code', '').strip()
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        # —— 邮箱校验 ——
        if not email or not re.match(r'^[\w.-]+@[\w.-]+\.\w+$', email):
            flash('请输入正确的邮箱地址！', 'error')
            return render_template('reset_password.html', email=email)

        user = User.query.filter_by(email=email).first()
        if not user:
            flash('该邮箱未注册！', 'error')
            return render_template('reset_password.html', email=email)

        # —— 验证码校验 ——
        if not email_code:
            flash('请输入邮箱验证码！', 'error')
            return render_template('reset_password.html', email=email)

        vc = VerificationCode.query.filter_by(
            email=email,
            code=email_code,
            purpose='reset_password',
            is_used=False
        ).filter(
            VerificationCode.expires_at > datetime.datetime.now()
        ).order_by(VerificationCode.created_at.desc()).first()

        if not vc:
            flash('验证码错误或已过期，请重新发送！', 'error')
            return render_template('reset_password.html', email=email)

        # —— 密码校验 ——
        if len(new_password) < 6:
            flash('密码长度至少 6 位！', 'error')
            return render_template('reset_password.html', email=email, email_code=email_code)

        if new_password != confirm_password:
            flash('两次输入的密码不一致！', 'error')
            return render_template('reset_password.html', email=email, email_code=email_code)

        # —— 更新密码 ——
        vc.is_used = True  # 标记验证码已使用
        user.password_hash = generate_password_hash(new_password, method='pbkdf2:sha256')
        db.session.commit()

        flash('密码重置成功！请使用新密码登录', 'success')
        return redirect(url_for('login'))

    # GET 请求：显示重置密码页面
    email = request.args.get('email', '')
    email_code = request.args.get('email_code', '')
    return render_template('reset_password.html', email=email, email_code=email_code)


if __name__ == '__main__':
    import webbrowser
    import threading


    def open_browser():
        """延迟打开浏览器"""
        import time
        time.sleep(1.5)  # 等待服务器启动
        webbrowser.open('http://127.0.0.1:5000')


    # 只在主进程中打开浏览器（debug 模式下 Flask 会创建子进程）
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        print('\n' + '=' * 50)
        print('[INFO] Server starting...')
        print('[INFO] URL: http://127.0.0.1:5000')
        print('[INFO] Opening browser automatically...')
        print('=' * 50 + '\n')
        t = threading.Thread(target=open_browser)
        t.daemon = True
        t.start()

    app.run(debug=True, host='127.0.0.1', port=5000)
