import os
import json
import threading
import time
import logging
import traceback
import hashlib
import collections
import random
import jwt

# 为了简化测试，暂时不使用eventlet
# import eventlet
# eventlet.monkey_patch()

# 设置FLASK_APP环境变量
os.environ['FLASK_APP'] = 'app.py'

# 导入第三方库
from flask import Flask, request, jsonify, make_response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from celery import Celery
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# 导入自定义模块
from .utils import validate_and_clean_input, escape_html, generate_cache_key, RateLimiter, safe_json_loads
from .model_handlers import call_model_api, handle_simulation_request, format_model_response
from .task_manager import TaskPoolManager, BatchProcessor, ask_all_models as new_ask_all_models, initialize_task_manager, process_model_request, check_cached_result

# 加载环境变量
load_dotenv()

# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# 创建SocketIO实例
# 使用默认的异步模式
socketio = SocketIO(app, cors_allowed_origins="*")

# 配置CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# 配置Celery
app.config['CELERY_BROKER_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# 创建Celery实例
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# 定义批处理窗口大小（毫秒）
BATCH_WINDOW_MS = 100

# 批处理请求存储
batch_requests = {}

# 记录最后批处理时间
last_batch_process_time = 0

# 活跃的Socket.IO连接
active_connections = set()

# 模拟模式标志
SIMULATION_MODE = os.environ.get('SIMULATION_MODE', 'false').lower() == 'true'

# 导入系统监控库
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning('psutil库未安装，系统资源监控功能将不可用')

# 配置日志记录
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger('multi-ai-app')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# AI模型定义
def get_ai_models():
    # 检查环境变量，看是否在模拟模式下运行
    simulation_mode = os.environ.get('SIMULATION_MODE', 'false').lower() == 'true'
    
    # 基础模型配置
    models = {
        'doubao': {
            'name': 'Doubao',
            'enabled': True,
            'simulation': simulation_mode,
            'api_key': os.environ.get('DOUBAO_API_KEY', '')
        },
        'deepseek': {
            'name': 'DeepSeek',
            'enabled': True,
            'simulation': simulation_mode,
            'api_key': os.environ.get('DEEPSEEK_API_KEY', '')
        },
        'chatgpt': {
            'name': 'ChatGPT',
            'enabled': True,
            'simulation': simulation_mode,
            'api_key': os.environ.get('OPENAI_API_KEY', '')
        },
        'kimi': {
            'name': 'Kimi',
            'enabled': True,
            'simulation': simulation_mode,
            'api_key': os.environ.get('KIMI_API_KEY', '')
        },
        'hunyuan': {
            'name': 'HunYuan',
            'enabled': True,
            'simulation': simulation_mode,
            'api_key': os.environ.get('TX_HUNYUAN_API_KEY', '')
        },
        'gemini': {
            'name': 'Gemini',
            'enabled': True,
            'simulation': simulation_mode,
            'api_key': os.environ.get('GOOGLE_API_KEY', '')
        }
    }
    
    # 如果不在模拟模式下，检查API Key是否存在
    if not simulation_mode:
        for model_id, model_config in models.items():
            if not model_config['api_key']:
                model_config['enabled'] = False
                model_config['disabled_reason'] = 'API Key not configured'
    
    return models

# 初始化AI模型配置
AI_MODELS = get_ai_models()

# 初始化任务管理器
try:
    initialize_task_manager()
    logger.info("任务管理器初始化成功")
except Exception as e:
    logger.error(f"任务管理器初始化失败: {str(e)}")

import psutil
from functools import wraps

# API密钥验证装饰器
def require_api_key(f):
    """验证API密钥的装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 从请求头获取JWT
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authorization header is missing'}), 401
        
        # 提取JWT令牌
        try:
            auth_parts = auth_header.split()
            if len(auth_parts) != 2 or auth_parts[0].lower() != 'bearer':
                return jsonify({'error': 'Invalid authorization format. Use Bearer <token>'}), 401
            token = auth_parts[1]
        except (IndexError, AttributeError):
            return jsonify({'error': 'Invalid authorization format'}), 401
        
        # 验证JWT
        try:
            # 使用环境变量中设置的SECRET_KEY进行验证
            # 确保验证算法一致
            jwt_secret = os.getenv('JWT_SECRET')
            if not jwt_secret:
                logger.error('JWT_SECRET is not configured')
                return jsonify({'error': 'Internal server error during authentication'}), 500
            
            payload = jwt.decode(token, jwt_secret, algorithms=['HS256'], options={
                'verify_signature': True,
                'verify_exp': True,
                'verify_nbf': False,
                'verify_iat': True,
                'verify_aud': False
            })
            # 将用户信息注入到请求对象中
            user_id = payload.get('user_id')
            username = payload.get('username')
            
            # 验证必要的用户信息存在
            if not user_id or not username:
                return jsonify({'error': 'Token missing required information'}), 401
                
            # 将解码后的信息添加到请求上下文中
            request.user_info = payload
            
            logger.info(f'JWT验证通过: {user_id}')
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            logger.warning('JWT token has expired')
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            logger.error(f"JWT验证错误: {str(e)}")
            return jsonify({'error': 'Internal server error during authentication'}), 500
    
    return decorated_function

# JWT验证装饰器
def require_jwt(f):
    """验证请求中是否包含有效的JWT令牌"""
    def decorator(*args, **kwargs):
        # 从环境变量获取JWT密钥
        jwt_secret = os.getenv('JWT_SECRET')
        
        # 如果未配置JWT密钥，使用简单的API密钥验证
        if not jwt_secret:
            logger.warning('没有配置JWT密钥，跳过JWT验证')
            return require_api_key(f)(*args, **kwargs)
        
        # 从请求头中获取JWT令牌
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            logger.warning('Authorization头缺失')
            return make_response(jsonify({
                'success': False,
                'error': 'Unauthorized: Missing Authorization header'
            }), 401)
        
        # 检查Authorization头的格式
        if not auth_header.startswith('Bearer '):
            logger.warning('Authorization头格式错误')
            return make_response(jsonify({
                'success': False,
                'error': 'Unauthorized: Invalid Authorization header format'
            }), 401)
        
        # 提取JWT令牌
        token = auth_header.split(' ')[1]
        
        try:
            # 验证JWT令牌
            decoded = jwt.decode(token, jwt_secret, algorithms=['HS256'])
            logger.info(f'JWT验证通过: {decoded.get("user_id", "unknown")}')
            
            # 将解码后的信息添加到请求上下文中
            request.user_info = decoded
            
        except jwt.ExpiredSignatureError:
            logger.warning('JWT令牌已过期')
            return make_response(jsonify({
                'success': False,
                'error': 'Unauthorized: Token has expired'
            }), 401)
        except jwt.InvalidTokenError:
            logger.warning('无效的JWT令牌')
            return make_response(jsonify({
                'success': False,
                'error': 'Unauthorized: Invalid token'
            }), 401)
        
        return f(*args, **kwargs)
    
    # 保留原始函数的元数据
    decorator.__name__ = f.__name__
    decorator.__doc__ = f.__doc__
    return decorator

# 模型资源配置
MODEL_RESOURCES = {
    'doubao': {
        'timeout': 60,
        'priority': 1,
        'retry_count': 2
    },
    'deepseek': {
        'timeout': 60,
        'priority': 1,
        'retry_count': 2
    },
    'chatgpt': {
        'timeout': 120,
        'priority': 2,
        'retry_count': 3
    },
    'kimi': {
        'timeout': 90,
        'priority': 1,
        'retry_count': 2
    },
    'hunyuan': {
        'timeout': 90,
        'priority': 1,
        'retry_count': 2
    },
    'gemini': {
        'timeout': 120,
        'priority': 2,
        'retry_count': 3
    }
}

# 创建全局的requests session对象，实现HTTP连接池
http_session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=0.3,
    status_forcelist=[500, 502, 503, 504],
)
adapter = HTTPAdapter(pool_connections=20, pool_maxsize=30, max_retries=retry_strategy)
http_session.mount('http://', adapter)
http_session.mount('https://', adapter)

# 缓存配置
CACHE_TTL = 3600  # 缓存有效时间（秒）
CACHE_SIZE_LIMIT = 1000  # 最大缓存条目数

# 资源监控阈值
HIGH_CPU_THRESHOLD = 80  # CPU使用率高阈值（%）
HIGH_MEMORY_THRESHOLD = 85  # 内存使用率高阈值（%）
BATCH_CLIENT_THRESHOLD = 5  # 批处理客户端数量阈值

# 用于存储客户端消息队列的字典
client_message_queues = {}

# 用于保护client_message_queues的锁
message_queue_lock = threading.Lock()

# 发送消息给客户端的函数
def send_message_to_client(client_id, data):
    """发送消息给指定的客户端
    
    参数:
        client_id: 客户端ID
        data: 要发送的消息数据
    """
    try:
        # 尝试通过Socket.IO发送消息
        with app.app_context():
            # 为指定客户端发送消息
            socketio.emit('message', data, room=client_id)
        logger.debug(f'已通过Socket.IO发送消息给客户端: {client_id}')
    except Exception as e:
        # 如果Socket.IO发送失败，将消息存储到队列中供轮询使用
        logger.warning(f'Socket.IO发送消息失败，将消息添加到轮询队列: {str(e)}')
        with message_queue_lock:
            if client_id not in client_message_queues:
                client_message_queues[client_id] = []
            # 添加时间戳
            data_with_timestamp = data.copy()
            data_with_timestamp['queued_at'] = time.time()
            client_message_queues[client_id].append(data_with_timestamp)
            # 限制队列大小，防止内存溢出
            if len(client_message_queues[client_id]) > 100:
                client_message_queues[client_id] = client_message_queues[client_id][-50:]

# LRU缓存实现
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()
        self.last_accessed = {}
        
    def get(self, key):
        if key not in self.cache:
            return None
        # 更新访问时间
        self.last_accessed[key] = time.time()
        # 将访问的元素移到末尾（最近使用）
        self.cache.move_to_end(key)
        return self.cache[key]
        
    def put(self, key, value):
        if key in self.cache:
            # 如果键已存在，先移除
            del self.cache[key]
            # 同时从last_accessed中移除
            if key in self.last_accessed:
                del self.last_accessed[key]
        elif len(self.cache) >= self.capacity:
            # 如果缓存已满，删除最久未使用的元素（最前面的）
            oldest_key, _ = self.cache.popitem(last=False)
            # 同时从last_accessed中移除最久未使用的元素
            if oldest_key in self.last_accessed:
                del self.last_accessed[oldest_key]
        # 将新元素添加到末尾
        self.cache[key] = value
        self.last_accessed[key] = time.time()
        
    def clear(self):
        self.cache.clear()
        self.last_accessed.clear()
        
    def size(self):
        return len(self.cache)
        
    def get_all_keys(self):
        return list(self.cache.keys())
        
    def _sync_last_accessed(self):
        """确保last_accessed字典与cache字典保持同步"""
        # 找出cache中不存在的键并删除
        for key in list(self.last_accessed.keys()):
            if key not in self.cache:
                del self.last_accessed[key]
        # 记录当前时间
        current_time = time.time()
        # 确保cache中的所有键都在last_accessed中
        for key in self.cache:
            if key not in self.last_accessed:
                self.last_accessed[key] = current_time

# 从环境变量获取缓存配置
CACHE_SIZE_LIMIT = int(os.environ.get('CACHE_SIZE_LIMIT', 1000))
CACHE_TTL = int(os.environ.get('CACHE_TTL', 3600))  # 缓存有效时间（秒）

# 创建缓存实例
result_cache = LRUCache(CACHE_SIZE_LIMIT)

# 检查单个缓存项是否过期
def _is_cache_expired(cache, key, current_time=None):
    """检查单个缓存项是否过期"""
    if current_time is None:
        current_time = time.time()
    
    # 检查键是否存在于last_accessed中
    if key not in cache.last_accessed:
        # 如果键不在last_accessed中，则认为已过期
        logger.warning(f'缓存键 {key} 不在last_accessed字典中')
        return True
    
    # 计算已经过的时间
    elapsed_time = current_time - cache.last_accessed[key]
    # 如果已经超过TTL，则缓存项已过期
    return elapsed_time > CACHE_TTL

# 获取缓存，同时检查过期状态
def get_cached_response(model_id, query, **kwargs):
    """获取缓存响应，并在获取时检查是否过期（懒加载检查）"""
    if not result_cache:
        return None
    
    # 生成缓存键
    cache_key = generate_cache_key(model_id, query, **kwargs)
    if not cache_key:
        return None
    
    cached_response = result_cache.get(cache_key)
    
    # 懒加载检查：如果缓存存在但已过期，则返回None
    if cached_response and _is_cache_expired(result_cache, cache_key):
        logger.debug(f'缓存项已过期: {model_id}, {cache_key}')
        # LRUCache没有delete方法，所以这里不做删除操作
        return None
    
    return cached_response

# 缓存管理器线程
class CacheManager(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True
        self.running = True
        
    def run(self):
        while self.running:
            try:
                self.clean_expired_cache()
                # 每10分钟清理一次过期缓存
                time.sleep(600)
            except Exception as e:
                logger.error(f"Error in cache manager: {str(e)}")
                logger.error(traceback.format_exc())
                # 出错时暂停1分钟再继续
                time.sleep(60)
        
    def clean_expired_cache(self):
        """清理过期的缓存项"""
        current_time = time.time()
        expired_keys = []
        
        try:
            # 同步last_accessed和cache字典
            result_cache._sync_last_accessed()
            
            # 找出过期的键
            for key, access_time in list(result_cache.last_accessed.items()):
                if current_time - access_time > CACHE_TTL:
                    expired_keys.append(key)
            
            # 删除过期的键
            for key in expired_keys:
                if key in result_cache.cache:
                    del result_cache.cache[key]
                    if key in result_cache.last_accessed:
                        del result_cache.last_accessed[key]
            
            if expired_keys:
                logger.info(f"Cleaned {len(expired_keys)} expired cache items")
        except Exception as e:
            logger.error(f"Error cleaning expired cache: {str(e)}")

# 启动缓存管理器
cache_manager = CacheManager()
cache_manager.start()

# 生成缓存键
def generate_cache_key(model_id, query, **kwargs):
    """生成唯一的缓存键
    
    参数:
        model_id: 模型ID
        query: 查询文本
        **kwargs: 其他参数
        
    返回:
        唯一的缓存键字符串，如果query为空则返回None
    """
    # 如果查询文本为空，不生成缓存键
    if not query:
        logger.warning('尝试为为空的查询生成缓存键')
        return None
    
    # 创建参数字典，只包含影响结果的关键参数
    cache_params = {
        'model_id': model_id,
        'query': query
    }
    
    # 添加其他可能影响结果的参数
    for key, value in kwargs.items():
        if key in ['temperature', 'max_tokens', 'top_p', 'system_prompt']:
            cache_params[key] = value
    
    # 将字典转换为JSON字符串，然后计算MD5哈希值
    param_str = json.dumps(cache_params, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(param_str.encode('utf-8')).hexdigest()

# 系统资源监控器
class ResourceMonitor:
    def __init__(self):
        if PSUTIL_AVAILABLE:
            self.system = psutil.Process()
        else:
            self.system = None
        self.cpu_warning_threshold = HIGH_CPU_THRESHOLD
        self.memory_warning_threshold = HIGH_MEMORY_THRESHOLD
        self.disk_warning_threshold = 90  # 磁盘使用率警告阈值
        self.last_warning_time = {}
        self.warning_cooldown = 300  # 警告冷却时间(秒)
    
    def get_cpu_usage(self):
        if PSUTIL_AVAILABLE:
            return psutil.cpu_percent(interval=0.1)
        return 0  # 无法监控时返回0
    
    def get_memory_usage(self):
        if PSUTIL_AVAILABLE:
            return self.system.memory_percent()
        return 0  # 无法监控时返回0
    
    def get_network_io(self):
        if PSUTIL_AVAILABLE:
            return psutil.net_io_counters()
        return None
    
    def is_system_busy(self):
        # 如果CPU使用率超过80%或内存使用率超过85%，认为系统繁忙
        if PSUTIL_AVAILABLE:
            return self.get_cpu_usage() > HIGH_CPU_THRESHOLD or self.get_memory_usage() > HIGH_MEMORY_THRESHOLD
        return False  # 无法监控时默认不繁忙
    
    def _should_warn(self, resource_type):
        """检查是否应该发送警告(基于冷却时间)"""
        current_time = time.time()
        last_time = self.last_warning_time.get(resource_type, 0)
        
        if current_time - last_time > self.warning_cooldown:
            self.last_warning_time[resource_type] = current_time
            return True
        return False
    
    def check_all_resources(self):
        """检查所有系统资源"""
        try:
            if not PSUTIL_AVAILABLE:
                return True, {}
            
            # 初始化资源报告
            resources_report = {}
            all_ok = True
            
            # CPU使用情况检查
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1, percpu=False)
                load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
                cpu_ok = cpu_percent < self.cpu_warning_threshold
                all_ok = all_ok and cpu_ok
                
                resources_report['cpu'] = {
                    'usage': cpu_percent,
                    'load_avg': load_avg,
                    'ok': cpu_ok
                }
                
                if not cpu_ok and self._should_warn('cpu'):
                    logger.warning(f"CPU使用率过高: {cpu_percent}%, 系统负载: {load_avg}")
            except Exception as e:
                logger.error(f"检查CPU使用率失败: {str(e)}")
                resources_report['cpu'] = {'usage': 0, 'load_avg': (0, 0, 0), 'ok': True}
            
            # 内存使用情况检查
            try:
                memory = psutil.virtual_memory()
                available_mb = memory.available / (1024 * 1024)
                memory_ok = memory.percent < self.memory_warning_threshold
                all_ok = all_ok and memory_ok
                
                resources_report['memory'] = {
                    'usage': memory.percent,
                    'available_mb': available_mb,
                    'ok': memory_ok
                }
                
                if not memory_ok and self._should_warn('memory'):
                    logger.warning(f"内存使用率过高: {memory.percent}%, 可用内存: {available_mb:.2f}MB")
            except Exception as e:
                logger.error(f"检查内存使用率失败: {str(e)}")
                resources_report['memory'] = {'usage': 0, 'available_mb': 0, 'ok': True}
            
            # 磁盘使用情况检查
            try:
                # 尝试获取系统根目录或Windows系统分区
                partitions = psutil.disk_partitions()
                # 优先检查系统分区
                system_partition = '/'  # Linux/Mac默认
                for p in partitions:
                    if hasattr(psutil, 'OS_WINDOWS') and getattr(psutil, 'OS_WINDOWS', False) and p.mountpoint == 'C:':
                        system_partition = 'C:'
                        break
                    elif p.device.endswith('rootfs'):
                        system_partition = p.mountpoint
                        break
                
                disk = psutil.disk_usage(system_partition)
                available_gb = disk.free / (1024 * 1024 * 1024)
                disk_ok = disk.percent < self.disk_warning_threshold
                all_ok = all_ok and disk_ok
                
                resources_report['disk'] = {
                    'usage': disk.percent,
                    'available_gb': available_gb,
                    'path': system_partition,
                    'ok': disk_ok
                }
                
                if not disk_ok and self._should_warn('disk'):
                    logger.warning(f"磁盘使用率过高: {disk.percent}%({system_partition}), 可用空间: {available_gb:.2f}GB")
            except Exception as e:
                logger.error(f"检查磁盘使用率失败: {str(e)}")
                resources_report['disk'] = {'usage': 0, 'available_gb': 0, 'path': '/', 'ok': True}
            
            # 记录资源使用情况(定期记录，避免日志过多)
            if random.random() < 0.1:  # 10%的概率记录
                cpu_usage = resources_report.get('cpu', {}).get('usage', 0)
                mem_usage = resources_report.get('memory', {}).get('usage', 0)
                disk_usage = resources_report.get('disk', {}).get('usage', 0)
                logger.info(f"系统资源使用情况: CPU={cpu_usage}% Mem={mem_usage}% Disk={disk_usage}%")
            
            return all_ok, resources_report
        except Exception as e:
            logger.error(f"检查系统资源失败: {str(e)}")
            return True, {}  # 出错时默认返回资源正常

# 创建资源监控实例
resource_monitor = ResourceMonitor()

# 任务池管理：使用task_manager.py中的TaskPoolManager类

# 模拟AI模型响应数据
SIMULATION_RESPONSES = {
    'doubao': [
        "这是豆包AI的回答。豆包是字节跳动开发的人工智能助手，能够理解用户的问题并提供准确的信息。",
        "豆包AI为您解答。我可以协助您完成各种任务，包括但不限于获取知识、提供建议和创意生成。",
        "感谢您的提问！豆包AI致力于提供高质量的智能服务体验，希望我的回答能够满足您的需求。"
    ],
    'deepseek': [
        "DeepSeek AI为您提供回答。我们专注于为用户提供专业、准确的信息和建议。",
        "这是深度求索AI的响应。作为一家专注于AGI的公司，我们致力于打造具有通用智能的AI系统。",
        "感谢使用DeepSeek！我们的AI系统能够处理各种复杂问题，并提供结构化、有逻辑的回答。"
    ],
    'chatgpt': [
        "ChatGPT的回答：我是一个由OpenAI开发的人工智能助手，可以帮助您解答问题、提供信息或完成各种文字任务。",
        "这是ChatGPT的回应。作为一个大型语言模型，我能够理解和生成自然语言，并提供有价值的见解。",
        "感谢您的提问！ChatGPT持续学习和进化，希望能为您提供更优质的服务。"
    ],
    'kimi': [
        "这是Kimi AI的回答。我们专注于多模态智能，能够处理文本、图像等多种数据类型。",
        "Kimi AI为您服务！我们的系统集成了最新的AI技术，能够提供精准、全面的信息和建议。",
        "感谢使用Kimi！我们致力于通过AI技术提升用户体验，解决实际问题。"
    ],
    'hunyuan': [
        "腾讯混元大模型为您回答。我们的AI系统融合了多种先进技术，能够提供高质量的智能服务。",
        "这是混元AI的回应。作为腾讯自主研发的大语言模型，我们致力于为用户创造价值。",
        "感谢您的提问！混元大模型不断优化和创新，努力为您提供更优质的回答。"
    ],
    'gemini': [
        "Gemini AI为您提供回答。我们的模型由Google DeepMind开发，具备强大的理解和生成能力。",
        "这是Google Gemini的回应。我们的多模态AI系统能够整合不同类型的信息，提供全面的解答。",
        "感谢使用Gemini！我们致力于通过AI技术赋能用户，帮助您解决各种复杂问题。"
    ]
}

# 重构后的call_ai_model函数
@celery.task(bind=True, name='app.call_ai_model')
def call_ai_model(self, model_id, query, other_results=None, request_id=None, client_ids=None, cache_key=None, **kwargs):
    """调用指定的AI模型进行处理，重构版本"""
    # 记录开始时间
    start_time = time.time()
    logger.info(f"收到模型调用请求: {model_id}, 请求ID: {request_id}")
    
    try:
        # 创建响应回调函数
        def response_callback(result, client_id):
            # 发送结果到Socket.IO（如果请求ID存在）
            if request_id:
                with app.app_context():
                    socketio.emit('ai_response', {
                        'request_id': request_id,
                        'model_id': model_id,
                        'result': result
                    })
            
            # 构造发送给客户端的数据
            client_data = {
                'model': model_id,
                'status': result.get('status', 'success'),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if result.get('status') == 'success':
                client_data['content'] = result.get('content', '')
            else:
                client_data['content'] = result.get('error', '未知错误')
            
            # 发送消息给客户端
            with app.app_context():
                send_message_to_client(client_id, client_data)
        
        # 处理单个客户端ID的情况
        if client_ids is None or len(client_ids) == 0:
            # 生成临时客户端ID
            client_id = f'temp_{time.time()}_{random.randint(1000, 9999)}'
            result = process_model_request(
                model_id, query, client_id,
                use_simulation=SIMULATION_MODE or AI_MODELS.get(model_id, {}).get('simulation', False),
                callback=None,  # 不使用回调，直接返回结果
                http_session=http_session
            )
            return result
        elif len(client_ids) == 1:
            # 单个客户端ID，直接调用process_model_request
            result = process_model_request(
                model_id, query, client_ids[0],
                use_simulation=SIMULATION_MODE or AI_MODELS.get(model_id, {}).get('simulation', False),
                callback=response_callback,
                http_session=http_session
            )
            return result
        else:
            # 多个客户端ID，使用任务池处理
            task_pool = TaskPoolManager()
            
            # 为每个客户端创建不同的任务
            for cid in client_ids:
                task_pool.submit_task(
                    process_model_request,
                    model_id, query, cid,
                    use_simulation=SIMULATION_MODE or AI_MODELS.get(model_id, {}).get('simulation', False),
                    callback=response_callback,
                    http_session=http_session
                )
            
            # 返回一个成功的响应
            return {
                'status': 'success',
                'model_id': model_id,
                'message': '请求已受理，结果将异步返回',
                'timestamp': int(time.time() * 1000),
                'response_time': int((time.time() - start_time) * 1000)
            }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"调用模型时发生错误 ({model_id}): {error_msg}")
        
        # 发送错误结果到Socket.IO
        if request_id:
            with app.app_context():
                socketio.emit('ai_response', {
                    'request_id': request_id,
                    'model_id': model_id,
                    'result': {
                        'status': 'error',
                        'model_id': model_id,
                        'error': error_msg,
                        'timestamp': int(time.time() * 1000),
                        'response_time': int((time.time() - start_time) * 1000)
                    }
                })
        
        # 如果是批处理请求，向多个客户端发送错误
        if client_ids and isinstance(client_ids, list):
            with app.app_context():
                for cid in client_ids:
                    error_data = {
                        'model': model_id,
                        'status': 'error',
                        'content': f'处理请求时发生错误: {str(e)}',
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    send_message_to_client(cid, error_data)
        
        return {
            'status': 'error',
            'model_id': model_id,
            'error': error_msg,
            'timestamp': int(time.time() * 1000),
            'response_time': int((time.time() - start_time) * 1000)
        }

# API路由：向所有AI模型提问
@app.route('/api/ask', methods=['POST'])
@require_jwt
def ask_all_models():
    """向所有AI模型发送问题请求，重构版本"""
    try:
        # 调用任务管理器中的批量查询函数
        response = new_ask_all_models(request, AI_MODELS, result_cache, http_session)
        return response
    except Exception as e:
        logger.error(f"批量查询处理错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'处理请求时发生错误: {str(e)}'
        }), 500

# API路由：重新生成指定AI模型的回答
@app.route('/api/regenerate', methods=['POST'])
@require_jwt
def regenerate_model():
    try:
        # 检查Content-Type
        if request.content_type != 'application/json':
            logger.warning('请求Content-Type不是application/json')
            return jsonify({'success': False, 'message': '请使用application/json格式提交请求'}), 415
            
        # 尝试解析JSON数据
        try:
            data = request.json
            if data is None:
                logger.warning('请求数据为空或格式错误')
                return jsonify({'success': False, 'message': '请求数据为空或格式错误'}), 400
        except Exception as e:
            logger.warning(f'解析JSON请求失败: {str(e)}')
            return jsonify({'success': False, 'message': 'JSON格式错误，请检查请求数据'}), 400
        
        model = data.get('model', '')
        question = data.get('question', '')
        other_results = data.get('other_results', {})
        
        # 输入验证
        cleaned_input, error = validate_and_clean_input(
            {'model': model, 'question': question}, 
            {'model': str, 'question': str}
        )
        if error:
            logger.error(f'输入验证失败: {error}')
            return jsonify({'success': False, 'message': error}), 400
        
        model = cleaned_input['model']
        question = cleaned_input['question']
        
        if not model or model not in AI_MODELS:
            logger.warning(f'无效的模型类型: {model}')
            return jsonify({'success': False, 'message': '无效的模型类型'}), 400
        
        if not question:
            logger.warning(f'模型 {model} 重新生成时问题为空')
            return jsonify({'success': False, 'message': '问题不能为空'}), 400
            
        # 验证问题长度
        if len(question) > 2000:
            logger.warning('问题过长')
            return jsonify({'success': False, 'message': '问题长度不能超过2000个字符'}), 400
        
        # 获取当前客户端ID
        client_id = request.headers.get('X-Client-ID', None)
        if not client_id:
            # 生成临时客户端ID
            client_id = f'temp_{time.time()}_{random.randint(1000, 9999)}'
            logger.info(f'未提供客户端ID，生成临时ID: {client_id}')
        
        logger.info(f'接收到重新生成请求，模型: {model}, 问题: {question[:50]}... 来自客户端: {client_id}')
        
        # 获取模型资源配置
        resources = MODEL_RESOURCES.get(model, {'timeout': 30, 'priority': 1, 'retries': 2})
        
        # 检查缓存
        cached_result = check_cached_result(model, question)
        if cached_result:
            logger.info(f'从缓存中获取{model}模型的回答结果')
            # 发送缓存的结果给客户端
            send_message_to_client(client_id, cached_result)
            return jsonify({
                'success': True, 
                'message': f'已从缓存获取{model}的回答',
                'client_id': client_id,
                'from_cache': True,
                'model_resources': resources
            })
        
        # 提交到任务池
        task_pool = TaskPoolManager()
        try:
            task_pool.submit_task(
                process_model_request, 
                model, question, client_id, 
                use_simulation=SIMULATION_MODE,
                http_session=http_session
            )
            logger.info(f'已请求重新生成{model}的回答')
            return jsonify({
                'success': True, 
                'message': f'已请求重新生成{model}的回答',
                'client_id': client_id,
                'model_resources': resources
            })
        except Exception as e:
            logger.error(f'提交{model}模型任务失败: {str(e)}')
            # 发送错误消息给客户端
            error_data = {
                'model': model,
                'status': 'error',
                'content': f'提交任务失败: {str(e)}',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            send_message_to_client(client_id, error_data)
            return jsonify({
                'success': False,
                'message': f'提交任务失败: {str(e)}'
            }), 500
        
    except Exception as e:
        logger.error(f'重新生成回答失败: {str(e)}')
        logger.debug(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': '处理请求时发生错误，请稍后重试',
            'error_type': str(type(e).__name__)
        }), 500

# API路由：检查服务器状态
@app.route('/api/status', methods=['GET'])
@require_jwt
def check_status():
    try:
        # 检查Celery连接状态
        try:
            celery_result = celery.control.ping(timeout=1)
            celery_status = len(celery_result) > 0
            logger.debug(f'Celery状态检查结果: {celery_status}')
        except Exception as e:
            celery_status = False
            logger.warning(f'检查Celery状态时出错: {str(e)}')
        
        status_data = {
            'success': True,
            'server_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'active_connections': len(active_connections),
            'celery_connected': celery_status,
            'simulation_mode': SIMULATION_MODE,
            'available_models': list(AI_MODELS.keys())
        }
        
        logger.debug(f'服务器状态: {json.dumps(status_data)}')
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f'检查服务器状态失败: {str(e)}')
        logger.debug(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': '检查服务器状态时发生错误',
            'error_type': str(type(e).__name__)
        }), 500

# API路由：用于HTTP轮询获取消息
@app.route('/api/poll', methods=['GET'])
@require_jwt
def poll_messages():
    """客户端通过HTTP轮询获取消息
    
    查询参数:
        client_id: 客户端ID（必需）
        last_id: 最后收到的消息ID（可选，用于增量获取）
    """
    try:
        # 获取客户端ID
        client_id = request.args.get('client_id')
        if not client_id:
            return jsonify({'success': False, 'error': 'client_id is required'}), 400
        
        # 获取最后收到的消息ID（如果有）
        last_id = request.args.get('last_id')
        
        # 检查是否有该客户端的消息队列
        with message_queue_lock:
            if client_id in client_message_queues and client_message_queues[client_id]:
                # 获取所有消息
                messages = client_message_queues[client_id]
                # 清空队列，防止重复获取
                client_message_queues[client_id] = []
                
                # 为消息添加唯一ID
                for i, msg in enumerate(messages):
                    msg['id'] = f"{client_id}_{int(time.time())}_{i}"
                
                logger.info(f'客户端 {client_id} 通过HTTP轮询获取了 {len(messages)} 条消息')
                return jsonify({
                    'success': True,
                    'messages': messages,
                    'has_more': False
                })
        
        # 如果没有新消息，返回空结果
        return jsonify({
            'success': True,
            'messages': [],
            'has_more': False
        })
        
    except Exception as e:
        logger.error(f'处理轮询请求失败: {str(e)}')
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

# 定期清理过期的消息队列
def cleanup_message_queues():
    """定期清理过期的消息队列"""
    while True:
        try:
            current_time = time.time()
            with message_queue_lock:
                expired_clients = []
                for client_id, messages in client_message_queues.items():
                    # 检查队列中的消息是否已过期（5分钟）
                    if messages and current_time - messages[0].get('queued_at', current_time) > 300:
                        expired_clients.append(client_id)
                
                # 删除过期的客户端消息队列
                for client_id in expired_clients:
                    del client_message_queues[client_id]
                    logger.info(f'已清理过期的消息队列: {client_id}')
        except Exception as e:
            logger.error(f'清理消息队列时出错: {str(e)}')
        
        # 每5分钟执行一次清理
        time.sleep(300)

# 启动消息队列清理线程
cleanup_thread = threading.Thread(target=cleanup_message_queues, daemon=True)
cleanup_thread.start()

# 注意：请使用run_backend.py启动应用，而不是直接运行此文件
# 这样可以确保Flask和Celery服务正确启动并协同工作

