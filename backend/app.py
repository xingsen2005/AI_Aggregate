import os
import json
import threading
import time
import logging
import traceback
import hashlib
import collections
import random
# 导入eventlet并进行monkey patch
import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from celery import Celery
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# 创建SocketIO实例
# 配置SocketIO使用eventlet作为异步模式
async_mode = 'eventlet'
socketio = SocketIO(app, async_mode=async_mode, cors_allowed_origins="*")

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
        elif len(self.cache) >= self.capacity:
            # 如果缓存已满，删除最久未使用的元素（最前面的）
            self.cache.popitem(last=False)
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

# 从环境变量获取缓存配置
CACHE_SIZE_LIMIT = int(os.environ.get('CACHE_SIZE_LIMIT', 1000))
CACHE_TTL = int(os.environ.get('CACHE_TTL', 3600))  # 缓存有效时间（秒）

# 创建缓存实例
result_cache = LRUCache(CACHE_SIZE_LIMIT)

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
        
        # 找出过期的键
        for key, access_time in result_cache.last_accessed.items():
            if current_time - access_time > CACHE_TTL:
                expired_keys.append(key)
        
        # 删除过期的键
        for key in expired_keys:
            if key in result_cache.cache:
                del result_cache.cache[key]
                del result_cache.last_accessed[key]
        
        if expired_keys:
            logger.info(f"Cleaned {len(expired_keys)} expired cache items")

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
        唯一的缓存键字符串
    """
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

# 资源监控类
class ResourceMonitor:
    def __init__(self):
        if PSUTIL_AVAILABLE:
            self.system = psutil.Process()
        else:
            self.system = None
    
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

# 创建资源监控实例
resource_monitor = ResourceMonitor()

# 并发控制的TaskPool类
class TaskPool:
    def __init__(self, max_workers=6):
        self.max_workers = max_workers
        self.semaphore = threading.Semaphore(max_workers)
        self.active_tasks = []
        self.lock = threading.Lock()
    
    def submit(self, task_func, *args, **kwargs):
        with self.semaphore:
            try:
                return task_func(*args, **kwargs)
            except Exception as e:
                logger.error(f'任务执行失败: {str(e)}')
                raise
    
    def submit_task(self, target, args=(), kwargs=None, priority=1):
        """
        提交任务到线程池
        
        Args:
            target: 要执行的函数
            args: 函数参数（元组）
            kwargs: 函数关键字参数（字典）
            priority: 任务优先级（数字越小优先级越高）
        """
        if kwargs is None:
            kwargs = {}
        
        # 创建一个线程来执行任务
        thread = threading.Thread(target=self._execute_task, args=(target, args, kwargs))
        
        # 存储任务信息
        with self.lock:
            self.active_tasks.append((thread, priority))
        
        # 启动线程
        thread.start()
        
        return thread
    
    def _execute_task(self, target, args, kwargs):
        with self.semaphore:
            try:
                target(*args, **kwargs)
            except Exception as e:
                logger.error(f'任务执行失败: {str(e)}')
                # 可以选择在这里添加重试逻辑
            finally:
                # 任务完成后从活跃任务列表中移除
                with self.lock:
                    # 找到并移除对应的线程
                    for i, (thread, _) in enumerate(self.active_tasks):
                        if thread == threading.current_thread():
                            self.active_tasks.pop(i)
                            break
    
    def wait_completion(self):
        """等待所有任务完成"""
        threads = []
        with self.lock:
            threads = [thread for thread, _ in self.active_tasks]
        
        for thread in threads:
            if thread.is_alive():
                thread.join()
    
    def get_active_count(self):
        """获取当前活跃任务数量"""
        with self.lock:
            return len(self.active_tasks)

# 创建全局TaskPool实例
task_pool = TaskPool(max_workers=min(10, len(AI_MODELS) * 2))

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

# 模拟AI模型处理函数
@celery.task(bind=True, name='app.call_ai_model')
def call_ai_model(self, model_id, query, other_results=None, request_id=None, client_ids=None, cache_key=None, **kwargs):
    """调用指定的AI模型进行处理
    
    参数:
        model_id: 模型ID
        query: 用户查询内容
        other_results: 其他模型的结果（用于批处理请求）
        request_id: 请求ID
        client_ids: 客户端ID列表（用于批处理请求）
        cache_key: 可选的缓存键
        **kwargs: 其他参数
    
    返回:
        包含响应内容和状态的字典
    """
    # 记录开始时间
    start_time = time.time()
    
    # 获取模型配置
    model_config = AI_MODELS.get(model_id)
    resource_config = MODEL_RESOURCES.get(model_id, {})
    
    if not model_config:
        return {
            'status': 'error',
            'model_id': model_id,
            'error': f'Model {model_id} not found',
            'timestamp': int(time.time() * 1000),
            'response_time': 0
        }
    
    if not model_config.get('enabled', False):
        return {
            'status': 'error',
            'model_id': model_id,
            'error': model_config.get('disabled_reason', 'Model not enabled'),
            'timestamp': int(time.time() * 1000),
            'response_time': 0
        }
    
    try:
        # 处理模拟模式
        if model_config.get('simulation', False) or SIMULATION_MODE:
            # 模拟网络延迟
            delay = random.uniform(1.0, 3.0)  # 随机延迟1-3秒
            time.sleep(delay)
            
            # 随机选择一个模拟响应
            responses = SIMULATION_RESPONSES.get(model_id, ["模拟响应"])
            response_text = random.choice(responses)
            
            result = {
                'status': 'success',
                'model_id': model_id,
                'content': response_text,
                'timestamp': int(time.time() * 1000),
                'response_time': int((time.time() - start_time) * 1000)
            }
        else:
            # 这里是实际调用AI模型API的代码
            # 目前使用模拟响应
            result = {
                'status': 'success',
                'model_id': model_id,
                'content': f"[{model_id.upper()}] This is a simulated response to: '{query}'",
                'timestamp': int(time.time() * 1000),
                'response_time': int((time.time() - start_time) * 1000)
            }
            
        # 如果是成功结果，将其存入缓存
        if result['status'] == 'success' and query:
            # 如果没有提供缓存键，生成一个
            if not cache_key:
                cache_key = generate_cache_key(model_id, query, **kwargs)
            
            # 将结果存入缓存
            result_cache.put(cache_key, result)
            logger.info(f"Cached result for model {model_id}, cache key: {cache_key[:10]}...")
        
        # 发送结果到Socket.IO（如果请求ID存在）
        if request_id:
            with app.app_context():
                socketio.emit('ai_response', {
                    'request_id': request_id,
                    'model_id': model_id,
                    'result': result
                })
        
        # 如果是批处理请求，向多个客户端发送结果
        if client_ids and isinstance(client_ids, list):
            with app.app_context():
                for cid in client_ids:
                    # 构造发送给客户端的数据
                    client_data = {
                        'model': model_id,
                        'status': result.get('status', 'success'),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    if result.get('status') == 'success':
                        client_data['content'] = result.get('content', '')
                    else:
                        client_data['content'] = result.get('error', 'Unknown error')
                    
                    # 发送消息给客户端
                    send_message_to_client(cid, client_data)
        
        return result
    except Exception as e:
        # 处理异常
        error_msg = str(e)
        logger.error(f"Error in call_ai_model ({model_id}): {error_msg}")
        logger.error(traceback.format_exc())
        
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

# API路由：提交问题给所有AI模型
@app.route('/api/ask', methods=['POST'])
def ask_all_models():
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
        
        question = data.get('question', '')
        other_results = data.get('other_results', {})
        
        # 验证问题长度
        if not question:
            logger.warning('提交问题为空')
            return jsonify({'success': False, 'message': '问题不能为空'}), 400
        
        if len(question) > 2000:
            logger.warning('问题过长')
            return jsonify({'success': False, 'message': '问题长度不能超过2000个字符'}), 400
        
        # 验证other_results格式
        if not isinstance(other_results, dict):
            logger.warning('other_results格式错误')
            return jsonify({'success': False, 'message': 'other_results必须是一个对象'}), 400
        
        # 获取当前客户端ID
        client_id = request.headers.get('X-Client-ID', None)
        if not client_id:
            # 生成临时客户端ID
            client_id = f'temp_{time.time()}_{random.randint(1000, 9999)}'
            logger.info(f'未提供客户端ID，生成临时ID: {client_id}')
        
        logger.info(f'接收到问题: {question[:50]}... 来自客户端: {client_id}')
        
        # 检查系统资源使用情况
        if PSUTIL_AVAILABLE:
            resource_monitor = ResourceMonitor()
            cpu_usage = resource_monitor.get_cpu_usage()
            memory_usage = resource_monitor.get_memory_usage()
            
            logger.info(f'当前系统资源使用情况 - CPU: {cpu_usage:.1f}%, 内存: {memory_usage:.1f}%')
            
            # 根据系统资源动态调整参数
            if cpu_usage > HIGH_CPU_THRESHOLD or memory_usage > HIGH_MEMORY_THRESHOLD:
                logger.warning(f'系统资源使用率过高 (CPU: {cpu_usage:.1f}%, 内存: {memory_usage:.1f}%)，将启用资源保护模式')
                # 如果资源紧张，增加批处理窗口，减少并发数
                batch_window = BATCH_WINDOW_MS * 2
            else:
                batch_window = BATCH_WINDOW_MS
        else:
            batch_window = BATCH_WINDOW_MS
            logger.info('系统监控模块不可用，使用默认批处理窗口')
        
        # 检查每个模型的缓存
        cached_results = {}
        models_to_process = []
        
        for model in AI_MODELS.keys():
            # 构建缓存键
            cache_key = generate_cache_key(model, question, other_results=other_results)
            # 检查缓存
            cached_result = result_cache.get(cache_key)
            if cached_result and cached_result.get('status') == 'success':
                cached_results[model] = cached_result
                logger.info(f'从缓存获取{model}模型的结果')
                # 立即将缓存结果发送给客户端
                send_message_to_client(client_id, cached_result)
            else:
                models_to_process.append(model)
        
        # 如果所有模型都有缓存结果，直接返回
        if len(cached_results) == len(AI_MODELS):
            logger.info('所有模型结果均命中缓存，直接返回')
            return jsonify({
                'success': True,
                'message': '所有模型结果均命中缓存',
                'client_id': client_id,
                'models_count': len(AI_MODELS),
                'all_from_cache': True,
                'cached_models': list(cached_results.keys())
            })
        
        # 如果部分模型有缓存结果，记录信息
        if cached_results:
            logger.info(f'已找到{len(cached_results)}个模型的缓存结果，需处理{len(models_to_process)}个模型')
            # 如果没有需要处理的模型，直接返回
            if not models_to_process:
                return jsonify({
                    'success': True,
                    'message': '所有模型结果均命中缓存',
                    'client_id': client_id,
                    'models_count': len(AI_MODELS),
                    'all_from_cache': True,
                    'cached_models': list(cached_results.keys())
                })
        
        # 批处理逻辑
        current_time = time.time()
        
        # 根据问题内容生成批处理键（对问题进行简单归一化）
        normalized_question = question.lower().strip()[:100]  # 取前100个字符作为批处理键
        batch_key = hashlib.md5(normalized_question.encode()).hexdigest()[:8]
        
        # 检查是否可以批处理
        if batch_key not in batch_requests:
            batch_requests[batch_key] = {
                'question': question,
                'other_results': other_results,
                'clients': [client_id],
                'created_at': current_time,
                'models_to_process': models_to_process.copy()  # 只包含需要处理的模型
            }
        else:
            # 将当前客户端添加到现有批次
            if client_id not in batch_requests[batch_key]['clients']:
                batch_requests[batch_key]['clients'].append(client_id)
                
            # 合并需要处理的模型列表，确保没有重复
            current_models = set(batch_requests[batch_key].get('models_to_process', AI_MODELS.keys()))
            new_models = set(models_to_process)
            batch_requests[batch_key]['models_to_process'] = list(current_models.union(new_models))
        
        # 检查是否应该处理批次
        should_process = False
        
        # 如果超过了批处理窗口时间
        if current_time - batch_requests[batch_key]['created_at'] > batch_window / 1000:
            should_process = True
        
        # 如果批次中的客户端数量达到阈值
        if len(batch_requests[batch_key]['clients']) >= BATCH_CLIENT_THRESHOLD:
            should_process = True
        
        if should_process:
            # 处理批次请求
            batch = batch_requests.pop(batch_key)
            question_to_process = batch['question']
            other_results_to_process = batch['other_results']
            client_ids = batch['clients']
            # 获取需要处理的模型列表，如果没有则处理所有模型
            models_to_process = batch.get('models_to_process', AI_MODELS.keys())
            
            logger.info(f'处理批处理请求，批次大小: {len(client_ids)}，问题: {question_to_process[:30]}...')
            
            # 异步调用所有AI模型
            task_ids = {}
            try:
                # 并行提交需要处理的模型任务
                for model in models_to_process:
                    try:
                        # 构建缓存键
                        cache_key = generate_cache_key(model, question_to_process, other_results=other_results_to_process)
                        
                        # 如果是批处理请求，传递client_ids参数
                        if len(client_ids) > 1:
                            task = call_ai_model.delay(model, question_to_process, other_results_to_process, None, client_ids, cache_key)
                        else:
                            task = call_ai_model.delay(model, question_to_process, other_results_to_process, client_id, None, cache_key)
                        task_ids[model] = task.id
                    except Exception as e:
                        logger.error(f'提交{model}模型任务失败: {str(e)}')
                        # 发送错误消息给所有客户端
                        error_data = {
                            'model': model,
                            'status': 'error',
                            'content': f'提交任务失败: {str(e)}',
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        if len(client_ids) > 1:
                            for cid in client_ids:
                                send_message_to_client(cid, error_data)
                        else:
                            send_message_to_client(client_id, error_data)
                
                # 检查是否有成功提交的任务
                if not task_ids:
                    # 如果全部失败且处于模拟模式，使用备用处理方式
                    if SIMULATION_MODE:
                        logger.info('所有模型任务提交失败，切换到备用处理方式')
                        # 为每个客户端创建一个后台线程来处理请求
                        for cid in client_ids:
                            threading.Thread(target=handle_simulation_request, args=(cid, question_to_process, other_results_to_process, models_to_process)).start()
                        logger.info(f'已使用备用方式处理问题，批次大小: {len(client_ids)}，模型数量: {len(models_to_process)}')
                        return jsonify({
                            'success': True,
                            'message': '问题已通过备用方式提交给所有AI模型',
                            'client_id': client_id if len(client_ids) == 1 else 'batch',
                            'models_count': len(models_to_process),
                            'using_fallback': True,
                            'batch_size': len(client_ids)
                        })
                    else:
                        # 非模拟模式下返回错误
                        return jsonify({
                            'success': False,
                            'message': '所有模型任务提交失败，请稍后重试',
                            'error_type': 'CeleryError'
                        }), 503
            except Exception as model_error:
                logger.error(f'提交模型任务失败: {str(model_error)}')
                # 当Redis不可用或Celery连接失败时，使用备用方式处理
                if SIMULATION_MODE or 'Redis' in str(model_error) or 'Connection' in str(model_error):
                    logger.info('Redis不可用或Celery连接失败，使用备用方式处理请求')
                    # 为每个客户端创建一个后台线程来处理请求
                    for cid in client_ids:
                        threading.Thread(target=handle_simulation_request, args=(cid, question_to_process, other_results_to_process, models_to_process)).start()
                    logger.info(f'已使用备用方式处理问题，批次大小: {len(client_ids)}，模型数量: {len(models_to_process)}')
                    return jsonify({
                        'success': True,
                        'message': 'Redis不可用，问题已通过备用方式提交给所有AI模型',
                        'client_id': client_id if len(client_ids) == 1 else 'batch',
                        'models_count': len(models_to_process),
                        'using_fallback': True,
                        'batch_size': len(client_ids)
                    })
            else:
                return jsonify({
                    'success': False,
                    'message': '提交模型请求失败，请稍后重试',
                    'error_type': str(type(model_error).__name__)
                }), 503  # 服务不可用
        
        # 更新最后批处理时间
        last_batch_process_time = time.time()
        
        logger.info(f'成功提交所有模型任务，批次大小: {len(client_ids)}')
        
        # 清理过期的批处理请求
        def cleanup_old_batches():
            current_time = time.time()
            expired_keys = []
            for key, batch in batch_requests.items():
                if current_time - batch['created_at'] > 30000 / 1000:  # 30秒过期时间
                    expired_keys.append(key)
            
            for key in expired_keys:
                # 处理过期的批次
                expired_batch = batch_requests.pop(key)
                expired_client_ids = expired_batch['clients']
                logger.info(f'处理过期的批处理请求，批次大小: {len(expired_client_ids)}')
                
                # 异步处理过期的批次
                def process_expired_batch():
                    try:
                        for cid in expired_client_ids:
                            threading.Thread(target=handle_simulation_request, args=(cid, expired_batch['question'], expired_batch['other_results'])).start()
                    except Exception as e:
                        logger.error(f'处理过期批次失败: {str(e)}')
                
                threading.Thread(target=process_expired_batch).start()
        
        # 异步清理过期的批处理请求
        threading.Thread(target=cleanup_old_batches).start()
        
        # 返回响应
        response_data = {
            'success': True,
            'message': '问题已提交给所有AI模型',
            'client_id': client_id if len(client_ids) == 1 else 'batch',
            'models_count': len(AI_MODELS),
            'task_ids': task_ids
        }
        
        # 如果是批处理请求，添加批处理信息
        if len(client_ids) > 1:
            response_data['batch_size'] = len(client_ids)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f'提交问题失败: {str(e)}')
        logger.debug(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': '处理请求时发生错误，请稍后重试',
            'error_type': str(type(e).__name__)
        }), 500

# API路由：重新生成指定AI模型的回答
@app.route('/api/regenerate', methods=['POST'])
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
        
        # 检查系统资源使用情况
        if PSUTIL_AVAILABLE:
            resource_monitor = ResourceMonitor()
            cpu_usage = resource_monitor.get_cpu_usage()
            memory_usage = resource_monitor.get_memory_usage()
            
            logger.info(f'当前系统资源使用情况 - CPU: {cpu_usage:.1f}%, 内存: {memory_usage:.1f}%')
            
            # 如果资源紧张，记录警告
            if cpu_usage > HIGH_CPU_THRESHOLD or memory_usage > HIGH_MEMORY_THRESHOLD:
                logger.warning(f'系统资源使用率过高 (CPU: {cpu_usage:.1f}%, 内存: {memory_usage:.1f}%)')
        
        # 获取模型资源配置
        resources = MODEL_RESOURCES.get(model, {'timeout': 30, 'priority': 1, 'retries': 2})
        
        # 构建缓存键
        cache_key = f'{model}_regenerate_{hash(question + str(other_results))}'
        
        # 检查缓存中是否有结果
        cached_result = result_cache.get(cache_key)
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
        
        # 异步调用指定AI模型
        try:
            task = call_ai_model.delay(model, question, other_results, client_id, cache_key)
            logger.info(f'已请求重新生成{model}的回答')
            return jsonify({
                'success': True, 
                'message': f'已请求重新生成{model}的回答',
                'client_id': client_id,
                'task_id': task.id,
                'model_resources': resources
            })
        except Exception as e:
            logger.error(f'提交{model}模型任务失败: {str(e)}')
            # 如果提交任务失败且处于模拟模式或Redis连接失败，使用备用方式处理
            if SIMULATION_MODE or 'Redis' in str(e) or 'Connection' in str(e):
                logger.info('Redis不可用或Celery连接失败，使用备用方式处理请求')
                # 使用TaskPool处理备用请求
                task_pool.submit_task(
                    target=simulate_model_response,
                    args=(model, question, other_results, client_id, cache_key),
                    priority=resources['priority']
                )
                return jsonify({
                    'success': True,
                    'message': f'已通过备用方式重新生成{model}模型的回答',
                    'client_id': client_id,
                    'using_fallback': True,
                    'model_resources': resources
                })
            else:
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

# 注意：请使用run_backend.py启动应用，而不是直接运行此文件
# 这样可以确保Flask和Celery服务正确启动并协同工作