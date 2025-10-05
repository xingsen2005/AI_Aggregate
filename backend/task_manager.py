import os
import time
import json
import logging
import threading
import concurrent.futures
from collections import defaultdict
from .utils import validate_and_clean_input, generate_cache_key, RateLimiter
from .model_handlers import call_model_api, handle_simulation_request, format_model_response

logger = logging.getLogger(__name__)

# 任务池配置
TASK_POOL_CONFIG = {
    'max_workers': min(10, len(['doubao', 'deepseek', 'chatgpt', 'kimi', 'hunyuan', 'gemini']) * 2),
    'thread_name_prefix': 'ModelCall'
}

# 批处理配置
BATCH_CONFIG = {
    'window_ms': 100,  # 批处理窗口大小（毫秒）
    'max_batch_size': 5  # 最大批处理数量
}

# 全局批处理状态
BATCH_STATE = {
    'active_batches': defaultdict(dict),  # 存储活跃的批处理任务
    'batch_lock': threading.RLock()  # 用于保护批处理状态的锁
}

# 速率限制器
RATE_LIMITER = RateLimiter(requests_per_minute=20)  # 每分钟最多20个请求

# 任务池管理器
class TaskPoolManager:
    """管理AI模型调用的任务池"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TaskPoolManager, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """初始化任务池"""
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=TASK_POOL_CONFIG['max_workers'],
            thread_name_prefix=TASK_POOL_CONFIG['thread_name_prefix']
        )
        self.running_tasks = set()
        self.tasks_lock = threading.RLock()
        logger.info(f"任务池已初始化，最大工作线程数: {TASK_POOL_CONFIG['max_workers']}")
    
    def submit_task(self, func, *args, **kwargs):
        """提交任务到任务池"""
        with self.tasks_lock:
            # 创建Future对象并添加到运行任务集合
            future = self.executor.submit(func, *args, **kwargs)
            self.running_tasks.add(future)
            
            # 添加回调以在任务完成时从集合中移除
            future.add_done_callback(lambda f: self._task_done(f))
            return future
    
    def _task_done(self, future):
        """任务完成回调"""
        with self.tasks_lock:
            if future in self.running_tasks:
                self.running_tasks.remove(future)
            
            # 处理可能的异常
            try:
                future.result()
            except Exception as e:
                logger.error(f"任务执行异常: {str(e)}")
    
    def get_active_task_count(self):
        """获取活跃任务数量"""
        with self.tasks_lock:
            return len(self.running_tasks)
    
    def shutdown(self, wait=True):
        """关闭任务池"""
        with self.tasks_lock:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=wait)
                logger.info("任务池已关闭")

# 批处理管理器
class BatchProcessor:
    """批处理管理器"""
    @staticmethod
    def get_batch_key(model_id):
        """生成批处理键"""
        # 使用模型ID和时间窗口生成批处理键
        current_time = int(time.time() * 1000)
        time_window = current_time // BATCH_CONFIG['window_ms']
        return f"{model_id}_{time_window}"
    
    @staticmethod
    def add_to_batch(model_id, question, client_id, callback=None):
        """将请求添加到批处理"""
        batch_key = BatchProcessor.get_batch_key(model_id)
        
        with BATCH_STATE['batch_lock']:
            # 检查批处理是否已存在
            if batch_key not in BATCH_STATE['active_batches']:
                # 创建新的批处理
                BATCH_STATE['active_batches'][batch_key] = {
                    'model_id': model_id,
                    'questions': [],
                    'client_ids': [],
                    'callbacks': [],
                    'created_time': time.time()
                }
            
            # 添加到批处理
            batch = BATCH_STATE['active_batches'][batch_key]
            batch['questions'].append(question)
            batch['client_ids'].append(client_id)
            if callback:
                batch['callbacks'].append(callback)
            
            # 如果达到最大批处理大小，立即处理
            if len(batch['questions']) >= BATCH_CONFIG['max_batch_size']:
                BatchProcessor.process_batch(batch_key)
                return True  # 表示批处理已触发
            
            # 否则设置定时器
            if len(batch['questions']) == 1:  # 只有第一个请求需要设置定时器
                timer = threading.Timer(
                    BATCH_CONFIG['window_ms'] / 1000.0,
                    BatchProcessor.process_batch, 
                    args=[batch_key]
                )
                timer.daemon = True
                timer.start()
            
            return False  # 表示请求已添加到批处理，但尚未触发处理
    
    @staticmethod
    def process_batch(batch_key):
        """处理批处理"""
        with BATCH_STATE['batch_lock']:
            # 检查批处理是否存在
            if batch_key not in BATCH_STATE['active_batches']:
                return
            
            # 获取批处理数据
            batch = BATCH_STATE['active_batches'].pop(batch_key)
        
        model_id = batch['model_id']
        questions = batch['questions']
        client_ids = batch['client_ids']
        callbacks = batch['callbacks']
        
        logger.info(f"处理批处理请求: {model_id}, 请求数量: {len(questions)}")
        
        # 在实际应用中，这里可以优化为批量API调用
        # 为简化，我们这里仍然逐个调用
        for i, (question, client_id) in enumerate(zip(questions, client_ids)):
            try:
                # 调用模型API
                result = call_model_api(model_id, question)
                formatted_result = format_model_response(model_id, result)
                
                # 调用回调函数（如果有）
                if i < len(callbacks) and callbacks[i]:
                    callbacks[i](formatted_result, client_id)
            except Exception as e:
                logger.error(f"批处理请求失败: {model_id}, 客户端ID: {client_id}, 错误: {str(e)}")
    
    @staticmethod
    def cleanup_old_batches(max_age_seconds=30):
        """清理过期的批处理"""
        current_time = time.time()
        
        with BATCH_STATE['batch_lock']:
            to_remove = [
                key for key, batch in BATCH_STATE['active_batches'].items()
                if current_time - batch['created_time'] > max_age_seconds
            ]
            
            for key in to_remove:
                logger.warning(f"清理过期的批处理: {key}")
                del BATCH_STATE['active_batches'][key]

# 缓存检查函数
def check_cached_result(model_id, question):
    """检查是否存在缓存结果"""
    from .app import result_cache  # 延迟导入以避免循环依赖
    
    if not result_cache or not model_id or not question:
        return None
    
    cache_key = generate_cache_key(model_id, question)
    cached_result = result_cache.get(cache_key)
    
    if cached_result:
        logger.info(f"命中缓存: {model_id}")
        # 检查缓存是否过期（可选的额外检查）
        if isinstance(cached_result, dict) and 'timestamp' in cached_result:
            cache_age = time.time() - cached_result['timestamp']
            if cache_age > 3600:  # 1小时后过期
                # LRUCache没有delete方法，所以这里不做删除操作
                return None
        
        return cached_result
    
    return None

# 缓存结果函数
def cache_result(model_id, question, result):
    """缓存API结果"""
    from .app import result_cache  # 延迟导入以避免循环依赖
    
    if not result_cache or not model_id or not question:
        return
    
    try:
        cache_key = generate_cache_key(model_id, question)
        # 添加时间戳以便于后续过期检查
        if isinstance(result, dict) and 'timestamp' not in result:
            result['timestamp'] = time.time()
        
        # 使用LRUCache的put方法缓存结果
        result_cache.put(cache_key, result)
        logger.info(f"结果已缓存: {model_id}")
    except Exception as e:
        logger.error(f"缓存结果失败: {model_id}, 错误: {str(e)}")

# 系统资源监控函数
def check_system_resources():
    """检查系统资源使用情况"""
    try:
        # 简化实现，实际应用中可能需要更详细的监控
        import psutil
        
        # 获取系统内存使用情况
        memory = psutil.virtual_memory()
        # 获取CPU使用率（1秒采样）
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 检查是否超过阈值
        memory_threshold = 80  # 80%内存使用率
        cpu_threshold = 80  # 80%CPU使用率
        
        return {
            'memory_usage': memory.percent,
            'cpu_usage': cpu_percent,
            'is_overloaded': memory.percent > memory_threshold or cpu_percent > cpu_threshold
        }
    except ImportError:
        logger.warning("psutil库未安装，无法进行详细的系统资源监控")
        return {
            'memory_usage': None,
            'cpu_usage': None,
            'is_overloaded': False
        }
    except Exception as e:
        logger.error(f"系统资源监控失败: {str(e)}")
        return {
            'memory_usage': None,
            'cpu_usage': None,
            'is_overloaded': False
        }

# 发送响应到客户端的回调函数
def create_response_callback(socketio=None, send_message_func=None):
    """创建发送响应到客户端的回调函数"""
    def response_callback(result, client_id):
        """处理API响应并发送给客户端"""
        try:
            # 格式化响应数据
            client_data = {
                'model': result.get('model_id'),
                'status': result.get('status'),
                'content': result.get('content', ''),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 使用提供的发送函数或默认的Socket.IO
            if send_message_func:
                send_message_func(client_id, client_data)
            elif socketio:
                socketio.emit('ai_response', client_data, room=client_id)
                logger.info(f"已通过Socket.IO发送响应: {client_id}, {result.get('model_id')}")
        except Exception as e:
            logger.error(f"发送响应到客户端失败: {client_id}, 错误: {str(e)}")
    
    return response_callback

# 批量查询所有模型的主函数
def ask_all_models(question, client_id, use_simulation=False, socketio=None, http_session=None, send_message_func=None):
    """批量查询所有支持的AI模型"""
    # 记录开始时间
    start_time = time.time()
    logger.info(f"收到批量查询请求，客户端ID: {client_id}")
    
    # 输入验证
    cleaned_input, error = validate_and_clean_input(
        {'question': question, 'client_id': client_id}, 
        {'question': str, 'client_id': str}
    )
    if error:
        logger.error(f"输入验证失败: {error}")
        return {'status': 'error', 'error': error}
    
    question = cleaned_input['question']
    client_id = cleaned_input['client_id']
    
    # 速率限制检查
    if not RATE_LIMITER.is_allowed(client_id):
        logger.warning(f"请求频率过高: {client_id}")
        return {'status': 'error', 'error': '请求频率过高，请稍后重试'}
    
    # 检查系统资源
    resource_status = check_system_resources()
    if resource_status['is_overloaded']:
        logger.warning(f"系统资源过载，拒绝请求: {client_id}")
        return {
            'status': 'error', 
            'error': '系统资源暂时不足，请稍后重试',
            'resource_status': resource_status
        }
    
    # 支持的模型列表
    ai_models = ['doubao', 'deepseek', 'chatgpt', 'kimi', 'hunyuan', 'gemini']
    
    # 创建响应回调函数
    response_callback = create_response_callback(socketio, send_message_func)
    
    # 获取任务池管理器实例
    task_pool = TaskPoolManager()
    
    # 提交任务到任务池
    submitted_tasks = []
    for model_id in ai_models:
        try:
            # 检查缓存
            cached_result = check_cached_result(model_id, question)
            if cached_result:
                # 直接发送缓存结果
                response_callback(cached_result, client_id)
                logger.info(f"使用缓存结果: {model_id}, 客户端ID: {client_id}")
                continue
            
            # 提交到任务池
            if use_simulation:
                task = task_pool.submit_task(
                    process_model_request, 
                    model_id, question, client_id, 
                    use_simulation=use_simulation, 
                    callback=response_callback,
                    http_session=http_session
                )
            else:
                # 尝试批处理
                added_to_batch = BatchProcessor.add_to_batch(
                    model_id, question, client_id, callback=response_callback
                )
                
                if not added_to_batch:
                    # 如果没有添加到批处理，直接提交任务
                    task = task_pool.submit_task(
                        process_model_request, 
                        model_id, question, client_id, 
                        use_simulation=use_simulation, 
                        callback=response_callback,
                        http_session=http_session
                    )
                    submitted_tasks.append(task)
        except Exception as e:
            logger.error(f"提交任务失败: {model_id}, 错误: {str(e)}")
            # 发送错误响应
            error_result = format_model_response(model_id, {}, error=str(e))
            response_callback(error_result, client_id)
    
    # 记录请求处理完成时间
    processing_time = int((time.time() - start_time) * 1000)
    logger.info(f"批量查询请求处理完成，客户端ID: {client_id}, 耗时: {processing_time}ms, 提交任务数: {len(submitted_tasks)}")
    
    return {
        'status': 'success',
        'message': '请求已受理，结果将异步返回',
        'model_count': len(ai_models),
        'processing_time': processing_time
    }

# 处理单个模型请求的函数
def process_model_request(model_id, question, client_id, use_simulation=False, callback=None, http_session=None):
    """处理单个模型的请求"""
    start_time = time.time()
    result = None
    
    try:
        # 检查缓存
        cached_result = check_cached_result(model_id, question)
        if cached_result:
            result = cached_result
            logger.info(f"处理请求时命中缓存: {model_id}, 客户端ID: {client_id}")
        else:
            # 调用模型API
            if use_simulation:
                api_result = handle_simulation_request(model_id, question)
            else:
                api_result = call_model_api(model_id, question, http_session)
            
            # 格式化结果
            result = format_model_response(model_id, api_result)
            
            # 缓存结果
            cache_result(model_id, question, result)
        
        # 调用回调函数
        if callback:
            callback(result, client_id)
        
        return result
    except Exception as e:
        error_msg = f"调用 {model_id} API失败: {str(e)}"
        logger.error(error_msg)
        
        # 格式化错误结果
        error_result = format_model_response(model_id, {}, error=error_msg)
        
        # 调用回调函数发送错误响应
        if callback:
            callback(error_result, client_id)
        
        return error_result
    finally:
        # 记录处理时间
        processing_time = int((time.time() - start_time) * 1000)
        logger.debug(f"模型请求处理完成: {model_id}, 客户端ID: {client_id}, 耗时: {processing_time}ms")

# 定期清理批处理和缓存的函数
def start_cleanup_threads():
    """启动定期清理线程"""
    # 批处理清理线程
    batch_cleanup_thread = threading.Thread(
        target=BatchProcessor.cleanup_old_batches, 
        args=(30,),  # 每30秒清理一次
        daemon=True
    )
    batch_cleanup_thread.start()
    
    # 定期清理的主循环
    def cleanup_loop():
        while True:
            try:
                # 清理批处理
                BatchProcessor.cleanup_old_batches(max_age_seconds=60)
                
                # 清理过期的缓存（如果缓存支持的话）
                # 这里省略具体实现，取决于缓存后端
                
                # 休眠一段时间
                time.sleep(60)  # 每分钟执行一次
            except Exception as e:
                logger.error(f"清理线程执行异常: {str(e)}")
                time.sleep(10)  # 发生异常时休眠更短时间
    
    # 启动主清理线程
    main_cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    main_cleanup_thread.start()
    
    logger.info("清理线程已启动")

# 初始化函数
def initialize_task_manager():
    """初始化任务管理器"""
    # 创建任务池管理器实例
    task_pool = TaskPoolManager()
    
    # 启动清理线程
    start_cleanup_threads()
    
    logger.info("任务管理器初始化完成")

# 确保初始化函数在模块加载时执行
# 注意：在实际应用中，可能需要在应用启动时手动调用initialize_task_manager()
# 这里不直接执行，以避免导入时自动初始化
# initialize_task_manager()