import threading
import time
import collections
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class AppContext:
    """应用程序上下文类，用于封装全局状态，提供线程安全的访问"""
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AppContext, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """初始化应用程序上下文"""
        # 线程锁，用于保护共享资源
        self._batch_lock = threading.RLock()
        self._connection_lock = threading.RLock()
        self._message_queue_lock = threading.RLock()
        
        # 批处理相关状态
        self._batch_requests = {}
        self._last_batch_process_time = 0
        
        # Socket.IO连接状态
        self._active_connections = set()
        
        # 客户端消息队列
        self._client_message_queues = {}
        
        # HTTP会话（连接池）
        self._http_session = None
        
        # 模型配置（将在app.py中初始化）
        self._ai_models = None
        self._model_resources = None
        
        # 模拟模式标志
        self._simulation_mode = False
        
    # 批处理相关方法
    def add_batch_request(self, model_id, request_data):
        """添加请求到批处理队列"""
        with self._batch_lock:
            if model_id not in self._batch_requests:
                self._batch_requests[model_id] = []
            self._batch_requests[model_id].append(request_data)
    
    def get_batch_requests(self):
        """获取所有批处理请求并清空队列"""
        with self._batch_lock:
            requests = self._batch_requests.copy()
            self._batch_requests = {}
            return requests
    
    def set_last_batch_process_time(self, timestamp):
        """设置最后批处理时间"""
        with self._batch_lock:
            self._last_batch_process_time = timestamp
    
    def get_last_batch_process_time(self):
        """获取最后批处理时间"""
        with self._batch_lock:
            return self._last_batch_process_time
    
    # Socket.IO连接相关方法
    def add_active_connection(self, client_id):
        """添加活跃连接"""
        with self._connection_lock:
            self._active_connections.add(client_id)
    
    def remove_active_connection(self, client_id):
        """移除活跃连接"""
        with self._connection_lock:
            if client_id in self._active_connections:
                self._active_connections.remove(client_id)
    
    def get_active_connections(self):
        """获取所有活跃连接"""
        with self._connection_lock:
            return self._active_connections.copy()
    
    def is_client_active(self, client_id):
        """检查客户端是否活跃"""
        with self._connection_lock:
            return client_id in self._active_connections
    
    # 消息队列相关方法
    def add_message_to_queue(self, client_id, data):
        """添加消息到客户端队列"""
        with self._message_queue_lock:
            if client_id not in self._client_message_queues:
                self._client_message_queues[client_id] = []
            # 添加时间戳
            data_with_timestamp = data.copy()
            data_with_timestamp['queued_at'] = time.time()
            self._client_message_queues[client_id].append(data_with_timestamp)
            # 限制队列大小，防止内存溢出
            if len(self._client_message_queues[client_id]) > 100:
                self._client_message_queues[client_id] = self._client_message_queues[client_id][-50:]
    
    
    def get_client_messages(self, client_id):
        """获取客户端的所有消息并清空队列"""
        with self._message_queue_lock:
            messages = self._client_message_queues.pop(client_id, [])
            return messages
    
    # HTTP会话相关方法
    def get_http_session(self):
        """获取HTTP会话（连接池）"""
        if self._http_session is None:
            self._http_session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=50)
            self._http_session.mount("http://", adapter)
            self._http_session.mount("https://", adapter)
        return self._http_session
    
    # 模型配置相关方法
    def set_ai_models(self, models):
        """设置AI模型配置"""
        self._ai_models = models
    
    def get_ai_models(self):
        """获取AI模型配置"""
        return self._ai_models
    
    def set_model_resources(self, resources):
        """设置模型资源配置"""
        self._model_resources = resources
    
    def get_model_resources(self):
        """获取模型资源配置"""
        return self._model_resources
    
    # 模拟模式相关方法
    def set_simulation_mode(self, mode):
        """设置模拟模式"""
        self._simulation_mode = mode
    
    def is_simulation_mode(self):
        """检查是否处于模拟模式"""
        return self._simulation_mode
    
    # 清理方法
    def cleanup_stale_messages(self, max_age_seconds=3600):
        """清理过期的消息队列"""
        current_time = time.time()
        with self._message_queue_lock:
            for client_id, messages in list(self._client_message_queues.items()):
                # 过滤出未过期的消息
                fresh_messages = [msg for msg in messages if current_time - msg.get('queued_at', 0) < max_age_seconds]
                if not fresh_messages:
                    # 如果没有新鲜消息，删除整个队列
                    del self._client_message_queues[client_id]
                else:
                    # 否则更新队列
                    self._client_message_queues[client_id] = fresh_messages

# 创建全局应用上下文实例
app_context = AppContext()