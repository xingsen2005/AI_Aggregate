import re
import json
import time
import logging
import hashlib
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# 输入验证和清洗函数
def validate_and_clean_input(data, schema=None):
    """验证和清洗输入数据"""
    if not data:
        return None, "输入数据为空"
        
    # 清洗字符串类型的字段
    cleaned_data = {}
    for key, value in data.items():
        if isinstance(value, str):
            # 去除首尾空白字符
            cleaned_value = value.strip()
            # 防止XSS攻击，进行HTML转义
            cleaned_value = escape_html(cleaned_value)
            # 限制字符串长度
            if len(cleaned_value) > 5000:
                return None, f"字段 '{key}' 长度超过限制"
            cleaned_data[key] = cleaned_value
        else:
            cleaned_data[key] = value
    
    # 如果提供了schema，进行进一步验证
    if schema:
        for field, field_type in schema.items():
            if field in cleaned_data:
                if not isinstance(cleaned_data[field], field_type):
                    return None, f"字段 '{field}' 类型错误，应为 {field_type.__name__}"
            elif field_type != bool and field_type != int and field_type != float:
                # 对于非基本类型，要求字段必须存在
                return None, f"缺少必需字段 '{field}'"
    
    return cleaned_data, None

# HTML转义函数
def escape_html(text):
    """HTML转义函数，防止XSS攻击"""
    if not isinstance(text, str):
        return text
    
    html_escape_table = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
        '/': '&#x2F;',
        '(': '&#40;',
        ')': '&#41;',
        '+': '&#43;',
        '=': '&#61;',
    }
    
    return ''.join(html_escape_table.get(c, c) for c in text)

# 生成缓存键
def generate_cache_key(model_id, query, **kwargs):
    """生成用于缓存的唯一键"""
    # 创建一个包含所有相关参数的字典
    key_dict = {
        'model_id': model_id,
        'query': query[:1000],  # 限制查询长度
        'timestamp': int(time.time() / 3600),  # 每小时更新一次缓存键
    }
    
    # 添加额外的关键字参数
    for k, v in kwargs.items():
        if isinstance(v, dict) or isinstance(v, list):
            # 对于复杂类型，将其转换为JSON字符串并取前100个字符
            try:
                key_dict[k] = json.dumps(v, ensure_ascii=False)[:100]
            except:
                key_dict[k] = str(v)[:100]
        else:
            key_dict[k] = str(v)[:100]
    
    # 生成MD5哈希值作为缓存键
    key_str = json.dumps(key_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

# 请求限流类
class RateLimiter:
    """请求限流工具类"""
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.request_times = {}
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id):
        """检查客户端是否被允许发送请求"""
        current_time = time.time()
        
        with self.lock:
            # 获取客户端的请求历史
            if client_id not in self.request_times:
                self.request_times[client_id] = []
            
            # 过滤掉一分钟前的请求
            self.request_times[client_id] = [
                t for t in self.request_times[client_id] 
                if current_time - t < 60
            ]
            
            # 检查是否超过限制
            if len(self.request_times[client_id]) >= self.requests_per_minute:
                return False, 60 - (current_time - self.request_times[client_id][0])
            
            # 记录新请求时间
            self.request_times[client_id].append(current_time)
            
            # 清理过期的客户端数据
            self._clean_expired_clients(current_time)
            
            return True, 0
    
    def _clean_expired_clients(self, current_time):
        """清理长时间没有活动的客户端数据"""
        expired_clients = []
        for client_id, times in self.request_times.items():
            if not times or current_time - times[-1] > 300:  # 5分钟没有活动
                expired_clients.append(client_id)
        
        for client_id in expired_clients:
            del self.request_times[client_id]

# 创建全局的限流实例
global_rate_limiter = RateLimiter(requests_per_minute=60)

# 格式化错误响应
def format_error_response(error_type, message, status_code=400):
    """格式化错误响应"""
    return {
        'success': False,
        'error_type': error_type,
        'message': message,
        'status_code': status_code
    }

# 格式化成功响应
def format_success_response(data=None, message="操作成功"):
    """格式化成功响应"""
    response = {
        'success': True,
        'message': message
    }
    if data:
        response['data'] = data
    return response

# 安全的JSON解析
def safe_json_loads(json_str):
    """安全地解析JSON字符串"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None

# 敏感信息屏蔽函数
def mask_sensitive_info(data, sensitive_fields=None):
    """屏蔽敏感信息"""
    if sensitive_fields is None:
        sensitive_fields = ['api_key', 'password', 'token', 'secret', 'key']
    
    if isinstance(data, dict):
        return {
            k: '***' if any(sf in k.lower() for sf in sensitive_fields) else mask_sensitive_info(v, sensitive_fields)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [mask_sensitive_info(item, sensitive_fields) for item in data]
    else:
        return data

# 系统资源监控帮助函数
def format_resource_usage(cpu_usage, memory_usage):
    """格式化系统资源使用情况"""
    return {
        'cpu_usage_percent': round(cpu_usage, 1),
        'memory_usage_percent': round(memory_usage, 1),
        'timestamp': int(time.time() * 1000)
    }

# 检查参数命名规范
def check_parameter_naming(params, allowed_params):
    """检查参数命名是否符合规范"""
    invalid_params = []
    for param in params:
        if param not in allowed_params:
            invalid_params.append(param)
    
    if invalid_params:
        return False, f"参数名不规范: {', '.join(invalid_params)}"
    
    return True, None