import os
import time
import json
import logging
import requests
import threading
from functools import wraps
from .utils import validate_and_clean_input, escape_html, generate_cache_key

logger = logging.getLogger(__name__)

# 模型配置映射表，用于统一管理模型API调用参数
MODEL_API_CONFIGS = {
    'doubao': {
        'api_key_env': 'DOUBAO_API_KEY',
        'url': 'https://api.doubao.com/v1/chat/completions',
        'headers_template': lambda api_key: {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        },
        'data_template': lambda query: {
            'model': 'doubao-pro',
            'messages': [{'role': 'user', 'content': query}],
            'temperature': 0.7
        },
        'response_parser': lambda response_data: response_data['choices'][0]['message']['content'] if response_data.get('choices') else None,
        'error_parser': lambda response_data: response_data.get('error', {}).get('message', 'Unknown error')
    },
    'deepseek': {
        'api_key_env': 'DEEPSEEK_API_KEY',
        'url': 'https://api.deepseek.com/chat/completions',
        'headers_template': lambda api_key: {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        },
        'data_template': lambda query: {
            'model': 'deepseek-chat',
            'messages': [{'role': 'user', 'content': query}],
            'temperature': 0.7
        },
        'response_parser': lambda response_data: response_data['choices'][0]['message']['content'] if response_data.get('choices') else None,
        'error_parser': lambda response_data: response_data.get('error', {}).get('message', 'Unknown error')
    },
    'chatgpt': {
        'api_key_env': 'OPENAI_API_KEY',
        'url': 'https://api.openai.com/v1/chat/completions',
        'headers_template': lambda api_key: {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        },
        'data_template': lambda query: {
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content': query}],
            'temperature': 0.7
        },
        'response_parser': lambda response_data: response_data['choices'][0]['message']['content'] if response_data.get('choices') else None,
        'error_parser': lambda response_data: response_data.get('error', {}).get('message', 'Unknown error')
    },
    'kimi': {
        'api_key_env': 'KIMI_API_KEY',
        'url': 'https://api.kimi.moonshot.cn/v1/chat/completions',
        'headers_template': lambda api_key: {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        },
        'data_template': lambda query: {
            'model': 'moonshot-v1-8k',
            'messages': [{'role': 'user', 'content': query}],
            'temperature': 0.7
        },
        'response_parser': lambda response_data: response_data['choices'][0]['message']['content'] if response_data.get('choices') else None,
        'error_parser': lambda response_data: response_data.get('error', {}).get('message', 'Unknown error')
    },
    'hunyuan': {
        'api_key_env': 'HUNYUAN_API_KEY',
        'secret_key_env': 'HUNYUAN_SECRET_KEY',
        'url': 'https://hunyuan.tencentcloudapi.com',
        'headers_template': lambda api_key, secret_key: {
            'Content-Type': 'application/json',
            'Authorization': f'HMAC-SHA256 {api_key}:{secret_key}'
        },
        'data_template': lambda query: {
            'Action': 'ChatCompletions',
            'Version': '2023-09-01',
            'Region': 'ap-guangzhou',
            'Model': 'hunyuan-pro',
            'Messages': [{'Role': 'user', 'Content': query}],
            'Temperature': 0.7
        },
        'response_parser': lambda response_data: response_data.get('Result'),
        'error_parser': lambda response_data: response_data.get('Error', {}).get('Message', 'Unknown error')
    },
    'gemini': {
        'api_key_env': 'GEMINI_API_KEY',
        'special_handler': True  # 需要特殊处理的模型
    }
}

# 重试装饰器
def retry_on_failure(max_retries=3, delay=1, backoff=2):
    """API调用失败自动重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            last_exception = None
            func_name = func.__name__
            
            while retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    # 重试成功后，如果是重试过的请求，记录恢复信息
                    if retries > 0:
                        logger.info(f"函数 {func_name} 在 {retries} 次重试后成功恢复")
                    return result
                except Exception as e:
                    retries += 1
                    last_exception = e
                    
                    if retries >= max_retries:
                        break
                    
                    # 指数退避延迟，添加一些随机性以避免雪崩效应
                    jitter_delay = min(current_delay * (0.9 + 0.2 * random.random()), 60)  # 最大延迟60秒
                    logger.warning(f"函数 {func_name} 调用失败，第{retries}次重试 ({retries}/{max_retries}): {str(e)}，延迟{jitter_delay:.2f}秒")
                    time.sleep(jitter_delay)
                    current_delay *= backoff  # 指数退避
            
            # 达到最大重试次数后仍然失败，记录错误并抛出异常
            error_msg = f"函数 {func_name} 达到最大重试次数({max_retries})，仍然失败: {str(last_exception)}"
            logger.error(error_msg)
            # 保留原始异常信息的同时添加重试上下文
            raise Exception(error_msg) from last_exception
        return wrapper
    return decorator

@retry_on_failure(max_retries=2, delay=1)
def call_model_api(model_id, query, http_session=None):
    """通用的模型API调用函数"""
    start_time = time.time()
    
    # 验证输入参数
    if not model_id or not query:
        logger.error("缺少必要参数: model_id或query")
        raise ValueError("缺少必要参数: model_id或query")
    
    # 验证参数类型
    if not isinstance(model_id, str) or not isinstance(query, str):
        logger.error("参数类型错误: model_id和query必须是字符串")
        raise TypeError("参数类型错误: model_id和query必须是字符串")
    
    # 清理并验证输入内容长度
    query = query.strip()
    if not query:
        logger.error("空的query参数")
        raise ValueError("空的query参数")
    
    # 记录请求信息，但避免记录完整的query内容
    query_preview = query[:100] + '...' if len(query) > 100 else query
    logger.info(f"处理模型请求: {model_id}, query: {query_preview}")
    
    # 验证和清洗输入
    cleaned_query, error = validate_and_clean_input({'query': query}, {'query': str})
    if error:
        logger.error(f"输入验证失败: {error}")
        raise ValueError(f"输入验证失败: {error}")
    query = cleaned_query['query']
    
    # 获取模型配置
    config = MODEL_API_CONFIGS.get(model_id)
    if not config:
        logger.error(f"不支持的模型: {model_id}")
        raise ValueError(f"不支持的模型: {model_id}")
    
    # 特殊模型处理
    if config.get('special_handler'):
        if model_id == 'gemini':
            return call_gemini_api(query)
        else:
            logger.error(f"未实现的特殊模型处理: {model_id}")
            raise ValueError(f"未实现的特殊模型处理: {model_id}")
    
    # 获取API密钥
    api_key = os.getenv(config['api_key_env'])
    if not api_key:
        logger.error(f"{config['api_key_env']} 环境变量未设置")
        raise ValueError(f"{config['api_key_env']} 环境变量未设置")
    
    # 对于需要secret_key的模型
    if 'secret_key_env' in config:
        secret_key = os.getenv(config['secret_key_env'])
        if not secret_key:
            logger.error(f"{config['secret_key_env']} 环境变量未设置")
            raise ValueError(f"{config['secret_key_env']} 环境变量未设置")
        headers = config['headers_template'](api_key, secret_key)
    else:
        headers = config['headers_template'](api_key)
    
    # 构建请求数据
    data = config['data_template'](query)
    
    # 使用提供的session或创建新的session
    session = http_session or requests.Session()
    
    try:
        # 发送请求
        response = session.post(config['url'], headers=headers, data=json.dumps(data))
        
        # 检查响应状态
        if response.status_code != 200:
            # 为不同的错误状态码提供更具体的错误信息
            error_msg = f"API调用失败: {model_id}, 状态码: {response.status_code}"
            logger.error(error_msg)
            if response.status_code == 429:
                raise requests.exceptions.HTTPError("请求过于频繁，请稍后再试", response=response)
            elif response.status_code == 401:
                raise requests.exceptions.HTTPError("API密钥无效或已过期", response=response)
            elif response.status_code >= 500:
                raise requests.exceptions.HTTPError("服务器内部错误，请稍后再试", response=response)
            
            # 尝试解析错误响应
            try:
                response_data = response.json()
                error_msg = config['error_parser'](response_data)
                raise ValueError(f"API错误: {error_msg}")
            except (json.JSONDecodeError, KeyError):
                raise ValueError(f"API返回非JSON格式错误: {response.text[:100]}...")
        
        # 尝试解析正常响应
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            raise ValueError(f"API返回的JSON格式错误: {response.text[:100]}...")
        
        content = config['response_parser'](response_data)
        if content:
            logger.info(f"API调用成功: {model_id}")
            return {
                'status': 'success',
                'content': content,
                'response_time': int((time.time() - start_time) * 1000)
            }
        else:
            logger.warning(f"{model_id} API返回了空内容")
            raise ValueError("API返回了空内容")
    except Exception as e:
        # 捕获所有异常并记录详细信息
        logger.error(f"调用 {model_id} API 失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

@retry_on_failure(max_retries=2, delay=1)
def call_gemini_api(query):
    """Gemini模型API调用特殊处理函数"""
    start_time = time.time()
    
    # 验证输入参数
    if not query:
        logger.error("缺少必要参数: query")
        raise ValueError("缺少必要参数: query")
    
    # 验证参数类型
    if not isinstance(query, str):
        logger.error("参数类型错误: query必须是字符串")
        raise TypeError("参数类型错误: query必须是字符串")
    
    # 清理并验证输入内容长度
    query = query.strip()
    if not query:
        logger.error("空的query参数")
        raise ValueError("空的query参数")
    
    # 记录请求信息，但避免记录完整的query内容
    query_preview = query[:100] + '...' if len(query) > 100 else query
    logger.info(f"处理Gemini请求, query: {query_preview}")
    
    # 验证和清洗输入
    cleaned_query, error = validate_and_clean_input({'query': query}, {'query': str})
    if error:
        logger.error(f"输入验证失败: {error}")
        raise ValueError(f"输入验证失败: {error}")
    query = cleaned_query['query']
    
    try:
        # 延迟导入，避免不必要的依赖加载
        import google.generativeai as genai
        
        # 获取API密钥
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY 环境变量未设置")
            raise ValueError("GEMINI_API_KEY 环境变量未设置")
        
        # 配置Gemini API
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # 发送请求
        response = model.generate_content(query)
        
        logger.info("Gemini API调用成功")
        return {
            'status': 'success',
            'content': response.text,
            'response_time': int((time.time() - start_time) * 1000)
        }
    except ImportError as e:
        logger.error(f"缺少Gemini依赖: {str(e)}")
        raise ImportError(f"缺少Gemini依赖: {str(e)}")
    except Exception as e:
        # 捕获所有异常并记录详细信息
        logger.error(f"调用Gemini API失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

# 模拟模式处理函数
def handle_simulation_request(model_id, query, **kwargs):
    """处理模拟模式下的请求"""
    import random
    
    # 延迟导入以避免循环依赖
    from .app import SIMULATION_RESPONSES
    
    start_time = time.time()
    
    # 模拟网络延迟
    delay = random.uniform(1.0, 3.0)  # 随机延迟1-3秒
    time.sleep(delay)
    
    # 随机选择一个模拟响应
    responses = SIMULATION_RESPONSES.get(model_id, ["模拟响应"])
    response_text = random.choice(responses)
    
    # 构造完整的模拟响应，包含所有必要字段
    return {
        'status': 'success',
        'content': response_text,
        'response_time': int((time.time() - start_time) * 1000),
        'timestamp': int(time.time()),  # 添加时间戳字段，用于缓存检查
        'model': model_id  # 添加模型标识
    }

# 模型响应格式化函数
def format_model_response(model_id, api_result, error=None):
    """格式化模型响应结果"""
    timestamp = int(time.time() * 1000)
    
    if error or api_result.get('status') != 'success':
        return {
            'status': 'error',
            'model_id': model_id,
            'error': error or api_result.get('error', '未知错误'),
            'timestamp': timestamp,
            'response_time': api_result.get('response_time', 0)
        }
    
    return {
        'status': 'success',
        'model_id': model_id,
        'content': escape_html(api_result['content']),  # 再次确保内容安全
        'timestamp': timestamp,
        'response_time': api_result['response_time']
    }

# 批处理结果构建函数
def build_batch_client_data(model_id, result):
    """构建批处理模式下发送给客户端的数据"""
    client_data = {
        'model': model_id,
        'status': result.get('status', 'success'),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if result.get('status') == 'success':
        client_data['content'] = result.get('content', '')
    else:
        client_data['content'] = result.get('error', 'Unknown error')
    
    return client_data