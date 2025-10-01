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
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    last_exception = e
                    
                    if retries >= max_retries:
                        logger.error(f"函数 {func.__name__} 调用失败，已达到最大重试次数: {str(e)}")
                        raise
                    
                    logger.warning(f"函数 {func.__name__} 调用失败，{retries} 秒后重试 ({retries}/{max_retries}): {str(e)}")
                    time.sleep(current_delay)
                    current_delay *= backoff  # 指数退避
            
            # 这行代码理论上不会执行，因为上面已经抛出异常
            raise last_exception
        return wrapper
    return decorator

# 通用模型API调用函数@retry_on_failure(max_retries=2, delay=1)
def call_model_api(model_id, query, http_session=None):
    """通用的模型API调用函数"""
    start_time = time.time()
    
    # 验证和清洗输入
    cleaned_query, error = validate_and_clean_input({'query': query}, {'query': str})
    if error:
        raise ValueError(f"输入验证失败: {error}")
    query = cleaned_query['query']
    
    # 获取模型配置
    config = MODEL_API_CONFIGS.get(model_id)
    if not config:
        raise ValueError(f"不支持的模型: {model_id}")
    
    # 特殊模型处理
    if config.get('special_handler'):
        if model_id == 'gemini':
            return call_gemini_api(query)
        else:
            raise ValueError(f"未实现的特殊模型处理: {model_id}")
    
    # 获取API密钥
    api_key = os.getenv(config['api_key_env'])
    if not api_key:
        raise ValueError(f"{config['api_key_env']} 环境变量未设置")
    
    # 对于需要secret_key的模型
    if 'secret_key_env' in config:
        secret_key = os.getenv(config['secret_key_env'])
        if not secret_key:
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
        response_data = response.json()
        
        # 检查响应状态
        if response.status_code == 200:
            content = config['response_parser'](response_data)
            if content:
                return {
                    'status': 'success',
                    'content': content,
                    'response_time': int((time.time() - start_time) * 1000)
                }
            else:
                raise ValueError("API返回了空内容")
        else:
            error_msg = config['error_parser'](response_data)
            raise ValueError(f"API错误: {error_msg}")
    except json.JSONDecodeError:
        raise ValueError(f"API返回的JSON格式错误: {response.text[:100]}...")
    except Exception as e:
        logger.error(f"调用 {model_id} API 失败: {str(e)}")
        raise

# Gemini模型特殊处理函数@retry_on_failure(max_retries=2, delay=1)
def call_gemini_api(query):
    """Gemini模型API调用特殊处理函数"""
    start_time = time.time()
    
    # 验证和清洗输入
    cleaned_query, error = validate_and_clean_input({'query': query}, {'query': str})
    if error:
        raise ValueError(f"输入验证失败: {error}")
    query = cleaned_query['query']
    
    try:
        # 延迟导入，避免不必要的依赖加载
        import google.generativeai as genai
        
        # 获取API密钥
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY 环境变量未设置")
        
        # 配置Gemini API
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # 发送请求
        response = model.generate_content(query)
        
        return {
            'status': 'success',
            'content': response.text,
            'response_time': int((time.time() - start_time) * 1000)
        }
    except ImportError as e:
        raise ImportError(f"缺少Gemini依赖: {str(e)}")
    except Exception as e:
        logger.error(f"调用Gemini API失败: {str(e)}")
        raise

# 模拟模式处理函数
def handle_simulation_request(model_id, query, **kwargs):
    """处理模拟模式下的请求"""
    import random
    
    start_time = time.time()
    
    # 模拟网络延迟
    delay = random.uniform(1.0, 3.0)  # 随机延迟1-3秒
    time.sleep(delay)
    
    # 模拟响应库
    simulation_responses = {
        'doubao': [
            "这是豆包模型的模拟响应。我可以帮助您解答各种问题。",
            "感谢您的提问！豆包模型为您提供智能化的回答服务。",
            "豆包AI已准备就绪，随时为您提供帮助。"
        ],
        'deepseek': [
            "DeepSeek模型很高兴为您服务。请问有什么可以帮助您的？",
            "这里是DeepSeek AI的回复。我们致力于提供高质量的智能服务。",
            "DeepSeek团队打造的AI助手为您提供专业的回答。"
        ],
        'chatgpt': [
            "ChatGPT为您提供响应。我是一个基于大语言模型的AI助手。",
            "这里是ChatGPT的回答。我可以协助您解决各种问题。",
            "ChatGPT已连接，随时为您提供帮助和支持。"
        ],
        'kimi': [
            "Kimi模型已上线。我们提供安全、高效的智能服务。",
            "感谢使用Kimi AI。我们的模型专注于中文语境下的理解与生成。",
            "Kimi智能助手随时为您提供专业的回答。"
        ],
        'hunyuan': [
            "腾讯混元大模型为您服务。我们致力于技术创新和用户体验。",
            "欢迎使用腾讯混元AI。我们的模型可以处理各种复杂的任务。",
            "混元大模型提供的智能服务，助力您的工作和生活。"
        ],
        'gemini': [
            "Google Gemini模型为您提供响应。我们的技术源自Google DeepMind。",
            "这里是Gemini AI的回答。我们的模型具备多模态理解能力。",
            "Gemini智能助手已准备就绪，可以帮助您解决各种问题。"
        ]
    }
    
    # 随机选择一个模拟响应
    responses = simulation_responses.get(model_id, ["模拟响应"])
    response_text = random.choice(responses)
    
    return {
        'status': 'success',
        'content': response_text,
        'response_time': int((time.time() - start_time) * 1000)
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

# 批处理结果构建函数def build_batch_client_data(model_id, result):
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