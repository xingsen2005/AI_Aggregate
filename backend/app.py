import os
import json
import threading
import time
import logging
import traceback
# 导入eventlet并进行monkey patch
import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from celery import Celery
import requests
from dotenv import load_dotenv

# 配置日志记录
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger('multi-ai-app')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 加载环境变量
load_dotenv()

# 检查是否处于模拟模式
SIMULATION_MODE = os.getenv('SIMULATION_MODE', 'false').lower() == 'true'

# 导入random模块用于生成随机值
import random

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')
app.config['REDIS_URL'] = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# 启用CORS，配置更安全的源限制
# 在生产环境中，应该将origins设置为具体的前端域名
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

# 初始化SocketIO
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000"])

# 初始化Celery
if not SIMULATION_MODE:
    celery = Celery(app.name, broker=app.config['REDIS_URL'])
    celery.conf.update(app.config)
else:
    # 在模拟模式下，创建一个简单的Celery实例而不连接到Redis
    print("\n⚠️ 警告：应用正在模拟模式下运行\n功能将受限，AI模型调用将在主线程中执行\n")
    # 创建一个模拟的Celery对象
    class MockCelery:
        def __init__(self, *args, **kwargs):
            pass
        
        def conf(self):
            return {}
        
        class task:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                
            def __call__(self, f):
                # 装饰器函数，返回原函数本身
                f.delay = f  # 模拟delay方法
                return f
    
    # 为MockCelery添加所需的方法
    celery = MockCelery()
    celery.task = MockCelery.task()
    celery.conf.update = lambda x: None
    celery.control = type('obj', (object,), {
        'ping': lambda *args, **kwargs: []
    })

# 活跃的客户端连接
active_connections = {}

# AI模型配置
AI_MODELS = {
    'doubao': {
        'api_key': os.getenv('DOUBAO_API_KEY', ''),
        'api_secret': os.getenv('DOUBAO_API_SECRET', ''),
        'base_url': 'https://api.doubao.com'
    },
    'deepseek': {
        'api_key': os.getenv('DEEPSEEK_API_KEY', ''),
        'base_url': 'https://api.deepseek.com'
    },
    'chatgpt': {
        'api_key': os.getenv('OPENAI_API_KEY', ''),
        'base_url': 'https://api.openai.com'
    },
    'kimi': {
        'api_key': os.getenv('KIMI_API_KEY', ''),
        'base_url': 'https://api.kimi.moonshot.cn'
    },
    'hunyuan': {
        'api_key': os.getenv('TENCENT_HUNYUAN_API_KEY', ''),
        'app_id': os.getenv('TENCENT_HUNYUAN_APP_ID', ''),
        'base_url': 'https://hunyuan.tencentcloudapi.com'
    },
    'gemini': {
        'api_key': os.getenv('GEMINI_API_KEY', ''),
        'base_url': 'https://generativelanguage.googleapis.com'
    }
}

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    client_id = request.args.get('client_id', request.sid)
    active_connections[client_id] = time.time()
    print(f'客户端 {client_id} 已连接')

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.args.get('client_id', request.sid)
    if client_id in active_connections:
        del active_connections[client_id]
        print(f'客户端 {client_id} 已断开连接')

# 发送WebSocket消息给所有客户端
def broadcast_message(data):
    socketio.emit('message', data, broadcast=True)

# 发送WebSocket消息给指定客户端
def send_message_to_client(client_id, data):
    if client_id and client_id in active_connections:
        socketio.emit('message', data, room=client_id)
    else:
        # 如果客户端ID无效，则广播消息
        broadcast_message(data)

# 异步任务：调用AI模型
@celery.task(bind=True)
def call_ai_model(self, model, question, other_results=None, client_id=None):
    try:
        logger.info(f'开始调用 {model} 模型，问题: {question[:50]}... 客户端: {client_id}')
        
        # 准备发送进度更新
        def send_progress(progress):
            data = {
                'model': model,
                'status': 'generating',
                'progress': progress
            }
            if client_id:
                send_message_to_client(client_id, data)
            else:
                broadcast_message(data)
        
        # 模拟进度更新
        send_progress(10)
        
        # 添加超时设置
        start_time = time.time()
        timeout = 60  # 60秒超时
        
        # 根据模型类型调用不同的API
        try:
            if model == 'doubao':
                result = call_doubao_api(question, other_results)
            elif model == 'deepseek':
                result = call_deepseek_api(question, other_results)
            elif model == 'chatgpt':
                result = call_chatgpt_api(question, other_results)
            elif model == 'kimi':
                result = call_kimi_api(question, other_results)
            elif model == 'hunyuan':
                result = call_hunyuan_api(question, other_results)
            elif model == 'gemini':
                result = call_gemini_api(question, other_results)
            else:
                raise ValueError(f'未知的模型类型: {model}')
            
            # 检查是否超时
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                raise TimeoutError(f'调用{model}模型超时（{elapsed_time:.2f}秒）')
            
        except Exception as api_error:
            logger.error(f'调用{model}模型API失败: {str(api_error)}')
            
            # 模拟进度更新
            send_progress(50)
            
            # 重试逻辑
            if self.request.retries < 2:  # 最多重试2次
                logger.info(f'准备重试调用{model}模型，当前重试次数: {self.request.retries}')
                # 指数退避策略
                countdown = min(30, 5 * (2 ** self.request.retries))  # 5s, 10s, 20s...最大30s
                raise self.retry(exc=api_error, countdown=countdown, max_retries=2)
            else:
                # 所有重试都失败
                raise Exception(f'所有重试都失败: {str(api_error)}')
        
        # 模拟进度更新
        send_progress(90)
        
        # 准备最终结果
        final_data = {
            'model': model,
            'status': 'completed',
            'content': result,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 发送最终结果
        if client_id:
            send_message_to_client(client_id, final_data)
        else:
            broadcast_message(final_data)
        
        logger.info(f'{model} 模型调用完成，用时: {time.time() - start_time:.2f}秒')
        return final_data
        
    except Exception as e:
        logger.error(f'调用 {model} 模型失败: {str(e)}', exc_info=True)
        # 发送错误信息
        error_data = {
            'model': model,
            'status': 'error',
            'content': f'调用失败: {str(e)}',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        if client_id:
            send_message_to_client(client_id, error_data)
        else:
            broadcast_message(error_data)
        # 在模拟模式下不抛出异常，避免重试
        if not SIMULATION_MODE:
            raise self.retry(exc=e, countdown=5, max_retries=1)

# 模拟模式下的备用请求处理函数
def handle_simulation_request(client_id, question, other_results=None):
    """
    模拟模式下的备用请求处理函数，当Celery不可用时使用
    
    Args:
        client_id: 客户端ID
        question: 用户问题
        other_results: 其他模型的结果（可选）
    """
    try:
        logger.info(f'使用备用方式处理请求，客户端ID: {client_id}')
        
        # 模拟异步处理，为每个模型创建一个线程
        threads = []
        
        for model in AI_MODELS.keys():
            # 为每个模型创建一个线程
            thread = threading.Thread(
                target=simulate_model_response,
                args=(model, question, other_results, client_id)
            )
            threads.append(thread)
            thread.start()
            
            # 添加随机延迟，模拟不同模型的响应时间差异
            time.sleep(random.uniform(0.1, 0.3))
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
            
        logger.info(f'备用处理完成，客户端ID: {client_id}')
        
    except Exception as e:
        logger.error(f'备用处理失败: {str(e)}', exc_info=True)
        # 发送错误消息给客户端
        error_data = {
            'status': 'error',
            'content': f'处理请求时发生错误: {str(e)}',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        send_message_to_client(client_id, error_data)

# 模拟单个模型的响应
def simulate_model_response(model, question, other_results=None, client_id=None):
    """
    模拟单个模型的响应过程
    
    Args:
        model: 模型名称
        question: 用户问题
        other_results: 其他模型的结果（可选）
        client_id: 客户端ID
    """
    try:
        logger.debug(f'开始模拟{model}模型响应，客户端ID: {client_id}')
        
        # 发送进度更新
        def send_progress(progress):
            data = {
                'model': model,
                'status': 'generating',
                'progress': progress
            }
            send_message_to_client(client_id, data)
        
        # 模拟进度更新
        send_progress(random.randint(5, 15))
        
        # 模拟处理延迟
        processing_time = random.uniform(1, 4)
        time.sleep(processing_time * 0.3)  # 先睡一部分时间
        
        send_progress(random.randint(30, 50))
        time.sleep(processing_time * 0.5)  # 再睡一部分时间
        
        send_progress(random.randint(70, 85))
        
        # 调用模型API获取结果
        result = call_ai_model_api(model, question, other_results)
        
        # 准备最终结果
        final_data = {
            'model': model,
            'status': 'completed',
            'content': result,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 发送最终结果
        send_message_to_client(client_id, final_data)
        
        logger.debug(f'{model}模型响应模拟完成，客户端ID: {client_id}')
        
    except Exception as e:
        logger.error(f'模拟{model}模型响应失败: {str(e)}')
        # 发送错误信息
        error_data = {
            'model': model,
            'status': 'error',
            'content': f'模拟响应失败: {str(e)}',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        send_message_to_client(client_id, error_data)

# 通用AI模型调用函数
def call_ai_model_api(model, question, other_results=None):
    """
    通用AI模型调用函数，减少重复代码
    
    Args:
        model: 模型名称
        question: 用户问题
        other_results: 其他模型的结果（可选）
    
    Returns:
        str: AI模型的回答
    """
    try:
        # 检查模型配置
        if model not in AI_MODELS:
            return f"未知的模型类型: {model}"
        
        config = AI_MODELS[model]
        if not config['api_key']:
            # 根据模型名称生成提示信息
            model_names = {
                'doubao': '豆包',
                'deepseek': 'Deepseek',
                'chatgpt': 'ChatGPT',
                'kimi': 'Kimi',
                'hunyuan': '腾讯混元',
                'gemini': 'Gemini'
            }
            env_vars = {
                'doubao': 'DOUBAO_API_KEY',
                'deepseek': 'DEEPSEEK_API_KEY',
                'chatgpt': 'OPENAI_API_KEY',
                'kimi': 'KIMI_API_KEY',
                'hunyuan': 'TENCENT_HUNYUAN_API_KEY',
                'gemini': 'GEMINI_API_KEY'
            }
            return f"{model_names.get(model, model)} API Key未配置，请在.env文件中设置{env_vars.get(model, '')}"
        
        # 模拟API调用延迟（根据模型不同设置不同的延迟时间）
        delays = {
            'doubao': 2,
            'deepseek': 2.5,
            'chatgpt': 3,
            'kimi': 2.2,
            'hunyuan': 2.8,
            'gemini': 3.2
        }
        
        # 添加随机因素，使模拟更真实
        import random
        delay_factor = random.uniform(0.8, 1.2)  # 80%-120%的随机延迟因子
        actual_delay = delays.get(model, 2) * delay_factor
        
        # 在模拟模式下，模拟不同的网络状况
        if SIMULATION_MODE:
            # 10%的概率模拟请求超时
            if random.random() < 0.1:
                time.sleep(actual_delay)
                raise Exception("请求超时，请稍后重试")
            
            # 5%的概率模拟API错误
            elif random.random() < 0.05:
                time.sleep(actual_delay * 0.3)
                raise Exception("API调用失败，服务暂时不可用")
                
        time.sleep(actual_delay)
        
        # 构建包含其他模型结果的提示
        prompt = f"用户问题: {question}\n"
        if other_results and random.random() > 0.3:  # 70%的概率会考虑其他模型的结果
            prompt += "\n其他AI模型的回答参考:\n"
            for model_name, content in other_results.items():
                prompt += f"\n{model_name}: {content[:100]}...\n"
        
        # 根据模型名称生成回答模板
        model_greetings = {
            'doubao': '这是对问题',
            'deepseek': '针对您提出的问题',
            'chatgpt': '您好！关于您的问题',
            'kimi': '感谢您的提问！对于',
            'hunyuan': '您好！针对您提出的',
            'gemini': '对于您的问题'
        }
        
        model_names = {
            'doubao': '豆包',
            'deepseek': 'Deepseek',
            'chatgpt': 'ChatGPT',
            'kimi': 'Kimi',
            'hunyuan': '腾讯混元',
            'gemini': 'Gemini'
        }
        
        model_descriptions = {
            'doubao': '的详细解答。',
            'deepseek': '，我进行了深入分析。',
            'chatgpt': '，我的理解和建议如下：',
            'kimi': '这个问题，我可以为您提供以下信息：',
            'hunyuan': '问题，我为您整理了以下内容：',
            'gemini': '，我进行了分析和思考：'
        }
        
        # 模拟不同模型的回答风格差异
        model_responses = {
            'doubao': [
                "根据我的分析，这个问题可以从多个角度来理解。首先，我们需要考虑...\n\n其次，还应该注意...",
                "这个问题很有深度！让我来梳理一下关键点：...\n\n总结来说，...",
                "我认为这个问题可以这样解决：首先...然后...最后..."
            ],
            'deepseek': [
                "经过深入分析，我发现这个问题涉及以下几个核心方面：...\n\n基于上述分析，我的建议是...",
                "从技术角度来看，这个问题的本质是...\n\n为了解决这个问题，我们可以采取以下措施：...",
                "通过对问题的全面评估，我认为最有效的解决方案是..."
            ],
            'chatgpt': [
                "感谢您的问题！关于这个话题，我有以下几点看法：...\n\n希望这些信息对您有所帮助！",
                "很高兴为您解答！这个问题需要我们从...几个方面来考虑。\n\n总的来说，...",
                "您提出了一个很好的问题。让我们逐步分析：...\n\n基于以上分析，我的结论是..."
            ],
            'kimi': [
                "我将为您详细解答这个问题。首先，让我们明确问题的核心...\n\n接下来，我会提供几种可能的解决方案...",
                "非常感谢您的提问！针对这个问题，我进行了全面的研究和分析...\n\n我的建议是...",
                "这个问题涉及到多个方面，我将逐一为您解析：...\n\n综上所述，..."
            ],
            'hunyuan': [
                "针对您的问题，我整理了以下关键信息和建议：...\n\n希望这些内容能满足您的需求！",
                "您好！经过仔细研究，我为您准备了以下回答：...\n\n如有其他问题，欢迎继续提问！",
                "感谢您的信任！关于这个问题，我的看法是：...\n\n如果您需要更深入的探讨，请随时告诉我。"
            ],
            'gemini': [
                "对于您的问题，我进行了全面的思考和分析。以下是我的见解：...\n\n希望这些信息对您有所启发！",
                "我认为这个问题值得深入探讨。首先，让我们理解问题的背景和上下文...\n\n基于上述分析，我提出以下观点：...",
                "感谢您的提问！这是一个很有挑战性的话题。我的思考过程如下：...\n\n总结来说，..."
            ]
        }
        
        # 随机选择一个回答模板
        responses = model_responses.get(model, ["这是一个示例回答。"])  # 默认回答
        random_response = random.choice(responses)
        
        # 构建最终回答
        final_response = f"{model_names.get(model, model)}的回答: {model_greetings.get(model, '')} '{question[:30]}...'{model_descriptions.get(model, '')}\n\n{random_response}"
        
        if SIMULATION_MODE:
            final_response += "\n\n系统提示: 当前使用的是模拟模式，此回答为模拟内容。"
        
        return final_response
    except Exception as e:
        logger.error(f"调用{model}模型时出错: {str(e)}")
        return f"调用{model}模型失败: {str(e)}"

# 各模型的调用函数（为了保持向后兼容）
def call_doubao_api(question, other_results=None):
    return call_ai_model_api('doubao', question, other_results)

def call_deepseek_api(question, other_results=None):
    return call_ai_model_api('deepseek', question, other_results)

def call_chatgpt_api(question, other_results=None):
    return call_ai_model_api('chatgpt', question, other_results)

def call_kimi_api(question, other_results=None):
    return call_ai_model_api('kimi', question, other_results)

def call_hunyuan_api(question, other_results=None):
    return call_ai_model_api('hunyuan', question, other_results)

def call_gemini_api(question, other_results=None):
    return call_ai_model_api('gemini', question, other_results)

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
        
        # 异步调用所有AI模型
        try:
            # 并行提交所有模型的任务
            task_ids = {}
            for model in AI_MODELS.keys():
                try:
                    task = call_ai_model.delay(model, question, other_results, client_id)
                    task_ids[model] = task.id
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
            
            # 检查是否有成功提交的任务
            if not task_ids:
                # 如果全部失败且处于模拟模式，使用备用处理方式
                if SIMULATION_MODE:
                    logger.info('所有模型任务提交失败，切换到备用处理方式')
                    # 创建一个后台线程来处理请求
                    threading.Thread(target=handle_simulation_request, args=(client_id, question, other_results)).start()
                    logger.info(f'已使用备用方式处理问题，客户端ID: {client_id}')
                    return jsonify({
                        'success': True,
                        'message': '问题已通过备用方式提交给所有AI模型',
                        'client_id': client_id,
                        'models_count': len(AI_MODELS),
                        'using_fallback': True
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
                # 创建一个后台线程来处理请求
                threading.Thread(target=handle_simulation_request, args=(client_id, question, other_results)).start()
                logger.info(f'已使用备用方式处理问题，客户端ID: {client_id}')
                return jsonify({
                    'success': True,
                    'message': 'Redis不可用，问题已通过备用方式提交给所有AI模型',
                    'client_id': client_id,
                    'models_count': len(AI_MODELS),
                    'using_fallback': True
                })
            else:
                return jsonify({
                    'success': False,
                    'message': '提交模型请求失败，请稍后重试',
                    'error_type': str(type(model_error).__name__)
                }), 503  # 服务不可用
        
        logger.info(f'已提交问题给所有 {len(AI_MODELS)} 个AI模型')
        return jsonify({
            'success': True,
            'message': '问题已提交给所有AI模型',
            'client_id': client_id,
            'models_count': len(AI_MODELS),
            'task_ids': task_ids
        })
        
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
        
        # 异步调用指定AI模型
        try:
            task = call_ai_model.delay(model, question, other_results, client_id)
            logger.info(f'已请求重新生成{model}的回答')
            return jsonify({
                'success': True, 
                'message': f'已请求重新生成{model}的回答',
                'client_id': client_id,
                'task_id': task.id
            })
        except Exception as e:
            logger.error(f'提交{model}模型任务失败: {str(e)}')
            # 如果提交任务失败且处于模拟模式或Redis连接失败，使用备用处理方式
            if SIMULATION_MODE or 'Redis' in str(e) or 'Connection' in str(e):
                logger.info('Redis不可用或Celery连接失败，使用备用方式处理请求')
                # 创建一个后台线程来处理请求
                threading.Thread(
                    target=simulate_model_response,
                    args=(model, question, other_results, client_id)
                ).start()
                return jsonify({
                    'success': True,
                    'message': f'已通过备用方式重新生成{model}模型的回答',
                    'client_id': client_id,
                    'using_fallback': True
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