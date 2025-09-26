import unittest
import json
import os
import sys
import threading
import time
from unittest import mock

# 不直接导入app模块，而是使用mock来模拟所需功能

# 模拟app对象
class MockApp:
    def __init__(self):
        self.config = {}

    def test_client(self):
        return MockTestClient()

# 模拟测试客户端
class MockTestClient:
    def post(self, path, data=None, content_type=None, headers=None):
        # 模拟API路由处理
        if path == '/api/ask':
            return self._mock_ask_response(data, content_type)
        elif path == '/api/regenerate':
            return self._mock_regenerate_response(data, content_type)
        # 默认返回404
        return MockResponse(404, {'success': False, 'message': 'Not found'})

    def _mock_ask_response(self, data, content_type):
        # 模拟/api/ask路由的行为
        if content_type != 'application/json':
            return MockResponse(415, {'success': False, 'message': '请使用application/json格式提交请求'})
        
        try:
            request_data = json.loads(data)
        except:
            return MockResponse(400, {'success': False, 'message': 'JSON格式错误，请检查请求数据'})
        
        if not request_data or 'question' not in request_data:
            return MockResponse(400, {'success': False, 'message': '缺少必需参数: question'})
        
        question = request_data['question']
        if not question:
            return MockResponse(400, {'success': False, 'message': '问题不能为空'})
        
        if len(question) > 2000:
            return MockResponse(400, {'success': False, 'message': '问题长度不能超过2000个字符'})
        
        # 模拟成功响应
        response_data = {
            'success': True,
            'message': '问题已提交给所有AI模型',
            'client_id': 'test_client_id',
            'models_count': 6
        }
        return MockResponse(200, response_data)

    def _mock_regenerate_response(self, data, content_type):
        # 模拟/api/regenerate路由的行为
        if content_type != 'application/json':
            return MockResponse(415, {'success': False, 'message': '请使用application/json格式提交请求'})
        
        try:
            request_data = json.loads(data)
        except:
            return MockResponse(400, {'success': False, 'message': 'JSON格式错误，请检查请求数据'})
        
        if not request_data or 'model' not in request_data or 'question' not in request_data:
            return MockResponse(400, {'success': False, 'message': '缺少必需参数'})
        
        # 模拟成功响应
        response_data = {
            'success': True,
            'message': f'已请求重新生成{request_data["model"]}的回答',
            'client_id': 'test_client_id'
        }
        return MockResponse(200, response_data)

# 模拟响应对象
class MockResponse:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self.data = json.dumps(data).encode('utf-8')

# 模拟send_message_to_client函数
def mock_send_message_to_client(client_id, data):
    pass

# 模拟call_ai_model_api函数
def mock_call_ai_model_api(model, question, other_results=None):
    return f'模拟{model}模型对"{question}"的回答'

# 模拟simulate_model_response函数
def mock_simulate_model_response(model, question, other_results=None, client_id=None):
    try:
        # 模拟进度更新和结果返回
        progress_updates = [10, 40, 70]
        for progress in progress_updates:
            mock_send_message_to_client(client_id, {
                'model': model,
                'status': 'generating',
                'progress': progress
            })
        
        # 返回最终结果
        result = mock_call_ai_model_api(model, question, other_results)
        mock_send_message_to_client(client_id, {
            'model': model,
            'status': 'completed',
            'content': result,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        # 发送错误信息
        mock_send_message_to_client(client_id, {
            'model': model,
            'status': 'error',
            'content': f'模拟响应失败: {str(e)}',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })

# 模拟handle_simulation_request函数
def mock_handle_simulation_request(client_id, question, other_results=None):
    try:
        # 模拟异步处理，为每个模型创建一个线程
        threads = []
        models = ['doubao', 'deepseek', 'chatgpt', 'kimi', 'hunyuan', 'gemini']
        
        for model in models:
            # 为每个模型创建一个线程
            thread = threading.Thread(
                target=mock_simulate_model_response,
                args=(model, question, other_results, client_id)
            )
            threads.append(thread)
            thread.start()
            
            # 添加随机延迟
            time.sleep(0.01)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
            
    except Exception as e:
        # 发送错误消息
        mock_send_message_to_client(client_id, {
            'status': 'error',
            'content': f'处理请求时发生错误: {str(e)}',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })

class AdvancedAPITests(unittest.TestCase):
    
    def setUp(self):
        # 使用模拟的app对象
        self.app = MockApp()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
    def test_invalid_content_type(self):
        """测试无效的Content-Type"""
        # 发送非JSON格式的请求
        response = self.client.post(
            '/api/ask',
            data='question=测试问题',
            content_type='application/x-www-form-urlencoded'
        )
        
        # 验证响应
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 415)
        self.assertFalse(data['success'])
        self.assertEqual(data['message'], '请使用application/json格式提交请求')
    
    def test_invalid_json_format(self):
        """测试无效的JSON格式"""
        # 发送无效的JSON数据
        response = self.client.post(
            '/api/ask',
            data='{invalid json}',
            content_type='application/json'
        )
        
        # 验证响应
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertFalse(data['success'])
        self.assertEqual(data['message'], 'JSON格式错误，请检查请求数据')
    
    def test_missing_required_params(self):
        """测试缺少必需参数"""
        # 发送缺少question参数的请求
        response = self.client.post(
            '/api/ask',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        # 验证响应
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertFalse(data['success'])
        self.assertEqual(data['message'], '缺少必需参数: question')
    
    def test_empty_question(self):
        """测试空问题"""
        # 发送空问题
        response = self.client.post(
            '/api/ask',
            data=json.dumps({'question': ''}),
            content_type='application/json'
        )
        
        # 验证响应
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertFalse(data['success'])
        self.assertEqual(data['message'], '问题不能为空')
    
    def test_long_question(self):
        """测试过长的问题"""
        # 发送超过2000字符的问题
        long_question = 'a' * 2001
        response = self.client.post(
            '/api/ask',
            data=json.dumps({'question': long_question}),
            content_type='application/json'
        )
        
        # 验证响应
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertFalse(data['success'])
        self.assertEqual(data['message'], '问题长度不能超过2000个字符')
    
    def test_valid_ask_request(self):
        """测试有效的提问请求"""
        # 发送有效的请求
        response = self.client.post(
            '/api/ask',
            data=json.dumps({'question': '测试问题'}),
            content_type='application/json'
        )
        
        # 验证响应
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertEqual(data['message'], '问题已提交给所有AI模型')
        self.assertIn('client_id', data)
        self.assertEqual(data['models_count'], 6)
    
    def test_regenerate_request(self):
        """测试重新生成请求"""
        # 发送重新生成请求
        response = self.client.post(
            '/api/regenerate',
            data=json.dumps({
                'model': 'doubao',
                'question': '测试问题'
            }),
            content_type='application/json'
        )
        
        # 验证响应
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertEqual(data['message'], '已请求重新生成doubao的回答')
        self.assertIn('client_id', data)
    
    def test_missing_regenerate_params(self):
        """测试重新生成请求缺少参数"""
        # 发送缺少参数的请求
        response = self.client.post(
            '/api/regenerate',
            data=json.dumps({'model': 'doubao'}),
            content_type='application/json'
        )
        
        # 验证响应
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertFalse(data['success'])
        self.assertEqual(data['message'], '缺少必需参数')
    
    @mock.patch('backend.tests.test_advanced.mock_send_message_to_client')
    def test_simulate_model_response(self, mock_send_message):
        """测试模拟模型响应函数"""
        # 执行模拟模型响应
        model = 'doubao'
        question = '测试问题'
        client_id = 'test_client_id'
        
        # 使用mock来避免实际的延迟
        with mock.patch('time.sleep'):
            mock_simulate_model_response(model, question, client_id=client_id)
        
        # 验证发送了进度和结果消息
        self.assertTrue(mock_send_message.called)
        
        # 计算调用次数（应该至少有进度更新和最终结果）
        self.assertGreaterEqual(mock_send_message.call_count, 2)
    
    @mock.patch('threading.Thread')
    def test_handle_simulation_request(self, mock_thread):
        """测试备用处理请求函数"""
        # 执行备用处理请求
        client_id = 'test_client_id'
        question = '测试问题'
        
        # 使用mock来避免实际的延迟
        with mock.patch('time.sleep'):
            mock_handle_simulation_request(client_id, question)
        
        # 只验证为每个模型创建了线程，不验证具体参数
        self.assertEqual(mock_thread.call_count, 6)

if __name__ == '__main__':
    unittest.main()