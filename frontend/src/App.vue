<template>
  <div class="app-container">
    <header class="app-header">
      <h1>多AI聚合应用</h1>
      <el-button type="danger" @click="handleLogout">退出</el-button>
    </header>
    
    <main class="app-main">
      <!-- 用户问题输入区域 -->
      <div class="input-section">
        <el-input
          v-model="userInput"
          type="textarea"
          placeholder="请输入您的问题..."
          :rows="4"
          :autosize="{ minRows: 4, maxRows: 10 }"
          clearable
        />
        <div class="input-actions">
          <el-button type="primary" @click="submitQuestion">提交问题</el-button>
        </div>
      </div>
      
      <!-- AI模型结果展示区域 -->
      <div class="ai-results">
        <div class="ai-models-grid">
          <AI-Model-Card
            v-for="(model, key) in modelConfigs"
            :key="key"
            :model-name="model.name"
            :model-key="key"
            :result="results[key]"
            :is-loading="loading[key]"
            @regenerate="regenerateResponse"
            @delete="deleteResponse"
          />
        </div>
      </div>
    </main>
    
    <footer class="app-footer">
      <p>多AI聚合应用 © 2025</p>
    </footer>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import axios from 'axios';
import { io } from 'socket.io-client';
import AIModelCard from './components/AIModelCard.vue';

export default {
  name: 'App',
  components: {
    AIModelCard
  },
  setup() {
    // 用户输入
    const userInput = ref('');
    
    // 模型配置
    const modelConfigs = ref({
      doubao: { name: '豆包' },
      deepseek: { name: 'Deepseek' },
      chatgpt: { name: 'ChatGPT' },
      kimi: { name: 'Kimi' },
      hunyuan: { name: '腾讯混元' },
      gemini: { name: 'Gemini' }
    });
    
    // AI模型结果
    const results = ref({
      doubao: null,
      deepseek: null,
      chatgpt: null,
      kimi: null,
      hunyuan: null,
      gemini: null
    });
    
    // 加载状态
    const loading = ref({
      doubao: false,
      deepseek: false,
      chatgpt: false,
      kimi: false,
      hunyuan: false,
      gemini: false
    });
    
    // 客户端唯一ID
    const clientId = ref(generateClientId());
    
    // 生成客户端唯一ID
    function generateClientId() {
      return 'client_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
    }
    
    // Socket.IO连接
    let socket = null;
    

    
    // 连接Socket.IO
    const connectSocketIO = () => {
      try {
        // 连接Socket.IO服务器
        socket = io('http://localhost:5000', {
          query: { client_id: clientId.value },
          transports: ['websocket'],
          reconnection: true,
          reconnectionAttempts: 10,  // 增加重试次数
          reconnectionDelay: 2000,   // 增加重试间隔
          reconnectionDelayMax: 10000, // 最大重试间隔
          randomizationFactor: 0.5   // 添加随机因子，避免所有客户端同时重试
        });
        
        socket.on('connect', () => {
          console.log('Socket.IO连接已建立，客户端ID:', clientId.value);
          ElMessage.success('已连接到服务器');
        });
        
        socket.on('message', (data) => {
          try {
            if (data.model && data.content !== undefined) {
              // 更新对应模型的结果
              if (!results.value[data.model]) {
                results.value[data.model] = {};
              }
              
              if (data.status === 'generating') {
                results.value[data.model].status = 'generating';
                results.value[data.model].progress = data.progress || 0;
              } else if (data.status === 'completed') {
                results.value[data.model].status = 'completed';
                results.value[data.model].content = data.content;
                results.value[data.model].timestamp = new Date().toLocaleString();
              } else if (data.status === 'error') {
                results.value[data.model].status = 'error';
                results.value[data.model].content = data.content || '处理请求时发生错误';
                results.value[data.model].timestamp = new Date().toLocaleString();
                ElMessage.error(`${data.model} 模型处理失败: ${data.content}`);
              }
              
              // 响应式更新
              results.value = { ...results.value };
            }
          } catch (error) {
            console.error('解析Socket.IO消息失败:', error);
            ElMessage.error('处理服务器消息时发生错误');
          }
        });
        
        socket.on('disconnect', (reason) => {
          console.log('Socket.IO连接已关闭，原因:', reason);
          ElMessage.info('服务器连接已断开');
        });
        
        socket.on('connect_error', (error) => {
          console.error('Socket.IO连接错误:', error);
          // 只在初始连接失败时显示错误消息，避免重试时频繁提示
          if (socket && socket.reconnectAttempts === 0) {
            ElMessage.error('连接服务器失败，正在尝试重新连接...');
          }
        });
        
        socket.on('reconnect_attempt', (attemptNumber) => {
          console.log(`尝试重新连接... (第${attemptNumber}次)`);
        });
        
        socket.on('reconnect', (attemptNumber) => {
          console.log(`重新连接成功 (第${attemptNumber}次尝试)`);
          ElMessage.success('已重新连接到服务器');
        });
        
        socket.on('reconnect_error', (error) => {
          console.error('重新连接错误:', error);
        });
        
        socket.on('reconnect_failed', () => {
          console.error('所有重新连接尝试都失败了');
          ElMessage.error('无法重新连接到服务器，请刷新页面重试');
          // 5秒后自动尝试重新连接
          setTimeout(() => {
            console.log('尝试重新初始化连接...');
            if (socket) {
              socket.connect();
            } else {
              connectSocketIO();
            }
          }, 5000);
        });
      } catch (error) {
        console.error('创建Socket.IO连接失败:', error);
        ElMessage.error('初始化连接失败');
        // 5秒后重试
        setTimeout(() => {
          console.log('尝试重新创建Socket.IO连接...');
          connectSocketIO();
        }, 5000);
      }
    };
    
    // 提交问题
    const submitQuestion = async () => {
      if (!userInput.value.trim()) {
        ElMessage.warning('请输入问题内容');
        return;
      }
      
      try {
        // 获取其他模型的最新结果
        const otherResults = {};
        Object.keys(results.value).forEach(key => {
          if (results.value[key] && results.value[key].content) {
            otherResults[key] = results.value[key].content;
          }
        });
        
        // 提交问题到后端
        const response = await axios.post('http://localhost:5000/api/ask', {
          question: userInput.value,
          other_results: otherResults
        }, {
          headers: {
            'X-Client-ID': clientId.value
          }
        });
        
        if (response.data.success) {
          ElMessage.success('问题已提交');
          // 重置所有结果状态
          Object.keys(results.value).forEach(key => {
            results.value[key] = {
              status: 'generating',
              progress: 0
            };
            loading.value[key] = false;
          });
          results.value = { ...results.value };
        } else {
          ElMessage.error('提交失败，请重试');
        }
      } catch (error) {
        console.error('提交问题失败:', error);
        ElMessage.error('提交失败，请检查网络连接或稍后重试');
      }
    };
    
    // 重新生成指定模型的响应
    const regenerateResponse = async (model) => {
      if (!userInput.value.trim()) {
        ElMessage.warning('请先输入问题内容');
        return;
      }
      
      loading.value[model] = true;
      
      try {
        // 获取其他模型的最新结果
        const otherResults = {};
        Object.keys(results.value).forEach(key => {
          if (key !== model && results.value[key] && results.value[key].content) {
            otherResults[key] = results.value[key].content;
          }
        });
        
        // 请求重新生成
        const response = await axios.post(`http://localhost:5000/api/regenerate`, {
          model: model,
          question: userInput.value,
          other_results: otherResults
        }, {
          headers: {
            'X-Client-ID': clientId.value
          },
          timeout: 30000 // 设置30秒超时
        });
        
        if (response.data.success) {
          // 设置模型为生成中状态
          results.value[model] = {
            status: 'generating',
            progress: 0
          };
          results.value = { ...results.value };
        } else {
          ElMessage.error('重新生成失败，请重试');
        }
      } catch (error) {
        console.error(`重新生成${model}响应失败:`, error);
        ElMessage.error('重新生成失败，请检查网络连接或稍后重试');
      } finally {
        loading.value[model] = false;
      }
    };
    
    // 删除指定模型的响应
    const deleteResponse = (model) => {
      delete results.value[model];
      results.value = { ...results.value };
      ElMessage.success('内容已删除');
    };
    
    // 退出处理
    const handleLogout = () => {
      ElMessageBox.confirm(
        '确定要退出吗？',
        '提示',
        {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        }
      ).then(() => {
        // 退出逻辑
        ElMessage.success('已退出');
        // 在实际应用中，这里可以清除token、用户信息等
      }).catch(() => {
        // 取消退出
      });
    };
    
    // 组件挂载时连接WebSocket
    onMounted(() => {
      connectSocketIO();
    });
    
    // 组件卸载时关闭WebSocket连接
    onUnmounted(() => {
      if (socket) {
        socket.disconnect();
      }
    });
    
    return {
      userInput,
      results,
      loading,
      clientId,
      getHeader,
      formatContent,
      submitQuestion,
      regenerateResponse,
      deleteResponse,
      handleLogout
    };
  }
};
</script>

<style scoped>
.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  padding: 20px;
  background-color: #fff;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.app-header h1 {
  margin: 0;
  color: #1890ff;
  font-size: 24px;
}

.app-main {
  flex: 1;
  padding: 20px;
  background-color: #f5f5f5;
  overflow-y: auto;
}

.input-section {
  background-color: #fff;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.input-actions {
  margin-top: 10px;
  display: flex;
  justify-content: flex-end;
}

.ai-results {
  margin-top: 20px;
}

.ai-models-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
}

.ai-card {
  transition: all 0.3s ease;
}

.ai-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.ai-content {
  min-height: 200px;
  max-height: 400px;
  overflow-y: auto;
  padding: 10px 0;
}

.ai-actions {
  margin-top: 15px;
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.empty-state {
  min-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #999;
}

.app-footer {
  padding: 20px;
  background-color: #fff;
  text-align: center;
  color: #666;
  border-top: 1px solid #e8e8e8;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .ai-models-grid {
    grid-template-columns: 1fr;
  }
  
  .app-header {
    flex-direction: column;
    gap: 10px;
  }
}
</style>