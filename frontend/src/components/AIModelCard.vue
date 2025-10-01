<template>
  <el-card class="ai-card" :header="getHeader()">
    <div class="ai-content" v-if="result">
      <div v-html="formatContent(result.content)"></div>
      <div class="ai-actions">
        <el-button
          size="small"
          type="primary"
          @click="regenerateResponse"
          :loading="isLoading"
        >
          <el-icon><Refresh /></el-icon>
          重新生成
        </el-button>
        <el-button
          size="small"
          @click="deleteResponse"
        >
          <el-icon><Delete /></el-icon>
          删除内容
        </el-button>
      </div>
    </div>
    <div v-else class="empty-state">
      暂无内容
    </div>
  </el-card>
</template>

<script>
import { Refresh, Delete } from '@element-plus/icons-vue';
import { ElMessage } from 'element-plus';
import { ref, computed } from 'vue';

export default {
  name: 'AIModelCard',
  components: {
    Refresh,
    Delete
  },
  props: {
    modelName: {
      type: String,
      required: true
    },
    modelKey: {
      type: String,
      required: true
    },
    result: {
      type: Object,
      default: null
    },
    isLoading: {
      type: Boolean,
      default: false
    }
  },
  emits: ['regenerate', 'delete'],
  methods: {
    getHeader() {
      if (this.result && this.result.status === 'generating') {
        return this.modelName + ' (生成中...)';
      }
      return this.modelName;
    },
    formatContent(content) {
      if (!content) return '';
      
      // 加强的XSS防护
      const escapeHtml = (unsafe) => {
        return unsafe
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#039;')
          .replace(/\//g, '&#x2F;')
          .replace(/\(/g, '&#40;')
          .replace(/\)/g, '&#41;')
          .replace(/\+/g, '&#43;')
          .replace(/\=/g, '&#61;');
      };
      
      // 先进行HTML转义，再进行Markdown解析
      const escapedContent = escapeHtml(content);
      
      // 简单的Markdown解析，实际应用中可以使用更完整的Markdown解析库
      return escapedContent
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    },
    regenerateResponse() {
      this.$emit('regenerate', this.modelKey);
    },
    deleteResponse() {
      this.$emit('delete', this.modelKey);
    }
  }
};
</script>

<style scoped>
.ai-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.ai-content {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.ai-actions {
  margin-top: 10px;
  display: flex;
  justify-content: flex-end;
  gap: 5px;
}

.empty-state {
  text-align: center;
  color: #999;
  padding: 20px 0;
}
</style>