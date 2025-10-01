#!/usr/bin/env python
import os
import sys

# 设置工作目录为项目根目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 启动Redis提示
print('''\n注意：在运行此应用前，请确保Redis服务器已启动。\n\nWindows系统安装Redis方法:\n1. 下载Redis安装包: https://github.com/microsoftarchive/redis/releases\n2. 安装并启动Redis服务\n\nLinux/Mac系统安装Redis方法:\nsudo apt-get install redis-server  # Ubuntu/Debian\nbrew install redis                  # macOS\n然后启动Redis服务: redis-server\n\n如果您暂时没有安装Redis，可以使用模拟模式（功能会受限）。\n''')

# 检查Redis是否可用
def check_redis_available():
    try:
        import redis
        r = redis.Redis()
        r.ping()
        return True
    except Exception:
        return False

redis_available = check_redis_available()
use_simulation = False

if not redis_available:
    print('\n检测到Redis服务不可用。')
    print('自动使用模拟模式启动服务...')
    use_simulation = True
    # 设置环境变量，让app.py知道当前是模拟模式
    os.environ['SIMULATION_MODE'] = 'true'
else:
    print('Redis服务可用。')

# 直接导入并运行Flask应用
print('\n正在启动Flask应用...')
import backend.app

# 如果应用程序没有在导入时自动运行，手动运行它
if hasattr(backend.app, 'app') and not backend.app.app.debug:
    print('\n服务启动成功！\n\n当前运行的服务：\n1. Flask Web服务器\n\n使用指南：\n- API地址: http://localhost:5000/api\n\n按 Ctrl+C 停止服务。\n')
    backend.app.app.run(host='0.0.0.0', port=5000, debug=True)