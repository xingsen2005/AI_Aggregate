#!/usr/bin/env python
import os
import sys
import subprocess
import time
import signal
import platform

# 检查系统类型
is_windows = platform.system() == 'Windows'

# 定义命令
flask_command = []
celery_command = []

if is_windows:
    # Windows系统使用cmd命令
    flask_command = ['cmd', '/c', 'set FLASK_APP=app.py && flask run --host=0.0.0.0 --port=5000']
    celery_command = ['cmd', '/c', 'celery -A app.celery worker --loglevel=info -P eventlet']
else:
    # Unix/Linux/Mac系统使用bash命令
    flask_command = ['bash', '-c', 'export FLASK_APP=app.py && flask run --host=0.0.0.0 --port=5000']
    celery_command = ['bash', '-c', 'celery -A app.celery worker --loglevel=info']

# 存储进程对象
processes = []

# 处理信号，用于优雅退出
def signal_handler(sig, frame):
    print('\n正在停止所有服务...')
    for p in processes:
        try:
            if is_windows:
                p.kill()
            else:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception as e:
            print(f'停止进程时出错: {e}')
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
    use_simulation = input('是否使用模拟模式启动服务？(y/n): ').lower() == 'y'
    if not use_simulation:
        print('已取消启动。')
        sys.exit(0)
else:
    # 询问用户是否继续
    if input('是否继续启动服务？(y/n): ').lower() != 'y':
        print('已取消启动。')
        sys.exit(0)

if not use_simulation:
    # 启动Celery worker
    print('\n正在启动Celery worker...')
    celery_env = os.environ.copy()
    if is_windows:
        # 在Windows上使用shell=True来支持命令字符串
        celery_proc = subprocess.Popen(celery_command, shell=True, env=celery_env)
    else:
        # 在Unix系统上，使用start_new_session=True来创建新的进程组
        celery_proc = subprocess.Popen(celery_command, shell=True, preexec_fn=os.setsid, env=celery_env)
    processes.append(celery_proc)

    # 等待Celery启动
    print('等待Celery worker初始化...')
    time.sleep(3)
else:
    print('\n使用模拟模式启动服务，将跳过Celery worker的启动...')
    # 设置环境变量，让app.py知道当前是模拟模式
    os.environ['SIMULATION_MODE'] = 'true'

# 启动Flask应用
print('\n正在启动Flask应用...')
flask_env = os.environ.copy()
if is_windows:
    flask_proc = subprocess.Popen(flask_command, shell=True, env=flask_env)
else:
    flask_proc = subprocess.Popen(flask_command, shell=True, preexec_fn=os.setsid, env=flask_env)
processes.append(flask_proc)

# 启动成功提示
print('''\n服务启动成功！\n\n当前运行的服务：\n1. Flask Web服务器 (端口: 5000)\n2. Celery Worker (异步任务处理)\n\n使用指南：\n- 前端页面: 请在浏览器中打开 frontend/index.html\n- API地址: http://localhost:5000/api\n- WebSocket地址: ws://localhost:5000/ws\n\n按 Ctrl+C 停止所有服务。\n''')

# 保持主进程运行
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    signal_handler(signal.SIGINT, None)