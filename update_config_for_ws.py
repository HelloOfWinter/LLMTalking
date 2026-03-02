# update_config_for_ws.py
"""
更新配置文件以支持WebSocket服务器
"""

import json
from pathlib import Path

def update_config():
    """更新配置文件"""
    config_path = Path("./config/config.json")
    
    if not config_path.exists():
        print("❌ 配置文件不存在")
        return False
    
    try:
        # 读取现有配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 添加WebSocket服务器配置
        if "server" not in config:
            config["server"] = {}
        
        config["server"]["websocket_host"] = "0.0.0.0"
        config["server"]["websocket_port"] = 8765
        
        # 保存更新后的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("✅ 配置文件已更新")
        print("   WebSocket主机: 0.0.0.0")
        print("   WebSocket端口: 8765")
        
        return True
        
    except Exception as e:
        print(f"❌ 更新配置文件失败: {e}")
        return False

def create_godot_project_structure():
    """创建Godot项目目录结构建议"""
    print("\n🎮 Godot 4 项目结构建议:")
    print("""
your_godot_project/
├── project.godot          # Godot项目文件
├── icon.png              # 项目图标
├── README.md             # 项目说明
├── scenes/               # 场景目录
│   └── main.tscn         # 主场景
├── scripts/              # GDScript脚本目录
│   ├── digital_life_client.gd  # WebSocket客户端
│   ├── audio_manager.gd        # 音频管理器
│   └── simple_ui.gd            # 简单UI
└── assets/               # 资源文件
    └── sounds/           # 音频资源
    """)

def print_quick_start_guide():
    """打印快速启动指南"""
    print("\n🚀 快速启动指南:")
    print("=" * 50)
    print("1. 启动WebSocket服务器:")
    print("   python api_server.py")
    print("   或使用: python quick_start_server.py")
    print()
    print("2. 在Godot 4中:")
    print("   - 创建一个新项目")
    print("   - 复制 godot_client_demo.gd 到 scripts/ 目录")
    print("   - 创建一个主场景并附加该脚本")
    print("   - 或使用提供的 simple_ui.gd 作为UI")
    print()
    print("3. 连接测试:")
    print("   - 运行Godot项目")
    print("   - 点击录音按钮开始录音")
    print("   - 点击停止按钮发送录音")
    print("   - 等待AI回复并播放")
    print()
    print("4. 故障排除:")
    print("   - 检查服务器是否运行: 访问 ws://localhost:8765")
    print("   - 查看服务器日志了解详细错误")
    print("   - 确保音频设备正常工作")

if __name__ == "__main__":
    print("🔄 更新数字生命系统配置...")
    
    # 更新配置文件
    update_config()
    
    # 显示Godot项目结构
    create_godot_project_structure()
    
    # 显示快速启动指南
    print_quick_start_guide()