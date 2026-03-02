# init_config.py
"""
初始化配置文件 - 一次性运行
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import ConfigManager

def init_configuration():
    """初始化配置"""
    print("=" * 60)
    print("初始化数字生命系统配置")
    print("=" * 60)
    
    # 初始化配置管理器
    config_manager = ConfigManager()
    
    print("\n1. 检查配置文件...")
    if config_manager.config_file.exists():
        print("✅ 配置文件已存在")
    else:
        print("✅ 已创建默认配置文件")
    
    print("\n2. 检查角色配置...")
    characters = config_manager.list_characters()
    if characters:
        print(f"✅ 找到 {len(characters)} 个角色:")
        for char in characters:
            print(f"   - {char['name']} ({char['id']})")
    else:
        print("❌ 未找到角色配置")
    
    print("\n3. 验证配置...")
    errors = config_manager.validate_config()
    has_errors = False
    
    for category, error_list in errors.items():
        if error_list:
            has_errors = True
            print(f"❌ {category}:")
            for error in error_list:
                print(f"   - {error}")
        else:
            print(f"✅ {category}: 正常")
    
    print("\n4. 配置摘要:")
    print(f"   API 端口: {config_manager.get('server.api_port')}")
    print(f"   WebUI 端口: {config_manager.get('webui_port')}")
    print(f"   当前角色: {config_manager.get('character.current')}")
    print(f"   设备类型: {config_manager.get('hardware.device')}")
    
    if has_errors:
        print("\n⚠️  配置存在一些问题，请检查:")
        print("   - 确保模型文件路径正确")
        print("   - 确保参考音频文件存在")
        print("\n运行以下命令查看详细信息:")
        print("   python config_tool.py validate")
    else:
        print("\n✅ 配置初始化完成!")
        print("\n下一步:")
        print("   1. 运行测试: python test_config.py")
        print("   2. 启动WebUI: 等待后续实现")
        print("   3. 启动API服务器: 等待后续实现")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    init_configuration()