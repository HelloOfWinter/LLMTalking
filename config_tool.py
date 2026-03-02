# config_tool.py
"""
配置工具脚本 - 用于管理配置文件
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import ConfigManager

def main():
    parser = argparse.ArgumentParser(description="配置管理工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # init命令：初始化配置
    init_parser = subparsers.add_parser("init", help="初始化配置文件")
    
    # list命令：列出配置
    list_parser = subparsers.add_parser("list", help="列出当前配置")
    list_parser.add_argument("--key", help="指定配置键")
    
    # get命令：获取配置项
    get_parser = subparsers.add_parser("get", help="获取配置项")
    get_parser.add_argument("key", help="配置键（如：system.name）")
    
    # set命令：设置配置项
    set_parser = subparsers.add_parser("set", help="设置配置项")
    set_parser.add_argument("key", help="配置键（如：system.name）")
    set_parser.add_argument("value", help="配置值")
    
    # validate命令：验证配置
    subparsers.add_parser("validate", help="验证配置有效性")
    
    # reset命令：重置配置
    subparsers.add_parser("reset", help="重置为默认配置")
    
    # characters命令：管理角色
    char_parser = subparsers.add_parser("characters", help="管理角色配置")
    char_subparsers = char_parser.add_subparsers(dest="char_command")
    
    # 列出角色
    char_subparsers.add_parser("list", help="列出所有角色")
    
    # 查看角色
    char_view = char_subparsers.add_parser("view", help="查看角色详情")
    char_view.add_argument("character_id", help="角色ID")
    
    # 创建角色
    char_create = char_subparsers.add_parser("create", help="创建新角色")
    char_create.add_argument("character_id", help="角色ID")
    char_create.add_argument("--name", help="角色名称", required=True)
    char_create.add_argument("--description", help="角色描述", default="")
    
    # 删除角色
    char_delete = char_subparsers.add_parser("delete", help="删除角色")
    char_delete.add_argument("character_id", help="角色ID")
    
    # 切换当前角色
    char_switch = char_subparsers.add_parser("switch", help="切换当前角色")
    char_switch.add_argument("character_id", help="角色ID")
    
    args = parser.parse_args()
    
    # 初始化配置管理器
    config_manager = ConfigManager()
    
    if args.command == "init":
        print("初始化配置文件...")
        config_manager.reset_to_default()
        print("✅ 配置文件初始化完成")
        
    elif args.command == "list":
        if args.key:
            value = config_manager.get(args.key)
            print(f"{args.key}: {value}")
        else:
            import json
            print(json.dumps(config_manager.config, indent=2, ensure_ascii=False))
            
    elif args.command == "get":
        value = config_manager.get(args.key)
        print(f"{args.key}: {value}")
        
    elif args.command == "set":
        # 尝试将值转换为合适的类型
        value = args.value
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():
            value = float(value)
            
        success = config_manager.set(args.key, value)
        if success:
            config_manager.save()
            print(f"✅ 设置 {args.key} = {value}")
        else:
            print(f"❌ 设置失败")
            
    elif args.command == "validate":
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
        
        if not has_errors:
            print("\n✅ 配置验证通过")
        else:
            print("\n❌ 配置存在错误，请修复")
            sys.exit(1)
            
    elif args.command == "reset":
        confirm = input("确定要重置为默认配置吗？(y/N): ")
        if confirm.lower() == 'y':
            config_manager.reset_to_default()
            print("✅ 配置已重置")
        else:
            print("操作取消")
            
    elif args.command == "characters":
        if args.char_command == "list":
            characters = config_manager.list_characters()
            current_char = config_manager.get("character.current")
            
            print(f"当前角色: {current_char or '无'}")
            print("\n所有角色:")
            for char in characters:
                status = "✓" if char["id"] == current_char else " "
                enabled = "✅" if char["enabled"] else "❌"
                print(f"  {status} [{enabled}] {char['id']} - {char['name']}")
                if char["description"]:
                    print(f"      {char['description']}")
                    
        elif args.char_command == "view":
            char_config = config_manager.load_character(args.character_id)
            if char_config:
                import json
                print(json.dumps(char_config, indent=2, ensure_ascii=False))
            else:
                print(f"❌ 角色 '{args.character_id}' 不存在")
                
        elif args.char_command == "create":
            # 创建新角色
            template = config_manager.get_character_template()
            template["id"] = args.character_id
            template["name"] = args.name
            template["description"] = args.description
            
            success = config_manager.save_character(args.character_id, template)
            if success:
                print(f"✅ 角色 '{args.character_id}' 创建成功")
            else:
                print(f"❌ 角色创建失败")
                
        elif args.char_command == "delete":
            confirm = input(f"确定要删除角色 '{args.character_id}' 吗？(y/N): ")
            if confirm.lower() == 'y':
                success = config_manager.delete_character(args.character_id)
                if success:
                    print(f"✅ 角色 '{args.character_id}' 已删除")
                else:
                    print(f"❌ 删除失败")
            else:
                print("操作取消")
                
        elif args.char_command == "switch":
            char_config = config_manager.load_character(args.character_id)
            if char_config:
                config_manager.set("character.current", args.character_id)
                config_manager.save()
                print(f"✅ 已切换到角色 '{args.character_id}'")
            else:
                print(f"❌ 角色 '{args.character_id}' 不存在")
                
    else:
        parser.print_help()

if __name__ == "__main__":
    main()