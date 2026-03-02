# webui.py
"""
数字生命系统 WebUI 控制面板 (Gradio 4.x)
支持系统配置、模型配置、角色管理、服务器控制、日志查看、模型转换
"""

import sys
import os
import json
import time
import subprocess
import threading
import queue
import io
import contextlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator

import gradio as gr

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import ConfigManager

# 尝试导入 genie_tts，用于模型转换
try:
    import genie_tts as genie
    GENIE_AVAILABLE = True
except ImportError:
    GENIE_AVAILABLE = False
    print("[警告] genie-tts 未安装，模型转换功能将不可用。")

# ==================== 全局变量 ====================
config_manager = ConfigManager()
server_process = None
log_messages = []          # 存储所有日志（最多1000条）
log_lock = threading.Lock()
MAX_LOG_LINES = 1000

# ==================== 辅助函数 ====================
def get_all_logs() -> str:
    """获取所有日志"""
    with log_lock:
        return "\n".join(log_messages)

def clear_logs() -> str:
    """清除日志"""
    with log_lock:
        log_messages.clear()
    return "日志已清除"

def get_logs() -> str:
    """获取最近100条日志（用于定时刷新）"""
    with log_lock:
        return "\n".join(log_messages[-100:])

def read_server_output(process: subprocess.Popen):
    """读取子进程输出并存入日志列表"""
    for line in iter(process.stdout.readline, ''):
        if line:
            with log_lock:
                log_messages.append(line.rstrip())
                if len(log_messages) > MAX_LOG_LINES:
                    log_messages.pop(0)
        else:
            break
    with log_lock:
        log_messages.append("[系统] 服务器进程已退出")

def start_server(host: str, port: int) -> str:
    """启动后端服务器进程"""
    global server_process
    if server_process and server_process.poll() is None:
        return "服务器已在运行中"

    config_manager.save()

    cmd = [sys.executable, "api_server_fixed.py", "--host", host, "--port", str(port)]
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        threading.Thread(target=read_server_output, args=(server_process,), daemon=True).start()
        return f"服务器启动成功，监听 {host}:{port}"
    except Exception as e:
        return f"启动失败: {str(e)}"

def stop_server() -> str:
    """停止后端服务器进程"""
    global server_process
    if server_process and server_process.poll() is None:
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        server_process = None
        with log_lock:
            log_messages.append("[系统] 服务器已停止")
        return "服务器已停止"
    return "服务器未运行"

def restart_server(host: str, port: int) -> str:
    """重启服务器"""
    stop_server()
    time.sleep(1)
    return start_server(host, port)

def get_model_status() -> str:
    """获取模型管理器的详细状态（基于配置文件检查）"""
    current_char = config_manager.get("character.current", "")
    lines = []
    lines.append(f"当前角色: {current_char}")

    # ASR
    asr_path = config_manager.get("models.asr.model_path", "")
    exists = os.path.exists(asr_path) if asr_path else False
    lines.append(f"ASR模型路径: {asr_path}  [{'存在' if exists else '不存在'}]")

    # LLM
    llm_base = config_manager.get("models.llm.base_model", "")
    exists = os.path.exists(llm_base) if llm_base else False
    lines.append(f"LLM基础模型: {llm_base}  [{'存在' if exists else '不存在'}]")

    llm_lora = config_manager.get("models.llm.lora_path", "")
    if llm_lora:
        exists = os.path.exists(llm_lora) if llm_lora else False
        lines.append(f"LLM LoRA路径: {llm_lora}  [{'存在' if exists else '不存在'}]")

    # TTS (Genie)
    if current_char:
        char_config = config_manager.load_character(current_char)
        if char_config:
            tts = char_config.get("tts", {})
            onnx_dir = tts.get("onnx_dir", "")
            exists = os.path.exists(onnx_dir) if onnx_dir else False
            lines.append(f"Genie ONNX目录: {onnx_dir}  [{'存在' if exists else '不存在'}]")
            ref_audio = tts.get("ref_audio", "")
            exists = os.path.exists(ref_audio) if ref_audio else False
            lines.append(f"参考音频: {ref_audio}  [{'存在' if exists else '不存在'}]")
    else:
        lines.append("未选择当前角色，无法检查TTS配置")

    return "\n".join(lines)

# ==================== 系统配置相关 ====================
def load_system_config() -> Dict[str, Any]:
    """加载系统配置"""
    config = config_manager.config
    return {
        "host": config.get("server", {}).get("websocket_host", "0.0.0.0"),
        "port": config.get("server", {}).get("websocket_port", 8765),
    }

def save_system_config(host: str, port: int) -> str:
    """保存系统配置"""
    config_manager.set("server.websocket_host", host)
    config_manager.set("server.websocket_port", int(port))
    config_manager.save()
    return "系统配置已保存"

# ==================== 模型配置相关 ====================
def load_model_config() -> Dict[str, Any]:
    """加载模型配置"""
    config = config_manager.config
    models = config.get("models", {})
    asr = models.get("asr", {})
    llm = models.get("llm", {})
    return {
        "asr_model_name": asr.get("model_name", "whisper-small"),
        "asr_model_path": asr.get("model_path", "./whisper-small"),
        "asr_device": asr.get("device", "cpu"),
        "llm_mode": llm.get("mode", "local"),
        "llm_base_model": llm.get("base_model", "./SLM/Qwen3-1.7B"),
        "llm_lora_path": llm.get("lora_path", ""),
        "llm_device": llm.get("device", "cuda"),
        "llm_4bit": llm.get("load_in_4bit", False),
        "llm_api_type": llm.get("api_type", "deepseek"),
        "llm_api_key": llm.get("api_key", ""),
        "llm_api_base": llm.get("api_base", ""),
        "llm_api_model": llm.get("api_model_name", "deepseek-chat"),
    }

def save_model_config(
    asr_model_name, asr_model_path, asr_device,
    llm_mode, llm_base_model, llm_lora_path, llm_device, llm_4bit,
    llm_api_type, llm_api_key, llm_api_base, llm_api_model,
) -> str:
    """保存模型配置"""
    config_manager.set("models.asr.model_name", asr_model_name)
    config_manager.set("models.asr.model_path", asr_model_path)
    config_manager.set("models.asr.device", asr_device)

    config_manager.set("models.llm.mode", llm_mode)
    config_manager.set("models.llm.base_model", llm_base_model)
    config_manager.set("models.llm.lora_path", llm_lora_path)
    config_manager.set("models.llm.device", llm_device)
    config_manager.set("models.llm.load_in_4bit", llm_4bit)
    config_manager.set("models.llm.api_type", llm_api_type)
    config_manager.set("models.llm.api_key", llm_api_key)
    config_manager.set("models.llm.api_base", llm_api_base)
    config_manager.set("models.llm.api_model_name", llm_api_model)

    config_manager.save()
    return "模型配置已保存"

# ==================== 角色管理相关 ====================
def get_characters_list() -> List[str]:
    """获取所有角色ID列表（用于下拉框）"""
    chars = config_manager.list_characters()
    return [c["id"] for c in chars]

def get_character_names() -> Dict[str, str]:
    """获取角色ID到显示名称的映射"""
    chars = config_manager.list_characters()
    return {c["id"]: c["name"] for c in chars}

def load_character_form(character_id: str) -> List:
    """根据角色ID加载表单数据"""
    if not character_id or character_id == "":
        # 新建角色时返回空表单
        return ["", "", "", "", "", "", 0.7, 300, "", "", "zh", "", "", "zh"]

    char_config = config_manager.load_character(character_id)
    if not char_config:
        return ["", "", "", "", "", "", 0.7, 300, "", "", "zh", "", "", "zh"]

    name = char_config.get("name", "")
    description = char_config.get("description", "")
    llm_config = char_config.get("llm", {})
    dialog = char_config.get("dialog", {})
    tts = char_config.get("tts", {})

    base_model = llm_config.get("base_model", "")
    lora_path = llm_config.get("lora_path", "")
    system_prompt = dialog.get("system_prompt", "")
    temperature = dialog.get("temperature", 0.7)
    max_tokens = dialog.get("max_tokens", 300)

    # Genie TTS 字段
    onnx_dir = tts.get("onnx_dir", "")
    ref_audio = tts.get("ref_audio", "")
    ref_text = tts.get("ref_text", "")
    language = tts.get("language", "zh")
    ref_lang = tts.get("ref_lang", "zh")

    return [
        character_id,        # ID只读
        name,
        description,
        base_model,
        lora_path,
        system_prompt,
        temperature,
        max_tokens,
        onnx_dir,
        ref_audio,
        ref_text,
        language,
        ref_lang
    ]

def save_character_form(
    character_id: str,
    name: str,
    description: str,
    base_model: str,
    lora_path: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    onnx_dir: str,
    ref_audio: str,
    ref_text: str,
    language: str,
    ref_lang: str
) -> str:
    """保存角色配置（如果character_id为空，则创建新角色）"""
    if not character_id.strip():
        return "角色ID不能为空"
    existing = config_manager.load_character(character_id)
    if existing is None:
        char_config = config_manager.get_character_template()
        char_config["id"] = character_id
    else:
        char_config = existing

    char_config["name"] = name
    char_config["description"] = description

    # LLM
    if "llm" not in char_config:
        char_config["llm"] = {}
    char_config["llm"]["base_model"] = base_model
    char_config["llm"]["lora_path"] = lora_path
    char_config["llm"]["use_base_model"] = (lora_path == "")

    # Dialog
    if "dialog" not in char_config:
        char_config["dialog"] = {}
    char_config["dialog"]["system_prompt"] = system_prompt
    char_config["dialog"]["temperature"] = temperature
    char_config["dialog"]["max_tokens"] = max_tokens

    # TTS (Genie)
    if "tts" not in char_config:
        char_config["tts"] = {}
    char_config["tts"]["engine"] = "genie"
    char_config["tts"]["onnx_dir"] = onnx_dir
    char_config["tts"]["ref_audio"] = ref_audio
    char_config["tts"]["ref_text"] = ref_text
    char_config["tts"]["language"] = language
    char_config["tts"]["ref_lang"] = ref_lang

    success = config_manager.save_character(character_id, char_config)
    if success:
        if not config_manager.get("character.current"):
            config_manager.set("character.current", character_id)
            config_manager.save()
        return f"角色 '{name}' 保存成功"
    else:
        return "保存失败"

def delete_character(character_id: str) -> str:
    """删除角色"""
    if not character_id:
        return "请选择角色"
    name_map = get_character_names()
    name = name_map.get(character_id, character_id)
    success = config_manager.delete_character(character_id)
    if success:
        if config_manager.get("character.current") == character_id:
            config_manager.set("character.current", "")
            config_manager.save()
        return f"角色 '{name}' 已删除"
    else:
        return f"删除角色 '{name}' 失败"

def set_current_character(character_id: str) -> str:
    """设为当前角色"""
    if not character_id:
        return "请选择角色"
    if not config_manager.load_character(character_id):
        return f"角色 '{character_id}' 不存在"
    config_manager.set("character.current", character_id)
    config_manager.save()
    return f"当前角色已切换为 {character_id}"

# ==================== 模型转换功能 ====================
def convert_model_ui(pth_path: str, ckpt_path: str, output_dir: str, progress=gr.Progress()) -> Generator[str, None, None]:
    """
    调用 genie_tts.convert_to_onnx 进行模型转换，并实时返回日志。
    """
    if not GENIE_AVAILABLE:
        yield "错误：genie-tts 未安装，请先运行 `pip install genie-tts`"
        return

    if not pth_path or not ckpt_path or not output_dir:
        yield "请填写所有输入路径。"
        return

    if not os.path.exists(pth_path):
        yield f"错误：SoVITS 权重文件不存在 - {pth_path}"
        return
    if not os.path.exists(ckpt_path):
        yield f"错误：GPT 权重文件不存在 - {ckpt_path}"
        return

    # 创建输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        yield f"错误：无法创建输出目录 - {e}"
        return

    yield f"开始转换模型...\nSoVITS: {pth_path}\nGPT: {ckpt_path}\n输出目录: {output_dir}\n"

    # 用于捕获 print 输出的队列
    log_queue = queue.Queue()
    stop_event = threading.Event()

    def target():
        # 重定向 stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            # 调用转换函数
            genie.convert_to_onnx(
                torch_pth_path=pth_path,
                torch_ckpt_path=ckpt_path,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"转换过程中发生异常: {e}")
        finally:
            # 恢复 stdout
            sys.stdout = old_stdout
            stop_event.set()

    thread = threading.Thread(target=target)
    thread.start()

    # 循环读取 stdout 并放入队列
    # 由于我们重定向了 sys.stdout，无法直接从线程读取，因此需要在目标函数内部实时将输出放入队列
    # 但简化起见，我们可以在目标函数内部使用一个自定义的 writer 将输出写入队列。
    # 更简单：我们可以在 target 中捕获 print 并放入队列。我们修改 target 的实现：

    # 我们重写 target，使用 contextlib.redirect_stdout 配合自定义流。
    # 但由于我们需要实时从主线程获取，最好使用一个自定义的 StringIO 并定期 flush。
    # 下面采用 queue 和自定义 write 方法。

    class QueueWriter(io.StringIO):
        def __init__(self, q):
            super().__init__()
            self.q = q
        def write(self, s):
            if s.strip():
                self.q.put(s)
            super().write(s)

    # 重新定义 target
    def target_with_queue():
        writer = QueueWriter(log_queue)
        with contextlib.redirect_stdout(writer):
            try:
                genie.convert_to_onnx(
                    torch_pth_path=pth_path,
                    torch_ckpt_path=ckpt_path,
                    output_dir=output_dir
                )
            except Exception as e:
                print(f"转换过程中发生异常: {e}")
        stop_event.set()

    thread = threading.Thread(target=target_with_queue)
    thread.start()

    # 从队列中读取日志并 yield
    while not stop_event.is_set() or not log_queue.empty():
        try:
            line = log_queue.get(timeout=0.2)
            yield line
        except queue.Empty:
            continue

    # 读取剩余的日志
    while not log_queue.empty():
        yield log_queue.get_nowait()

    yield "\n转换完成！"
    # 列出输出目录中的文件
    try:
        files = os.listdir(output_dir)
        if files:
            yield f"\n生成的文件 ({len(files)}):"
            for f in files:
                yield f"  - {f}"
        else:
            yield "\n警告：输出目录为空，可能转换未成功。"
    except Exception as e:
        yield f"\n无法列出输出目录: {e}"

# ==================== 构建UI ====================
def build_ui() -> gr.Blocks:
    """构建Gradio界面"""
    with gr.Blocks(title="数字生命系统控制面板", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 数字生命系统控制台")

        # ---------- 系统配置标签页 ----------
        with gr.Tab("系统配置"):
            with gr.Row():
                host_input = gr.Textbox(label="服务器主机", value="0.0.0.0")
                port_input = gr.Number(label="服务器端口", value=8765, precision=0)
            save_sys_btn = gr.Button("保存系统配置")
            sys_status = gr.Textbox(label="状态", interactive=False)

            save_sys_btn.click(
                fn=save_system_config,
                inputs=[host_input, port_input],
                outputs=sys_status
            )

        # ---------- 模型状态标签页 ----------
        with gr.Tab("模型状态"):
            model_status_output = gr.Textbox(label="模型路径检查", lines=15, interactive=False)
            refresh_status_btn = gr.Button("刷新状态")
            refresh_status_btn.click(
                fn=get_model_status,
                outputs=model_status_output
            )
            demo.load(fn=get_model_status, outputs=model_status_output)

        # ---------- 模型配置标签页 ----------
        with gr.Tab("模型配置"):
            gr.Markdown("### ASR 配置")
            with gr.Row():
                asr_model_name = gr.Textbox(label="ASR模型名称", value="whisper-small")
                asr_model_path = gr.Textbox(label="ASR模型路径", value="./whisper-small")
                asr_device = gr.Dropdown(label="ASR设备", choices=["cuda", "cpu"], value="cpu")

            gr.Markdown("### LLM 配置")
            llm_mode = gr.Radio(label="LLM运行模式", choices=["local", "api"], value="local", interactive=True)

            with gr.Group(visible=True) as local_group:
                with gr.Row():
                    llm_base_model = gr.Textbox(label="本地基础模型路径", value="./SLM/Qwen3-1.7B")
                    llm_lora_path = gr.Textbox(label="LoRA路径（可选）", value="")
                with gr.Row():
                    llm_device = gr.Dropdown(label="本地模型设备", choices=["cuda", "cpu"], value="cuda")
                    llm_4bit = gr.Checkbox(label="使用4-bit量化", value=False)

            with gr.Group(visible=False) as api_group:
                with gr.Row():
                    llm_api_type = gr.Dropdown(label="API类型", choices=["openai", "deepseek", "zhipu"], value="deepseek")
                    llm_api_model = gr.Textbox(label="API模型名称", value="deepseek-chat")
                with gr.Row():
                    llm_api_key = gr.Textbox(label="API密钥", type="password", value="")
                    llm_api_base = gr.Textbox(label="API基础地址（可选）", value="")

            def toggle_llm_groups(mode):
                return {
                    local_group: gr.update(visible=(mode == "local")),
                    api_group: gr.update(visible=(mode == "api"))
                }
            llm_mode.change(
                fn=toggle_llm_groups,
                inputs=llm_mode,
                outputs=[local_group, api_group]
            )

            save_model_btn = gr.Button("保存模型配置")
            model_status = gr.Textbox(label="状态", interactive=False)

            save_model_btn.click(
                fn=save_model_config,
                inputs=[
                    asr_model_name, asr_model_path, asr_device,
                    llm_mode, llm_base_model, llm_lora_path, llm_device, llm_4bit,
                    llm_api_type, llm_api_key, llm_api_base, llm_api_model
                ],
                outputs=model_status
            )

        # ---------- 角色管理标签页 ----------
        with gr.Tab("角色管理"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 角色列表")
                    character_selector = gr.Dropdown(
                        label="选择角色",
                        choices=get_characters_list(),
                        interactive=True,
                        value=None
                    )
                    with gr.Row():
                        new_char_btn = gr.Button("新建角色")
                        refresh_list_btn = gr.Button("刷新列表")
                    with gr.Row():
                        set_current_btn = gr.Button("设为当前角色")
                        delete_char_btn = gr.Button("删除角色", variant="stop")
                    current_char_display = gr.Textbox(label="当前角色", value=config_manager.get("character.current", ""), interactive=False)

                with gr.Column(scale=2):
                    gr.Markdown("### 角色配置")
                    char_id = gr.Textbox(label="角色ID", info="新建时需填写，保存后不可修改")
                    char_name = gr.Textbox(label="角色名称")
                    char_desc = gr.Textbox(label="描述", lines=2)

                    gr.Markdown("#### LLM 设置")
                    with gr.Row():
                        llm_base = gr.Textbox(label="基础模型路径", value="./SLM/Qwen3-1.7B")
                        llm_lora = gr.Textbox(label="LoRA路径（可选）", value="")
                    with gr.Row():
                        system_prompt = gr.Textbox(label="系统提示词", lines=3)
                    with gr.Row():
                        temperature = gr.Slider(label="温度", minimum=0.1, maximum=2.0, value=0.7, step=0.1)
                        max_tokens = gr.Number(label="最大生成长度", value=300, precision=0)

                    gr.Markdown("#### TTS 设置 (Genie引擎)")
                    with gr.Row():
                        onnx_dir = gr.Textbox(label="ONNX模型目录", placeholder="例如 ./tts_models/feibi/tts_models")
                        language = gr.Dropdown(label="语言", choices=["zh", "en", "ja"], value="zh")
                    with gr.Row():
                        ref_audio = gr.Textbox(label="参考音频路径")
                        ref_text = gr.Textbox(label="参考文本")
                        ref_lang = gr.Dropdown(label="参考语言", choices=["zh", "en", "ja"], value="zh")

                    save_char_btn = gr.Button("保存角色", variant="primary")
                    char_save_status = gr.Textbox(label="操作状态", interactive=False)

            # 事件绑定
            def refresh_char_list():
                return gr.Dropdown(choices=get_characters_list())
            refresh_list_btn.click(fn=refresh_char_list, outputs=character_selector)

            character_selector.change(
                fn=load_character_form,
                inputs=character_selector,
                outputs=[char_id, char_name, char_desc, llm_base, llm_lora, system_prompt, temperature, max_tokens, onnx_dir, ref_audio, ref_text, language, ref_lang]
            )

            def new_character():
                return [
                    "", "", "",
                    "./SLM/Qwen3-1.7B", "",
                    "",
                    0.7, 300,
                    "", "", "", "zh", "zh"
                ]
            new_char_btn.click(
                fn=new_character,
                outputs=[char_id, char_name, char_desc, llm_base, llm_lora, system_prompt, temperature, max_tokens, onnx_dir, ref_audio, ref_text, language, ref_lang]
            )

            save_char_btn.click(
                fn=save_character_form,
                inputs=[char_id, char_name, char_desc, llm_base, llm_lora, system_prompt, temperature, max_tokens, onnx_dir, ref_audio, ref_text, language, ref_lang],
                outputs=char_save_status
            ).then(
                fn=refresh_char_list, outputs=character_selector
            ).then(
                fn=lambda: config_manager.get("character.current", ""), outputs=current_char_display
            )

            set_current_btn.click(
                fn=set_current_character,
                inputs=character_selector,
                outputs=char_save_status
            ).then(
                fn=lambda: config_manager.get("character.current", ""), outputs=current_char_display
            )

            delete_char_btn.click(
                fn=delete_character,
                inputs=character_selector,
                outputs=char_save_status
            ).then(
                fn=refresh_char_list, outputs=character_selector
            ).then(
                fn=lambda: config_manager.get("character.current", ""), outputs=current_char_display
            )

            demo.load(fn=lambda: config_manager.get("character.current", ""), outputs=current_char_display)

        # ---------- 模型转换标签页 ----------
        with gr.Tab("模型转换"):
            if not GENIE_AVAILABLE:
                gr.Markdown("⚠️ **genie-tts 未安装**，请运行 `pip install genie-tts` 启用此功能。")
            else:
                gr.Markdown("将 GPT-SoVITS 的 `.pth` 和 `.ckpt` 模型文件转换为 GENIE 格式的 ONNX 文件夹。")
                with gr.Row():
                    pth_input = gr.Textbox(label="SoVITS 权重文件 (.pth)", placeholder="例如：D:/models/SoVITS_weights_v4/Ye.pth")
                    ckpt_input = gr.Textbox(label="GPT 权重文件 (.ckpt)", placeholder="例如：D:/models/GPT_weights_v4/Ye.ckpt")
                output_dir_input = gr.Textbox(label="输出文件夹", placeholder="例如：D:/genie_models/Ye")
                convert_btn = gr.Button("开始转换", variant="primary")
                convert_output = gr.Textbox(label="转换日志", lines=20, interactive=False)

                convert_btn.click(
                    fn=convert_model_ui,
                    inputs=[pth_input, ckpt_input, output_dir_input],
                    outputs=convert_output
                )

        # ---------- 服务器控制标签页 ----------
        with gr.Tab("服务器控制"):
            with gr.Row():
                start_btn = gr.Button("启动服务器", variant="primary")
                stop_btn = gr.Button("停止服务器")
                restart_btn = gr.Button("重启服务器")
            server_status = gr.Textbox(label="操作状态", interactive=False)

            log_output = gr.Textbox(label="服务器日志", lines=20, interactive=False)

            start_btn.click(
                fn=start_server,
                inputs=[host_input, port_input],
                outputs=server_status
            )
            stop_btn.click(
                fn=stop_server,
                outputs=server_status
            )
            restart_btn.click(
                fn=restart_server,
                inputs=[host_input, port_input],
                outputs=server_status
            )

            log_timer = gr.Timer(value=2)
            log_timer.tick(fn=get_logs, outputs=log_output)

        # ---------- 日志查看标签页 ----------
        with gr.Tab("日志查看"):
            with gr.Row():
                clear_log_btn = gr.Button("清除日志")
                refresh_log_btn = gr.Button("刷新日志")
            full_log_output = gr.Textbox(label="完整服务器日志", lines=30, max_lines=1000, interactive=False)

            refresh_log_btn.click(fn=get_all_logs, outputs=full_log_output)
            clear_log_btn.click(fn=clear_logs, outputs=full_log_output).then(fn=get_all_logs, outputs=full_log_output)

            log_timer_full = gr.Timer(value=3)
            log_timer_full.tick(fn=get_all_logs, outputs=full_log_output)

        # ---------- 加载初始值 ----------
        demo.load(
            fn=lambda: (
                load_system_config()["host"],
                load_system_config()["port"],
                *load_model_config().values()
            ),
            outputs=[
                host_input, port_input,
                asr_model_name, asr_model_path, asr_device,
                llm_mode, llm_base_model, llm_lora_path, llm_device, llm_4bit,
                llm_api_type, llm_api_key, llm_api_base, llm_api_model
            ]
        )

    return demo

# ==================== 入口 ====================
if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
