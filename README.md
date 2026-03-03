# 🗣️ LLMTalking - 数字生命语音对话系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/)
[![Godot 4.6](https://img.shields.io/badge/Godot-4.6_Mono-478cbf.svg)](https://godotengine.org/)

**LLMTalking** 是一个端到端的实时语音对话系统。它由 Godot 客户端和 Python 后端组成，集成了语音识别（ASR）、大语言模型（LLM）和语音合成（TTS），支持角色扮演、对话历史管理，并提供 WebUI 控制面板。

## ✨ 核心特性

- 🎙️ **实时语音对话**：Godot 客户端录音并通过 WebSocket 发送，后端依次调用 ASR、LLM、TTS，返回合成语音流式播放。
- 🧠 **智能回复**：
  - ASR：使用 **Whisper** 模型（支持 CPU/GPU）。
  - LLM：支持**本地模型**（如 Qwen，可加载 LoRA）或**API模式**（DeepSeek、OpenAI、智谱等）。
  - TTS：基于 **Genie-TTS** 引擎（对 GPT-SoVITS 的轻量级优化），支持多角色音色。
- 🎭 **角色扮演**：每个角色可独立配置系统提示词、LLM 参数、TTS 音色（需提供对应的 ONNX 模型文件夹）。
- 🌐 **WebUI 控制面板**：基于 Gradio 构建，可视化管理系统配置、角色、模型转换、服务器启停与日志。
- 🔄 **模型转换**：内置脚本将 GPT-SoVITS 训练好的 `.pth` 和 `.ckpt` 模型转换为 Genie 引擎可用的 ONNX 格式。
- 📜 **对话历史**：自动保存对话记录，支持加载、保存和删除历史会话。

## 🚀 快速开始

### 环境要求
- Python **3.11.9**（其他 3.9+ 版本可能也可行，但推荐 3.11.9）
- Godot **4.6 Mono 版**（用于运行客户端）
- （可选）NVIDIA GPU + CUDA（用于加速 ASR/LLM）

### 1. 克隆项目
git clone https://github.com/HelloOfWinter/LLMTalking.git
cd LLMTalking
cd server
pip install -r requirements.txt
注意：本仓库不包含 Genie-TTS 的模型代码，您需要自行克隆并安装：

bash
# 在主目录下克隆 Genie-TTS
git clone https://github.com/High-Logic/Genie-TTS.git
cd Genie-TTS
Genie-TTS 是轻量级推理引擎的核心，用于将 GPT-SoVITS 模型转换为 ONNX 并进行高性能合成。

3. 配置 Godot 客户端
下载 Godot 4.6 Mono 版

在 Godot 中打开本项目的 godot/the.zip 文件夹作为项目

首次打开时，Godot 会自动还原 NuGet 包（LibVLCSharp, PvRecorder），若失败请手动安装

4. 准备模型与配置
复制配置示例：
可根据需要修改或新增。每个角色需指定：

LLM 基础模型路径、LoRA 路径（可选）,也可使用api

TTS 的 ONNX 模型目录（由下一步转换得到）、参考音频路径

转换 TTS 模型（如使用自有 GPT-SoVITS 模型）：
python scripts/convert_to_genie.py --pth /path/to/your/model.pth --ckpt /path/to/your/model.ckpt --output /path/to/output_dir
将输出目录（如 /path/to/output_dir）填入角色配置的 onnx_dir 字段。
如果您暂无模型，可使用 Genie-TTS 自带的预定义角色快速体验，详见 Genie-TTS 文档。
. 启动服务
方式一：直接启动后端
cd server
python api_server_fixed.py
# 默认监听 ws://0.0.0.0:8765
方式二：通过 WebUI 管理（推荐）
cd webui
python webui.py
访问 http://127.0.0.1:7860，在 WebUI 中可：

启动/停止/重启后端服务器

查看实时日志

管理系统、模型、角色配置

转换模型

6. 运行客户端
在 Godot 编辑器中运行项目，点击界面上的“录音”按钮开始对话。确保客户端能连接到后端的 WebSocket 地址（默认 ws://localhost:8765/ws）。


🛠️ 模型转换详解
若您训练了自己的 GPT-SoVITS 模型（版本需为 V2 或 V2ProPlus），可使用 scripts/convert_to_genie.py 将其转换为 Genie 引擎所需的 ONNX 格式。

示例：

bash
python scripts/convert_to_genie.py \
    --pth "D:/models/SoVITS_weights_v4/Ye.pth" \
    --ckpt "D:/models/GPT_weights_v4/Ye.ckpt" \
    --output "D:/genie_models/Ye"
转换后文件夹中会生成多个 .onnx 和 .bin 文件，将该文件夹路径填入角色配置的 onnx_dir 即可。

🙏 致谢
本项目基于众多优秀的开源项目构建，特此致谢：

Genie-TTS：轻量级推理引擎，极大优化了 GPT-SoVITS 的 CPU 推理性能。

GPT-SoVITS：强大的少样本语音合成与转换模型。

Godot Engine：免费、开源的游戏引擎，提供跨平台的客户端运行环境。

HuggingFace Transformers、Whisper、FastAPI、Gradio 等开源库。

所有贡献代码、提出建议、参与测试的社区朋友。

🤝 贡献指南
欢迎提交 Issue 和 Pull Request！在贡献前请确保：

代码风格与现有代码保持一致

为新增功能添加必要的注释

确保原有功能不受影响

📄 许可证
本项目采用 MIT 许可证。您可以自由使用、修改和分发，但需保留原版权声明。详见 LICENSE 文件。

开始创造您的数字生命吧！ 🎉
