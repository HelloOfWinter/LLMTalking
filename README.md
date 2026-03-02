LLMTalking
A complete real-time voice dialogue system with streaming ASR, LLM, and TTS. It enables natural conversations with AI characters through your microphone, with support for both local models (via Hugging Face Transformers) and cloud APIs (DeepSeek, OpenAI, etc.). The system features a WebSocket server, Gradio WebUI for management, and a Godot client for cross-platform voice interaction.

Features
End-to-end voice conversation pipeline
Capture microphone audio → Speech recognition (ASR) → Language model (LLM) → Text-to-speech (TTS) → Playback

Streaming output
LLM responses are streamed sentence-by-sentence, allowing TTS synthesis and playback to begin before the full response is generated.

Multiple LLM backends

Local models (via Transformers, supports quantization 4/8-bit)

API mode: DeepSeek, OpenAI, Zhipu (and any OpenAI-compatible API)

Fast TTS with Genie
Uses Genie TTS (ONNX runtime) for low-latency, high-quality voice synthesis. Supports conversion from GPT-SoVITS models.

Flexible character management
Define multiple AI characters with custom system prompts, LLM base models, LoRA adapters, and TTS voice settings.

Conversation history
Save, load, and delete past conversations; maintain context across sessions.

WebUI control panel
Built with Gradio, you can:

Configure system, models, and characters

Start/stop the server and view live logs

Convert GPT-SoVITS models to Genie ONNX format

Godot client included
A C# client for Godot 4 that handles microphone recording (via PvRecorder), WebSocket communication, and audio playback.

Architecture
text
┌─────────────┐      ┌─────────────────────────────────┐      ┌─────────────┐
│ Godot Client│─────▶│    WebSocket Server (FastAPI)   │─────▶│   ASR Model │
└─────────────┘      │   - handles audio chunks        │      │  (Whisper)  │
                     │   - manages conversations       │      └─────────────┘
                     │   - streams responses           │             │
                     └───────────────┬─────────────────┘             ▼
                                     │                       ┌─────────────────┐
                                     │                       │  LLM (local/API)│
                                     │                       └─────────────────┘
                                     │                               │
                                     ▼                               ▼
                            ┌─────────────────┐              ┌─────────────────┐
                            │   TTS (Genie)   │◀─────────────│   Sentence      │
                            └─────────────────┘              │   Streamer      │
Requirements
Python 3.9+ (recommended 3.10)

CUDA-compatible GPU (optional, for acceleration)

Godot 4.3+ (if using the Godot client)

Installation
Clone the repository:

bash
git clone https://github.com/yourusername/LLMTalking.git
cd LLMTalking
Create a virtual environment (optional but recommended):

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Python dependencies:

bash
pip install -r requirements.txt
For GPU support, you may need to install a CUDA‑compatible version of PyTorch separately:

bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Download required models:

ASR: Place a Whisper model (e.g., whisper-small) in ./whisper-small (or change path in config). You can download using Hugging Face:

python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
WhisperProcessor.from_pretrained("openai/whisper-small", cache_dir="./whisper-small")
WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", cache_dir="./whisper-small")
LLM (local): Download a model (e.g., Qwen3) into ./SLM/Qwen3-4B-Instruct-2507 or adjust base_model path in character config.

TTS (Genie): Convert a GPT-SoVITS model pair (.pth and .ckpt) to ONNX format using the built‑in converter in the WebUI or via convert_to_genie.py.

Configuration
All settings are stored in config/config.json. You can edit it manually or use the WebUI.

System: server host/port, hardware options.

Models: ASR model path/device, LLM mode (local/api), API keys, etc.

Characters: each character has its own LLM base, LoRA, TTS settings, and system prompt.

Use the WebUI (webui.py) to manage everything visually.

Usage
Start the WebUI (optional, for configuration and server control)
bash
python webui.py
Then open http://127.0.0.1:7860 in your browser.

Start the WebSocket server directly
bash
python api_server_fixed.py
By default, the server listens on 0.0.0.0:8765.

Run the Godot client
Open Godot 4, create a new project.

Copy AudioClient.cs into the scripts folder.

Attach the script to a main scene.

Ensure the UI nodes (RecordButton, StopButton, HistoryButton, StatusLabel) are present, or adjust the script to match your scene tree.

Run the project. Make sure the Python server is already running.

WebSocket Protocol
The server communicates with clients using JSON messages over WebSocket.

Client → Server
Start a new conversation

json
{"type": "new_conversation"}
Send audio chunk (after creating conversation)

json
{
  "type": "audio_chunk",
  "seq": 0,
  "total": 5,
  "data": "base64_encoded_pcm_data",
  "sample_rate": 16000,
  "conversation_id": "..."
}
History management

json
{"type": "history_request", "action": "list"}
{"type": "history_request", "action": "load", "conversation_id": "..."}
{"type": "history_request", "action": "save", "conversation_id": "...", "title": "..."}
{"type": "history_request", "action": "delete", "conversation_id": "..."}
Ping

json
{"type": "ping"}
Server → Client
Conversation created

json
{"type": "conversation_created", "conversation_id": "..."}
Sentence start

json
{"type": "sentence_start", "idx": 0, "text": "...", "total_chunks": 3, "sample_rate": 22050}
Audio chunk for a sentence

json
{"type": "audio_chunk", "sentence_idx": 0, "seq": 0, "data": "base64..."}
Sentence end

json
{"type": "sentence_end", "idx": 0}
Stream end (all sentences sent)

json
{"type": "stream_end"}
History responses (history_list, history_load, history_saved, history_deleted)

Pong

json
{"type": "pong"}
Project Structure
text
LLMTalking/
├── api_server_fixed.py          # Main WebSocket server
├── model_manager_optimized.py   # Model loading/inference logic
├── genie_adapter.py             # Genie TTS wrapper
├── config.py                     # Configuration management
├── config_tool.py                # CLI for config
├── init_config.py                # One-time config initializer
├── webui.py                       # Gradio control panel
├── convert_to_genie.py           # GPT-SoVITS → Genie converter
├── patch_model_manager.py        # Helper to remove special chars
├── run_server_simple.py          # Simple launcher
├── update_config_for_ws.py       # Add WebSocket settings
├── audio_utils.py                # PCM/WAV helpers
├── requirements.txt              # Python dependencies
├── README.md                      # This file
├── config/                        # Configuration directory (created on first run)
│   ├── config.json                # Main config
│   ├── characters/                 # Character JSON files
│   └── backups/                    # Auto backups
├── conversation_history/          # Saved conversations
└── (optional) SLM/                # Local LLM models
└── (optional) whisper-small/      # ASR model
Contributing
Contributions are welcome! Please open an issue or submit a pull request.

License
MIT License

Enjoy building your own AI companion!