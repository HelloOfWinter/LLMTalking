import os
import json
import time
import uuid
import asyncio
import logging
import torch
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI
from fastapi.websockets import WebSocket
import soundfile as sf
import warnings
from datetime import datetime
from model_manager_optimized import ModelManager
import base64
import tempfile
from starlette.websockets import WebSocketDisconnect
import sys
import io

# 强制标准输出使用 UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_server")

# 确保使用CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"[SYSTEM] 使用设备: {device}")
if device == "cuda":
    torch.cuda.empty_cache()
    logger.info(f"[SYSTEM] 已清理CUDA缓存")
    torch.backends.cudnn.benchmark = True
    logger.info("[SYSTEM] 已启用CUDA优化")
else:
    logger.warning("[SYSTEM] CUDA不可用，使用CPU")

# 配置路径
BASE_DIR = Path.cwd()
HISTORY_DIR = BASE_DIR / "conversation_history"
HISTORY_DIR.mkdir(exist_ok=True)

# 加载角色配置
def load_character_config(character: str) -> Dict:
    """加载角色配置文件"""
    config_path = BASE_DIR / "characters" / f"{character}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"角色配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 对话历史管理
def load_conversation_history(conversation_id: Optional[str] = None) -> List[Dict]:
    if conversation_id:
        history_file = HISTORY_DIR / f"{conversation_id}.json"
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    else:
        histories = []
        for file in HISTORY_DIR.glob("*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                histories.append({
                    "id": file.stem,
                    "title": history.get("title", "对话历史"),
                    "timestamp": history.get("timestamp", "unknown")
                })
        return histories

def save_conversation_history(conversation_id: str, history_data: List[Dict], title: Optional[str] = None) -> bool:
    if title is None:
        title = f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    history_data = {
        "title": title,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": history_data
    }
    history_file = HISTORY_DIR / f"{conversation_id}.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
    return True

def delete_conversation_history(conversation_id: str) -> bool:
    history_file = HISTORY_DIR / f"{conversation_id}.json"
    if history_file.exists():
        history_file.unlink()
        return True
    return False

class ConversationManager:
    def __init__(self):
        self.conversation_history = {}
        self.audio_buffers = {}
        self.current_conversation_id = None

    def get_conversation_history(self, conversation_id: Optional[str] = None) -> Dict:
        if conversation_id is None:
            conversation_id = self.current_conversation_id
        if conversation_id is None:
            return {"messages": []}
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = {"messages": []}
        return self.conversation_history[conversation_id]

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = {"messages": []}
        self.conversation_history[conversation_id]["messages"].append({
            "role": role,
            "content": content
        })

    def get_current_conversation_id(self) -> str:
        if self.current_conversation_id is None:
            self.current_conversation_id = str(uuid.uuid4())
        return self.current_conversation_id

    def create_new_conversation(self) -> str:
        new_id = str(uuid.uuid4())
        self.current_conversation_id = new_id
        self.conversation_history[new_id] = {"messages": []}
        return new_id

    def set_conversation_history(self, conversation_id: str, messages: List[Dict]):
        self.conversation_history[conversation_id] = {"messages": messages}
        self.current_conversation_id = conversation_id

class WebSocketHandler:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.conversation_manager = ConversationManager()
        self.last_ping_time = time.time()
        self.ping_interval = 30

    # ---------- 新增：文本净化函数 ----------
    def _clean_text_for_tts(self, text: str) -> str:
        """
        净化文本，使其更适合 TTS 合成：
        - 删除括号及其内容（中英文括号）
        - 将中文逗号、句号替换为英文逗号、句号
        - 压缩连续重复的标点符号（如将 "......" 替换为 ".")
        - 去除多余空白
        """
        if not text:
            return text

        # 删除括号及括号内内容（非贪婪）
        text = re.sub(r'[（(][^（）()]*[）)]', '', text)

        # 将中文标点替换为英文标点
        text = text.replace('，', ',').replace('。', '.')

        # 压缩连续重复的英文标点
        text = re.sub(r'\.{2,}', '.', text)   # 连续点号 -> 单个点
        text = re.sub(r',{2,}', ',', text)     # 连续逗号 -> 单个逗号
        text = re.sub(r'!{2,}', '!', text)     # 连续感叹号 -> 单个感叹号
        text = re.sub(r'\?{2,}', '?', text)    # 连续问号 -> 单个问号（虽禁用，但保留处理）

        # 压缩连续的中文省略号（……）为单个句号
        text = re.sub(r'…{2,}', '.', text)

        # 去除首尾空白，并将多个连续空白压缩为一个空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    async def handle_connection(self, websocket: WebSocket):
        await websocket.accept()
        logger.info(f"[CONNECT] 客户端连接: {websocket.client}")
        try:
            while True:
                message = await websocket.receive_text()
                try:
                    data = json.loads(message)
                    message_type = data.get("type", "")
                    logger.info(f"[RECV] 收到消息类型: {message_type}")
                    if message_type == "audio_chunk":
                        await self.handle_audio_chunk(websocket, data)
                    elif message_type == "new_conversation":
                        await self.handle_new_conversation(websocket)
                    elif message_type == "ping":
                        await self.handle_ping(websocket)
                    elif message_type == "history_request":
                        await self.handle_history_request(websocket, data)
                    else:
                        logger.warning(f"[WARN] 未知消息类型: {message_type}")
                except json.JSONDecodeError:
                    logger.error(f"[ERROR] 无效的JSON消息: {message}")
        except WebSocketDisconnect:
            logger.info(f"[DISCONNECT] 客户端主动断开连接: {websocket.client}")
        except Exception as e:
            logger.error(f"[ERROR] 处理消息时出错: {str(e)}", exc_info=True)
        finally:
            await websocket.close()
            logger.info(f"[DISCONNECT] 连接已关闭: {websocket.client}")

    async def handle_new_conversation(self, websocket: WebSocket):
        new_id = self.conversation_manager.create_new_conversation()
        await websocket.send_text(json.dumps({
            "type": "conversation_created",
            "conversation_id": new_id
        }))
        logger.info(f"[NEW] 新建对话: {new_id}")

    async def handle_audio_chunk(self, websocket: WebSocket, data: Dict):
        try:
            client_conversation_id = data.get("conversation_id")
            if not client_conversation_id:
                new_id = self.conversation_manager.create_new_conversation()
                await websocket.send_text(json.dumps({
                    "type": "conversation_created",
                    "conversation_id": new_id
                }))
                conversation_id = new_id
                logger.info(f"[NEW] 自动创建新对话: {conversation_id}")
            else:
                conversation_id = client_conversation_id
                self.conversation_manager.get_conversation_history(conversation_id)

            # 确保当前角色已设置
            if self.model_manager.current_character is None:
                current_char = self.model_manager.config_manager.get("character.current")
                if current_char:
                    self.model_manager.current_character = current_char
                    logger.info(f"[ROLE] 从配置加载当前角色: {current_char}")
                else:
                    logger.warning("[WARN] 当前角色未设置，将使用默认提示词")

            audio_data = data.get("data", "")
            if not audio_data:
                logger.error("[ERROR] 无效的音频数据")
                return

            audio_bytes = base64.b64decode(audio_data)
            seq = data.get("seq", 0)
            total = data.get("total", 1)
            sample_rate = data.get("sample_rate", 16000)

            logger.info(f"[AUDIO] 收到分片 seq={seq}, total={total}, conversation_id={conversation_id}")

            if conversation_id not in self.conversation_manager.audio_buffers:
                self.conversation_manager.audio_buffers[conversation_id] = []

            self.conversation_manager.audio_buffers[conversation_id].append({
                "seq": seq,
                "data": audio_bytes
            })

            current_count = len(self.conversation_manager.audio_buffers[conversation_id])
            logger.info(f"[AUDIO] 当前已接收分片数: {current_count}/{total}")

            if current_count == total:
                logger.info(f"[AUDIO] 所有分片接收完毕，开始处理音频")
                audio_buffer = self.conversation_manager.audio_buffers[conversation_id]
                audio_buffer.sort(key=lambda x: x["seq"])
                full_audio = b''.join([item["data"] for item in audio_buffer])

                del self.conversation_manager.audio_buffers[conversation_id]

                await self._handle_audio_message(websocket, full_audio, conversation_id, sample_rate)
                logger.info(f"[AUDIO] 音频处理完成")
            else:
                logger.info(f"[AUDIO] 等待剩余分片...")

        except Exception as e:
            logger.error(f"[ERROR] 处理音频分片时出错: {str(e)}", exc_info=True)
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"处理音频分片失败: {str(e)}"
            }))

    async def _handle_audio_message(self, websocket: WebSocket, audio_bytes: bytes, conversation_id: str, sample_rate: int = 16000):
        """处理完整的音频消息（增加文本净化和连接断开处理）"""
        temp_audio_path = None
        try:
            # 保存到临时文件
            logger.info(f"[AUDIO] 开始处理音频消息，长度={len(audio_bytes)}字节，conversation_id={conversation_id}")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                temp_audio_path = tmp.name
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                sf.write(temp_audio_path, audio_array, sample_rate)
            logger.info(f"[AUDIO] 临时WAV文件已保存: {temp_audio_path}")

            # ----- ASR 阶段 -----
            start_time = time.time()
            try:
                text = self.model_manager.transcribe_audio(temp_audio_path)
                logger.info(f"[ASR] 识别结果: {text}")
                logger.info(f"[ASR] 识别耗时: {time.time() - start_time:.2f}秒")
            except Exception as e:
                logger.error(f"[ASR] 识别过程中出错: {e}", exc_info=True)
                await websocket.send_text(json.dumps({"type": "error", "message": f"语音识别失败: {str(e)}"}))
                return

            if not text.strip():
                logger.warning("[ASR] 识别结果为空，无法继续")
                await websocket.send_text(json.dumps({"type": "error", "message": "未能识别到有效语音"}))
                return

            # 添加用户消息到对话历史
            self.conversation_manager.add_message(conversation_id, "user", text)

            # 获取历史消息列表
            history = self.conversation_manager.get_conversation_history(conversation_id)["messages"]

            # 获取当前角色的系统提示
            character_config = self.model_manager.config_manager.load_character(self.model_manager.current_character) or {}
            dialog_config = character_config.get("dialog", {})
            system_prompt = dialog_config.get("system_prompt", "你是一个AI助手")

            # 构造完整的消息列表：系统提示 + 历史对话
            messages = [{"role": "system", "content": system_prompt}] + history
            logger.info(f"[LLM] 消息列表长度: {len(messages)}")

            # 创建队列用于传递句子
            sentence_queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            # ----- LLM 流式生成任务（在线程池中运行） -----
            def llm_stream_task():
                try:
                    logger.info("[LLM] 开始流式生成...")
                    self.model_manager.generate_response_stream_with_history(
                        messages,
                        lambda sent: asyncio.run_coroutine_threadsafe(sentence_queue.put(sent), loop).result()
                    )
                    logger.info("[LLM] 流式生成完成")
                except Exception as e:
                    logger.error(f"[LLM] 生成过程中出错: {e}", exc_info=True)
                    asyncio.run_coroutine_threadsafe(sentence_queue.put(f"[LLM错误: {e}]"), loop).result()
                finally:
                    asyncio.run_coroutine_threadsafe(sentence_queue.put(None), loop).result()

            llm_future = loop.run_in_executor(None, llm_stream_task)

            # ----- TTS 合成与发送任务（加入文本净化） -----
            sentence_idx = 0
            full_response_sentences = []  # 用于收集原始句子（供历史记录）

            async def tts_send_task():
                nonlocal sentence_idx
                while True:
                    try:
                        sentence = await sentence_queue.get()
                    except asyncio.CancelledError:
                        logger.info("[TTS] 任务被取消")
                        break

                    if sentence is None:
                        logger.info("[TTS] 收到结束信号，停止合成")
                        break

                    # 保存原始句子（用于历史记录）
                    full_response_sentences.append(sentence)

                    # 净化文本，用于 TTS 合成
                    cleaned_sentence = self._clean_text_for_tts(sentence)
                    # 检查净化后是否还有有效文本（非标点、非空白字符）
                    if not cleaned_sentence or not re.search(r'[^\s.,!?;:，。！？；：]', cleaned_sentence):
                        logger.info(f"[TTS] 句子 {sentence_idx} 净化后无有效文本，跳过合成")
                        sentence_idx += 1
                        continue

                    logger.info(f"[TTS] 原始句子[{sentence_idx}]: {sentence[:50]}...")
                    logger.info(f"[TTS] 净化后句子[{sentence_idx}]: {cleaned_sentence[:50]}...")

                    try:
                        sr, audio = await loop.run_in_executor(
                            None,
                            self.model_manager.synthesize_speech,
                            cleaned_sentence
                        )
                        if audio is not None and len(audio) > 0:
                            encoded = self._encode_audio(audio)
                            await self._send_sentence_audio(websocket, sentence_idx, sentence, encoded, sr)
                        else:
                            logger.warning(f"[WARN] 句子 {sentence_idx} 合成音频为空，跳过发送")
                    except WebSocketDisconnect:
                        logger.info("[TTS] 客户端断开，停止发送")
                        break
                    except Exception as e:
                        logger.error(f"[TTS] 句子 {sentence_idx} 合成失败: {e}", exc_info=True)
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": f"句子 {sentence_idx} 合成失败: {str(e)}"
                            }))
                        except:
                            pass

                    sentence_idx += 1

                try:
                    await websocket.send_text(json.dumps({"type": "stream_end"}))
                except:
                    pass
                logger.info("[TTS] 所有句子发送完毕")

            tts_task = asyncio.create_task(tts_send_task())

            # 等待 LLM 任务完成
            try:
                await llm_future
            except Exception as e:
                logger.error(f"[LLM] 任务异常: {e}", exc_info=True)
                # 已经通过队列传递了None，所以TTS任务会退出
                try:
                    await websocket.send_text(json.dumps({"type": "error", "message": f"语言模型生成失败: {str(e)}"}))
                except:
                    pass

            # 等待 TTS 任务完成（但允许它被取消）
            try:
                await tts_task
            except asyncio.CancelledError:
                logger.info("[TTS] 任务被取消")
            except Exception as e:
                logger.error(f"[TTS] 任务异常: {e}", exc_info=True)

            # 将AI回复添加到历史（使用原始句子，未经净化）
            if full_response_sentences:
                full_response = ''.join(full_response_sentences)
                self.conversation_manager.add_message(conversation_id, "assistant", full_response)
                logger.info(f"[HISTORY] 添加助手回复: {full_response[:50]}...")
            else:
                logger.warning("[HISTORY] 没有生成任何回复句子")

            # 保存对话历史
            save_conversation_history(conversation_id, self.conversation_manager.get_conversation_history(conversation_id)["messages"])

        except WebSocketDisconnect:
            logger.info("[HANDLE] 客户端断开，停止处理音频")
            # 无需再发送消息，直接退出
        except Exception as e:
            logger.error(f"[ERROR] _handle_audio_message 未捕获的异常: {e}", exc_info=True)
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"处理音频消息时发生内部错误: {str(e)}"
                }))
            except:
                pass
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                logger.info(f"[CLEAN] 临时文件已删除: {temp_audio_path}")

    def _encode_audio(self, audio_array: np.ndarray) -> str:
        if audio_array.dtype == np.int16:
            audio_bytes = audio_array.tobytes()
        else:
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_int16 = (audio_array * 32767.0 + 0.5).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
        return base64.b64encode(audio_bytes).decode('utf-8')

    async def _send_sentence_audio(self, websocket: WebSocket, idx: int, text: str, audio_b64: str, sample_rate: int, chunk_size: int = 32*1024):
        total_len = len(audio_b64)
        num_chunks = (total_len + chunk_size - 1) // chunk_size
        try:
            await websocket.send_text(json.dumps({
                "type": "sentence_start",
                "idx": idx,
                "text": text,
                "total_chunks": num_chunks,
                "sample_rate": sample_rate
            }))
            logger.info(f"[SEND] 句子 {idx} 开始发送，共 {num_chunks} 分片")
            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, total_len)
                chunk = audio_b64[start:end]
                await websocket.send_text(json.dumps({
                    "type": "audio_chunk",
                    "sentence_idx": idx,
                    "seq": i,
                    "data": chunk
                }))
                await asyncio.sleep(0.01)
            await websocket.send_text(json.dumps({
                "type": "sentence_end",
                "idx": idx
            }))
            logger.info(f"[SEND] 句子 {idx} 发送完成")
        except WebSocketDisconnect:
            logger.info(f"[SEND] 句子 {idx} 发送过程中客户端断开")
            raise  # 重新抛出，让上层捕获

    async def handle_ping(self, websocket: WebSocket):
        self.last_ping_time = time.time()
        await websocket.send_text(json.dumps({"type": "pong"}))
        logger.info("[PING] 已发送pong响应")

    async def handle_history_request(self, websocket: WebSocket, data: Dict):
        action = data.get("action", "")
        conversation_id = data.get("conversation_id", "")

        if action == "list":
            histories = load_conversation_history()
            await websocket.send_text(json.dumps({
                "type": "history_list",
                "histories": histories
            }))
            logger.info("[HISTORY] 已发送历史列表")
        elif action == "load":
            history = load_conversation_history(conversation_id)
            if history:
                messages = history.get("messages", [])
                self.conversation_manager.set_conversation_history(conversation_id, messages)
                await websocket.send_text(json.dumps({
                    "type": "history_load",
                    "history": history,
                    "conversation_id": conversation_id
                }))
                logger.info(f"[HISTORY] 加载对话 {conversation_id}，共 {len(messages)} 条消息")
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "历史对话不存在"
                }))
        elif action == "save":
            title = data.get("title", None)
            history_data = self.conversation_manager.get_conversation_history(conversation_id)
            if save_conversation_history(conversation_id, history_data["messages"], title):
                await websocket.send_text(json.dumps({
                    "type": "history_saved",
                    "conversation_id": conversation_id,
                    "title": title
                }))
                logger.info(f"[HISTORY] 保存对话 {conversation_id}")
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "保存对话历史失败"
                }))
        elif action == "delete":
            if delete_conversation_history(conversation_id):
                if self.conversation_manager.current_conversation_id == conversation_id:
                    self.conversation_manager.current_conversation_id = None
                if conversation_id in self.conversation_manager.conversation_history:
                    del self.conversation_manager.conversation_history[conversation_id]
                await websocket.send_text(json.dumps({
                    "type": "history_deleted",
                    "conversation_id": conversation_id
                }))
                logger.info(f"[HISTORY] 删除对话 {conversation_id}")
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "历史对话不存在"
                }))
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "无效的历史操作"
            }))

class Server:
    def __init__(self):
        self.app = FastAPI()
        self.model_manager = ModelManager()
        self.websocket_handler = WebSocketHandler(self.model_manager)
        self.setup_routes()

        # 加载所有模型
        self.load_models()

        # 确保历史目录存在
        HISTORY_DIR.mkdir(exist_ok=True)

        # 记录系统状态
        self.log_system_status()

    def load_models(self):
        try:
            self.model_manager.load_asr_model()
            logger.info("[OK] ASR模型加载成功")
        except Exception as e:
            logger.error(f"[ERROR] ASR模型加载失败: {str(e)}", exc_info=True)

        try:
            self.model_manager.load_llm_model()
            logger.info("[OK] LLM模型加载成功")
        except Exception as e:
            logger.error(f"[ERROR] LLM模型加载失败: {str(e)}", exc_info=True)

        try:
            self.model_manager.load_tts_model()
            logger.info("[OK] TTS模型加载成功")
        except Exception as e:
            logger.error(f"[ERROR] TTS模型加载失败: {str(e)}", exc_info=True)

        current_char = self.model_manager.config_manager.get("character.current")
        if current_char:
            self.model_manager.current_character = current_char
            logger.info(f"[ROLE] 当前角色设置为: {current_char}")
        else:
            logger.warning("[WARN] 配置中未设置当前角色，将使用默认提示词")
        
    def log_system_status(self):
        logger.info("=" * 50)
        logger.info("[SYSTEM] 系统状态")
        logger.info("=" * 50)
        logger.info(f"当前角色: {self.model_manager.current_character}")
        logger.info("")
        logger.info("设备分配:")
        logger.info(f"  ASR: {self.model_manager.device_config['asr']}")
        logger.info(f"  LLM: {self.model_manager.device_config['llm']}")
        logger.info(f"  TTS: {self.model_manager.device_config['tts']}")
        logger.info("")
        logger.info("模型加载状态:")
        logger.info(f"  ASR: {'[LOADED]' if self.model_manager.asr_model else '[NOT LOADED]'}")
        logger.info(f"  LLM: {'[LOADED]' if self.model_manager.llm_model else '[NOT LOADED]'}")
        logger.info(f"  TTS: {'[LOADED]' if self.model_manager.tts_model else '[NOT LOADED]'}")
        logger.info("")
        logger.info("GPU内存:")
        if torch.cuda.is_available():
            logger.info(f"  已分配: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
            logger.info(f"  已预留: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
            logger.info(f"  总计: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.2f} MB")
        else:
            logger.info("  GPU不可用")
        logger.info("=" * 50)

    def setup_routes(self):
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_handler.handle_connection(websocket)

    def start(self):
        logger.info("[START] WebSocket服务器已启动，监听 0.0.0.0:8765")
        logger.info("[INFO] 等待Godot客户端连接...")
        logger.info("[API] 支持的API消息:")
        logger.info("  - audio: 发送音频进行处理")
        logger.info("  - audio_chunk: 分片发送音频")
        logger.info("  - new_conversation: 新建对话")
        logger.info("  - ping: 心跳检测")
        logger.info("  - status: 查询系统状态")
        logger.info("  - history_request: 对话历史管理")
        logger.info("      * action=list: 列出所有对话历史")
        logger.info("      * action=load: 加载指定对话历史")
        logger.info("      * action=save: 保存当前对话历史")
        logger.info("      * action=delete: 删除指定对话历史")

        import uvicorn
        uvicorn.run(self.app, host="0.0.0.0", port=8765)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[SYSTEM] 使用设备: {device}")

    server = Server()
    server.start()