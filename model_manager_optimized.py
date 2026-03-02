import sys
import gc
import torch
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
import warnings
import re
import os

sys.path.insert(0, str(Path(__file__).parent))

from config import ConfigManager


class APIClient:
    """统一的API客户端，支持OpenAI兼容格式"""

    def __init__(self, api_type: str, api_key: str, api_base: str, model_name: str):
        self.api_type = api_type.lower()
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        self.model_name = model_name

        # 根据类型设置默认API地址
        if self.api_type == "deepseek" and not self.api_base:
            self.api_base = "https://api.deepseek.com/v1"
        elif self.api_type == "zhipu" and not self.api_base:
            self.api_base = "https://open.bigmodel.cn/api/paas/v4"
        elif self.api_type == "openai" and not self.api_base:
            self.api_base = "https://api.openai.com/v1"

    def generate(self, messages, temperature=0.7, max_tokens=300, top_p=0.9):
        """非流式生成（保留原方法，用于兼容旧代码）"""
        headers = {"Content-Type": "application/json"}
        if self.api_type == "zhipu":
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[API ERROR] 调用失败: {e}")
            return f"[API错误: {e}]"

    def generate_stream(self, messages, temperature=0.7, max_tokens=300, top_p=0.9):
        """流式生成，返回一个生成器，逐块产生文本内容"""
        headers = {"Content-Type": "application/json"}
        if self.api_type == "zhipu":
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            # 标准 OpenAI 格式：choices[0].delta.content
                            delta = chunk["choices"][0]["delta"]
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"[API] 解析流式数据失败: {e}, 原始行: {line}")
                            continue
        except Exception as e:
            print(f"[API ERROR] 流式调用失败: {e}")
            yield f"[API错误: {e}]"



class ModelManager:
    """优化版模型管理器 - 支持多设备分配，仅支持Genie TTS"""
    def generate_response_stream_with_history(self, messages: List[Dict], on_sentence: Callable[[str], None]) -> None:
        """
        流式生成回复，接受完整的消息列表（包含历史），每当检测到一个完整句子时调用 on_sentence。
        支持本地模型和API模式。
        """
        if self.llm_model is None:
            raise RuntimeError("LLM模型未加载")

        character_config = self.config_manager.load_character(self.current_character) or {}
        dialog_config = character_config.get("dialog", {})
        temperature = dialog_config.get("temperature", 0.7)
        top_p = dialog_config.get("top_p", 0.9)
        max_tokens = dialog_config.get("max_tokens", 300)

        if isinstance(self.llm_model, APIClient):
            # === API 流式模式 ===
            sentence_buffer = ""
            try:
                for content_chunk in self.llm_model.generate_stream(messages, temperature, max_tokens, top_p):
                    sentence_buffer += content_chunk
                    # 检测是否出现句子结束标点（中文句号、问号、感叹号）
                    if any(p in content_chunk for p in "。！？"):
                        # 按最后一个标点分割（简单实现，可优化）
                        parts = re.split(r'([。！？])', sentence_buffer)
                        if len(parts) > 1:
                            complete = ''.join(parts[:-1]) + parts[-1]
                            if complete.strip():
                                on_sentence(complete.strip())
                            sentence_buffer = ""
                # 处理剩余未触发标点的部分
                if sentence_buffer.strip():
                    on_sentence(sentence_buffer.strip())
            except Exception as e:
                error_msg = f"[API流式错误] {e}"
                print(error_msg)
                on_sentence(error_msg)
        else:
            # === 本地模型流式模式（原实现保持不变） ===
            from transformers import TextIteratorStreamer
            from threading import Thread

            chat_text = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.llm_tokenizer(chat_text, return_tensors="pt").to(self.llm_model.device)

            streamer = TextIteratorStreamer(self.llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.05,
                pad_token_id=self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id
            )

            thread = Thread(target=self.llm_model.generate, kwargs=generation_kwargs)
            thread.start()

            sentence_buffer = ""
            for new_text in streamer:
                sentence_buffer += new_text
                if any(p in new_text for p in "。！？"):
                    # 简单分割
                    parts = re.split(r'([。！？])', sentence_buffer)
                    if len(parts) > 1:
                        complete = ''.join(parts[:-1]) + parts[-1]
                        on_sentence(complete.strip())
                        sentence_buffer = ""
            if sentence_buffer.strip():
                on_sentence(sentence_buffer.strip())
    def __init__(self, config_manager: ConfigManager = None):
        """
        初始化优化版模型管理器

        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.config

        # 模型实例
        self.asr_model = None
        self.asr_processor = None
        self.llm_model = None   # 本地模型或API客户端
        self.llm_tokenizer = None
        self.tts_model = None    # 仅 GenieTTS

        # 设备分配
        self.device_config = {
            "asr": self.config_manager.get("models.asr.device", "cuda"),
            "llm": self.config_manager.get("models.llm.device", "cuda"),
            "tts": self.config_manager.get("models.tts.device", "cpu"),  # Genie 固定为CPU? 实际由ONNX Runtime决定
        }

        # 根据显存情况自动调整设备分配
        self._auto_adjust_device_allocation()

        # 当前加载的模型信息
        self.loaded_models = {
            "asr": None,
            "llm": None,
            "tts": None
        }

        # 当前角色
        self.current_character = None

        print("[TOOL] 优化版模型管理器初始化")
        print(f"   设备分配: ASR={self.device_config['asr']}, LLM={self.device_config['llm']}, TTS={self.device_config['tts']}")

    def _auto_adjust_device_allocation(self):
        """根据显存情况自动调整设备分配，但尊重用户手动设置"""
        if not torch.cuda.is_available():
            for model_type in self.device_config:
                self.device_config[model_type] = "cpu"
            return

        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU总内存: {total_memory:.1f} GB")

        user_set_asr = self.config_manager.get("models.asr.device") is not None
        user_set_llm = self.config_manager.get("models.llm.device") is not None
        user_set_tts = self.config_manager.get("models.tts.device") is not None

        if total_memory < 10:
            print("   [WARN] 显存较小，建议将部分模型置于CPU")
            if not user_set_asr:
                self.device_config["asr"] = "cpu"
                self.config_manager.set("models.asr.device", "cpu")
                print("   ASR 自动设置为 cpu")
            if not user_set_llm:
                self.device_config["llm"] = "cuda"
                self.config_manager.set("models.llm.device", "cuda")
                print("   LLM 自动设置为 cuda")
            if not user_set_tts:
                self.device_config["tts"] = "cpu"
                self.config_manager.set("models.tts.device", "cpu")
                print("   TTS 自动设置为 cpu")
            if not (user_set_asr and user_set_llm and user_set_tts):
                self.config_manager.save()
        else:
            print("   显存充足，保持用户配置")

    def load_asr_model(self) -> bool:
        """加载ASR模型（使用较小模型）"""
        try:
            print("\n[AUDIO] 加载ASR模型...")
            model_config = self.config.get("models", {}).get("asr", {})
            model_name = model_config.get("model_name", "whisper-small")
            model_path = model_config.get("model_path", "./whisper-small")
            device = self.device_config["asr"]

            if not Path(model_path).exists() and model_name == "whisper-medium":
                if Path("./whisper-small").exists():
                    model_name = "whisper-small"
                    model_path = "./whisper-small"
                    print(f"   [WARN] whisper-medium不存在，使用whisper-small")

            print(f"   模型: {model_name}")
            print(f"   路径: {model_path}")
            print(f"   设备: {device}")

            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            self.asr_processor = WhisperProcessor.from_pretrained(model_path)
            self.asr_model = WhisperForConditionalGeneration.from_pretrained(model_path)

            if device == "cuda":
                try:
                    self.asr_model = self.asr_model.to(device)
                    print(f"   模型已移动到 {device}")
                except torch.OutOfMemoryError:
                    print(f"   [WARN] GPU显存不足，回退到CPU")
                    device = "cpu"
                    self.device_config["asr"] = "cpu"
                    self.asr_model = self.asr_model.to("cpu")
            else:
                self.asr_model = self.asr_model.to("cpu")

            self.asr_model.eval()
            self.loaded_models["asr"] = {
                "name": model_name,
                "path": model_path,
                "device": device,
                "model_size": "small" if "small" in model_name.lower() else "medium"
            }
            print(f"[OK] ASR模型加载成功")
            return True
        except Exception as e:
            print(f"[ERROR] ASR模型加载失败: {e}")
            return False

    def load_llm_model(self, character_id: str = None) -> bool:
        """加载LLM模型（支持本地模型或API模式）"""
        try:
            print("\n[LLM] 加载LLM模型...")
            if character_id is None:
                character_id = self.config_manager.get("character.current")
            if not character_id:
                print("[ERROR] 未指定角色ID")
                return False

            character_config = self.config_manager.load_character(character_id)
            if not character_config:
                print(f"[ERROR] 角色 '{character_id}' 配置不存在")
                return False

            print(f"   角色: {character_config.get('name', character_id)}")

            llm_config = self.config_manager.get("models.llm", {})
            mode = llm_config.get("mode", "local").lower()

            if mode == "api":
                api_type = llm_config.get("api_type", "openai")
                api_key = llm_config.get("api_key", "")
                api_base = llm_config.get("api_base", "")
                api_model = llm_config.get("api_model_name", "deepseek-chat")
                if not api_key:
                    print("[ERROR] API模式需要提供api_key")
                    return False
                self.llm_model = APIClient(api_type, api_key, api_base, api_model)
                self.llm_tokenizer = None
                self.loaded_models["llm"] = {
                    "character": character_id,
                    "mode": "api",
                    "api_type": api_type,
                    "api_model": api_model
                }
                print(f"[OK] LLM API客户端初始化成功 ({api_type})")
                return True
            else:
                # 本地模型模式
                llm_config_char = character_config.get("llm", {})
                base_model = llm_config_char.get("base_model", "./SLM/Qwen3-4B-Instruct-2507")
                lora_path = llm_config_char.get("lora_path", "")
                use_base_model = llm_config_char.get("use_base_model", False)
                device = self.device_config["llm"]

                print(f"   基础模型: {base_model}")
                if lora_path and not use_base_model:
                    print(f"   LoRA适配器: {lora_path}")
                print(f"   设备: {device}")

                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                from peft import PeftModel

                self.llm_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

                quantization_config = None
                if device == "cuda":
                    load_in_4bit = self.config_manager.get("models.llm.load_in_4bit", True)
                    load_in_8bit = self.config_manager.get("models.llm.load_in_8bit", False)
                    if load_in_4bit:
                        print(f"   使用4位量化加载...")
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True
                        )
                    elif load_in_8bit:
                        print(f"   使用8位量化加载...")
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

                print(f"   加载模型...")
                try:
                    if device == "cuda":
                        self.llm_model = AutoModelForCausalLM.from_pretrained(
                            base_model,
                            quantization_config=quantization_config,
                            device_map="auto" if quantization_config else {"": 0},
                            torch_dtype=torch.float16 if not quantization_config else None,
                            trust_remote_code=True
                        ).eval()
                    else:
                        self.llm_model = AutoModelForCausalLM.from_pretrained(
                            base_model,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        ).eval()
                        self.llm_model = self.llm_model.to("cpu")
                except torch.OutOfMemoryError:
                    print(f"   [WARN] GPU显存不足，尝试回退到CPU...")
                    device = "cpu"
                    self.device_config["llm"] = "cpu"
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        base_model,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    ).eval()
                    self.llm_model = self.llm_model.to("cpu")

                if lora_path and Path(lora_path).exists() and not use_base_model:
                    print(f"   加载LoRA适配器...")
                    self.llm_model = PeftModel.from_pretrained(self.llm_model, lora_path)
                    if device == "cuda" and not hasattr(self.llm_model, 'hf_device_map'):
                        try:
                            self.llm_model = self.llm_model.to(device)
                            print(f"   模型已移动到 {device}")
                        except Exception as e:
                            print(f"   移动模型到GPU失败: {e}")

                self.current_character = character_id
                self.loaded_models["llm"] = {
                    "character": character_id,
                    "base_model": base_model,
                    "lora_path": lora_path if lora_path and not use_base_model else None,
                    "device": device,
                    "quantized": quantization_config is not None,
                    "mode": "local"
                }
                print(f"[OK] LLM本地模型加载成功")
                param_count = sum(p.numel() for p in self.llm_model.parameters())
                print(f"   模型参数量: {param_count:,}")
                if device == "cuda" and torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"   GPU内存使用: {memory_allocated:.2f} GB")
                    print(f"   GPU内存预留: {memory_reserved:.2f} GB")
                return True
        except Exception as e:
            print(f"[ERROR] LLM模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_tts_model(self, character_id: str = None) -> bool:
        """
        加载TTS模型，仅支持Genie引擎
        """
        try:
            print("\n[TTS] 加载TTS模型...")
            if character_id is None:
                character_id = self.config_manager.get("character.current")
            if not character_id:
                print("[ERROR] 未指定角色ID")
                return False

            character_config = self.config_manager.load_character(character_id)
            if not character_config:
                print(f"[ERROR] 角色 '{character_id}' 配置不存在")
                return False

            print(f"   角色: {character_config.get('name', character_id)}")

            # 获取角色特定的TTS配置（必须包含genie引擎字段）
            tts_config = character_config.get("tts", {})
            engine = tts_config.get("engine", "genie")  # 默认强制为genie
            if engine != "genie":
                print(f"[ERROR] 不支持的TTS引擎: {engine}，仅支持 genie")
                return False

            onnx_dir = tts_config.get("onnx_dir")
            language = tts_config.get("language", "zh")
            if not onnx_dir:
                print("[ERROR] Genie 引擎需要 onnx_dir 配置")
                return False

            # 语言代码转换
            if language.lower() in ['zh', 'chinese']:
                genie_lang = 'Chinese'
            elif language.lower() in ['en', 'english']:
                genie_lang = 'English'
            elif language.lower() in ['jp', 'japanese']:
                genie_lang = 'Japanese'
            else:
                genie_lang = 'Chinese'

            from genie_adapter import GenieTTS

            self.tts_model = GenieTTS(
                character_id=character_id,
                onnx_dir=onnx_dir,
                language=genie_lang,
                device=self.device_config["tts"],
                is_half=self.config_manager.get("hardware.use_half_precision", False)
            )

            # 设置参考音频
            ref_audio = tts_config.get("ref_audio")
            ref_text = tts_config.get("ref_text", "")
            ref_lang = tts_config.get("ref_lang", language)
            if ref_audio:
                ref_audio_path = os.path.abspath(ref_audio)
                self.tts_model.set_ref_audio(ref_audio_path, ref_text, ref_lang)
            else:
                print("[WARN] Genie 引擎未指定参考音频，可能无法工作")

            self.loaded_models["tts"] = {
                "character": character_id,
                "engine": "genie",
                "device": self.device_config["tts"],
                "onnx_dir": onnx_dir
            }
            print(f"[OK] TTS模型加载成功")
            return True

        except Exception as e:
            print(f"[ERROR] TTS模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_all_models(self, character_id: str = None) -> bool:
        """加载所有模型（带智能调度）"""
        print("\n" + "=" * 60)
        print("[SYSTEM] 智能加载所有模型")
        print("=" * 60)

        success = True

        print("\n1. 加载ASR模型...")
        if not self.load_asr_model():
            print("   [WARN] ASR模型加载失败，尝试继续...")
            success = False

        print("\n2. 加载LLM模型...")
        if not self.load_llm_model(character_id):
            print("   [WARN] LLM模型加载失败，尝试调整策略...")
            llm_config = self.config_manager.get("models.llm", {})
            if llm_config.get("mode", "local").lower() == "local":
                print("   尝试将LLM移到CPU...")
                self.device_config["llm"] = "cpu"
                self.config_manager.set("models.llm.device", "cpu")
                self.config_manager.save()
                if not self.load_llm_model(character_id):
                    print("   [ERROR] LLM模型在CPU上也加载失败")
                    success = False
            else:
                print("   [ERROR] API模式加载失败，无法回退到CPU")
                success = False

        print("\n3. 加载TTS模型...")
        if not self.load_tts_model(character_id):
            print("   [WARN] TTS模型加载失败")
            success = False

        print("\n" + "=" * 60)
        if success:
            print("[OK] 所有模型加载完成!")
        else:
            print("[WARN] 部分模型加载失败")
        print("=" * 60)

        return success

    def unload_model(self, model_type: str):
        """卸载指定类型的模型"""
        try:
            if model_type == "asr":
                if self.asr_model is not None:
                    del self.asr_model
                    self.asr_model = None
                if self.asr_processor is not None:
                    del self.asr_processor
                    self.asr_processor = None
                self.loaded_models["asr"] = None
                print(f"[OK] ASR模型已卸载")
            elif model_type == "llm":
                if self.llm_model is not None:
                    del self.llm_model
                    self.llm_model = None
                if self.llm_tokenizer is not None:
                    del self.llm_tokenizer
                    self.llm_tokenizer = None
                self.loaded_models["llm"] = None
                print(f"[OK] LLM模型已卸载")
            elif model_type == "tts":
                if self.tts_model is not None:
                    del self.tts_model
                    self.tts_model = None
                self.loaded_models["tts"] = None
                print(f"[OK] TTS模型已卸载")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"[ERROR] 卸载模型失败: {e}")

    def transcribe_audio(self, audio_path: str) -> str:
        """语音识别"""
        if not self.asr_model or not self.asr_processor:
            print("[ERROR] ASR模型未加载")
            return ""
        try:
            import torchaudio
            import torchaudio.functional as F
            print(f"[RECOG] 识别音频: {audio_path}")
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                print(f"   重采样: {sample_rate}Hz -> 16000Hz")
                waveform = F.resample(waveform, sample_rate, 16000)
                sample_rate = 16000
            inputs = self.asr_processor(
                waveform.squeeze().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            device = self.loaded_models["asr"]["device"] if self.loaded_models["asr"] else "cpu"
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                predicted_ids = self.asr_model.generate(inputs["input_features"])
            transcription = self.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"[OK] 识别结果: {transcription[:50]}...")
            return transcription
        except Exception as e:
            print(f"[ERROR] 语音识别失败: {e}")
            return ""

    def generate_response(self, text: str, max_tokens: int = None) -> str:
        """生成文本回复（非流式，备用）"""
        if self.llm_model is None:
            print("[ERROR] LLM模型未加载")
            return ""
        try:
            character_config = self.config_manager.load_character(self.current_character) or {}
            dialog_config = character_config.get("dialog", {})
            system_prompt = dialog_config.get("system_prompt", "你是一个AI助手")
            temperature = dialog_config.get("temperature", 0.7)
            top_p = dialog_config.get("top_p", 0.9)
            max_tokens = max_tokens or dialog_config.get("max_tokens", 300)

            print(f"[THINK] 生成回复: {text[:50]}...")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]

            if isinstance(self.llm_model, APIClient):
                response = self.llm_model.generate(messages, temperature, max_tokens, top_p)
                return response
            else:
                chat_text = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_device = self.llm_model.device
                inputs = self.llm_tokenizer(chat_text, return_tensors="pt").to(model_device)
                with torch.no_grad():
                    outputs = self.llm_model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        repetition_penalty=1.05,
                        pad_token_id=self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id
                    )
                response = self.llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
                return response
        except Exception as e:
            print(f"[ERROR] 文本生成失败: {e}")
            return ""

    def generate_response_stream(self, text: str, on_sentence: Callable[[str], None]) -> None:
        """
        流式生成回复，每当检测到一个完整句子时调用 on_sentence。
        支持本地模型和API模式。
        """
        if self.llm_model is None:
            raise RuntimeError("LLM模型未加载")

        character_config = self.config_manager.load_character(self.current_character) or {}
        dialog_config = character_config.get("dialog", {})
        system_prompt = dialog_config.get("system_prompt", "你是一个AI助手")
        temperature = dialog_config.get("temperature", 0.7)
        top_p = dialog_config.get("top_p", 0.9)
        max_tokens = dialog_config.get("max_tokens", 300)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        if isinstance(self.llm_model, APIClient):
            # API模式通常不支持流式，这里简化处理：一次性返回后用标点分割
            full_response = self.llm_model.generate(messages, temperature, max_tokens, top_p)
            sentences = re.split(r'([。！？，,.])', full_response)
            buffer = ""
            for part in sentences:
                buffer += part
                if part in "。！？，.,":
                    if buffer.strip():
                        on_sentence(buffer.strip())
                    buffer = ""
            if buffer.strip():
                on_sentence(buffer.strip())
        else:
            from transformers import TextIteratorStreamer
            from threading import Thread

            chat_text = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.llm_tokenizer(chat_text, return_tensors="pt").to(self.llm_model.device)

            streamer = TextIteratorStreamer(self.llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.05,
                pad_token_id=self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id
            )

            thread = Thread(target=self.llm_model.generate, kwargs=generation_kwargs)
            thread.start()

            sentence_buffer = ""
            for new_text in streamer:
                sentence_buffer += new_text
                if any(p in new_text for p in "。！？"):
                    # 按最后一个标点分割
                    sentences = re.split(r'([。！？])', sentence_buffer)
                    if len(sentences) > 1:
                        complete = ''.join(sentences[:-1]) + sentences[-1]
                        on_sentence(complete.strip())
                        sentence_buffer = ""
            if sentence_buffer.strip():
                on_sentence(sentence_buffer.strip())

    def synthesize_speech(self, text: str, output_path: str = None) -> Tuple[int, any]:
        """语音合成（仅Genie），失败返回 (None, None)"""
        if not self.tts_model:
            print("[ERROR] TTS模型未加载")
            return None, None
        try:
            print(f"[SYNTH] 合成语音: {text[:50]}...")
            sr, audio = self.tts_model.synthesize(text)
            if output_path and sr is not None and audio is not None and len(audio) > 0:
                self.tts_model.save_audio(audio, sr, output_path)
                print(f"[OK] 音频已保存: {output_path}")
            if sr is not None and audio is not None:
                print(f"[OK] 合成成功，采样率: {sr}Hz，长度: {len(audio)/sr:.2f}秒")
            else:
                print("[WARN] 合成音频为空")
            return sr, audio
        except Exception as e:
            print(f"[ERROR] 语音合成失败: {e}")
            return None, None

    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        status = {
            "current_character": self.current_character,
            "device_config": self.device_config,
            "models_loaded": {},
            "memory_info": {}
        }
        for model_type, model_info in self.loaded_models.items():
            status["models_loaded"][model_type] = model_info is not None
        if torch.cuda.is_available():
            status["memory_info"]["gpu"] = {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        status["model_details"] = self.loaded_models
        return status

    def print_status(self):
        """打印状态信息"""
        status = self.get_status()
        print("\n" + "=" * 60)
        print("[STATUS] 优化版模型管理器状态")
        print("=" * 60)
        print(f"当前角色: {status['current_character']}")
        print("\n设备分配:")
        for model_type, device in self.device_config.items():
            print(f"  {model_type.upper()}: {device}")
        print("\n模型加载状态:")
        for model_type, loaded in status["models_loaded"].items():
            status_icon = "[OK]" if loaded else "[ERROR]"
            model_info = self.loaded_models[model_type]
            device_info = f" ({model_info['device']})" if loaded and 'device' in model_info else ""
            mode_info = f" [模式: {model_info.get('mode')}]" if loaded and 'mode' in model_info else ""
            print(f"  {status_icon} {model_type.upper()}: {'已加载' if loaded else '未加载'}{device_info}{mode_info}")
        if "gpu" in status["memory_info"]:
            mem = status["memory_info"]["gpu"]
            print(f"\nGPU内存:")
            print(f"  已分配: {mem['allocated_gb']:.2f} GB")
            print(f"  已预留: {mem['reserved_gb']:.2f} GB")
            print(f"  总计: {mem['total_gb']:.1f} GB")
        print("=" * 60)