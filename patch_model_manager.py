# patch_model_manager.py
"""
修复model_manager_optimized.py中的特殊字符
"""

import re

def patch_model_manager(filepath: str):
    """修复model_manager_optimized.py文件中的特殊字符"""
    
    # 读取文件
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换所有特殊Unicode字符为纯文本
    replacements = {
        "🔧": "[TOOL]",
        "✅": "[OK]",
        "❌": "[ERROR]",
        "⚠️": "[WARN]",
        "🧠": "[LLM]",
        "🗣️": "[TTS]",
        "🔊": "[AUDIO]",
        "🔍": "[RECOG]",
        "💭": "[THINK]",
        "🧹": "[CLEAN]",
        "🚀": "[START]",
        "📊": "[STATUS]",
        "🔄": "[SWITCH]",
        "📨": "[RECV]",
        "📤": "[SEND]",
        "🎵": "[MUSIC]",
        "🎤": "[MIC]",
        "⏱️": "[TIME]",
        "🔗": "[CONNECT]",
        "🔌": "[DISCONNECT]",
        "📡": "[NETWORK]",
        "📝": "[NOTE]",
        "📋": "[LIST]",
        "💡": "[IDEA]",
        "💓": "[HEART]",
        "🎮": "[GAME]",
        "📱": "[PHONE]",
        "=" * 60: "=" * 60,  # 保留分隔线
    }
    
    for symbol, replacement in replacements.items():
        content = content.replace(symbol, replacement)
    
    # 移除所有其他可能的问题字符
    # 只保留ASCII字符和中文
    # content = re.sub(r'[^\x00-\x7F\u4e00-\u9fff]+', '', content)
    
    # 保存修复后的文件
    backup_path = filepath + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] 已创建备份文件: {backup_path}")
    
    # 创建修复版本
    fixed_path = filepath.replace('.py', '_fixed.py')
    with open(fixed_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] 已创建修复版本: {fixed_path}")
    
    return fixed_path


def create_simple_model_manager():
    """创建一个简化版的模型管理器"""
    simple_code = '''
# model_manager_simple.py
"""
简化版模型管理器 - 无特殊字符，适合Windows控制台
"""

import sys
import gc
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings

sys.path.insert(0, str(Path(__file__).parent))

from config import ConfigManager

class SimpleModelManager:
    """简化版模型管理器"""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.config
        
        # 模型实例
        self.asr_model = None
        self.asr_processor = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.tts_model = None
        
        # 设备配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 当前加载的模型信息
        self.loaded_models = {
            "asr": None,
            "llm": None,
            "tts": None
        }
        
        # 当前角色
        self.current_character = None
        
        print("[INIT] 模型管理器初始化")
        print(f"设备: {self.device}")
    
    def load_all_models(self, character_id: str = None) -> bool:
        """加载所有模型"""
        print("[LOAD] 开始加载所有模型")
        
        success = True
        
        # 尝试加载ASR
        try:
            print("[ASR] 加载ASR模型...")
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            self.asr_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            self.asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            self.asr_model.eval()
            self.loaded_models["asr"] = {"name": "whisper-small", "device": self.device}
            print("[OK] ASR模型加载成功")
        except Exception as e:
            print(f"[ERROR] ASR模型加载失败: {e}")
            success = False
        
        # 加载LLM
        if character_id is None:
            character_id = self.config_manager.get("character.current")
        
        try:
            print(f"[LLM] 加载LLM模型，角色: {character_id}")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # 获取角色配置
            char_config = self.config_manager.load_character(character_id)
            base_model = char_config.get("llm", {}).get("base_model", "./SLM/Qwen3-4B-Instruct-2507")
            
            self.llm_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            ).eval()
            
            self.current_character = character_id
            self.loaded_models["llm"] = {"character": character_id, "base_model": base_model}
            print("[OK] LLM模型加载成功")
        except Exception as e:
            print(f"[ERROR] LLM模型加载失败: {e}")
            success = False
        
        # 加载TTS
        try:
            print("[TTS] 加载TTS模型...")
            from direct_api import DirectTTS
            
            char_config = self.config_manager.load_character(character_id)
            tts_config = char_config.get("tts", {})
            sovits_path = tts_config.get("sovits_path", "./SoVITS_weights_v4/Ye.pth")
            gpt_path = tts_config.get("gpt_path", "./GPT_weights_v4/Ye.ckpt")
            
            self.tts_model = DirectTTS(
                sovits_path=sovits_path,
                gpt_path=gpt_path,
                version="v4",
                device=self.device,
                is_half=True
            )
            
            self.loaded_models["tts"] = {
                "character": character_id,
                "sovits_path": sovits_path,
                "gpt_path": gpt_path
            }
            print("[OK] TTS模型加载成功")
        except Exception as e:
            print(f"[ERROR] TTS模型加载失败: {e}")
            success = False
        
        if success:
            print("[OK] 所有模型加载完成")
        else:
            print("[WARN] 部分模型加载失败")
        
        return success
    
    def transcribe_audio(self, audio_path: str) -> str:
        """语音识别"""
        if not self.asr_model:
            return "[ERROR] ASR模型未加载"
        
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 简单处理：假设音频已经是16kHz单声道
            inputs = self.asr_processor(
                waveform.squeeze().numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            
            with torch.no_grad():
                predicted_ids = self.asr_model.generate(inputs["input_features"])
            
            transcription = self.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription
        except Exception as e:
            return f"[ERROR] 识别失败: {e}"
    
    def generate_response(self, text: str) -> str:
        """生成回复"""
        if not self.llm_model:
            return "[ERROR] LLM模型未加载"
        
        try:
            messages = [{"role": "user", "content": text}]
            chat_text = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = self.llm_tokenizer(chat_text, return_tensors="pt").to(self.llm_model.device)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = self.llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            return response
        except Exception as e:
            return f"[ERROR] 生成失败: {e}"
    
    def synthesize_speech(self, text: str):
        """语音合成"""
        if not self.tts_model:
            return None, None
        
        try:
            sr, audio = self.tts_model.synthesize(text)
            return sr, audio
        except Exception as e:
            print(f"[ERROR] 合成失败: {e}")
            return None, None
    
    def get_status(self):
        """获取状态"""
        return {
            "current_character": self.current_character,
            "device": self.device,
            "models_loaded": self.loaded_models
        }
'''
    
    with open('model_manager_simple.py', 'w', encoding='utf-8') as f:
        f.write(simple_code)
    
    print("[OK] 已创建简化版模型管理器: model_manager_simple.py")


if __name__ == "__main__":
    print("[PATCH] 开始修复模型管理器...")
    
    # 尝试修复现有的model_manager_optimized.py
    try:
        fixed_path = patch_model_manager("model_manager_optimized.py")
        print(f"[OK] 修复完成，使用 {fixed_path}")
    except Exception as e:
        print(f"[ERROR] 修复失败: {e}")
        print("[INFO] 创建简化版模型管理器...")
        create_simple_model_manager()