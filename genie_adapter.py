# genie_adapter.py
"""
Genie TTS 适配器，实现与 DirectTTS 相同的接口。
支持 GPU 加速（通过猴子补丁修改 Genie 的 providers）。
"""

import os
import asyncio
import numpy as np
from typing import Tuple, Optional

# 尝试启用 CUDA（如果可用）
import onnxruntime
import genie_tts
import genie_tts.ModelManager as mm

if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
    mm.model_manager.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("[Genie] 已启用 CUDA 加速")
else:
    print("[Genie] CUDA 不可用，使用 CPU")

import genie_tts as genie

class GenieTTS:
    """Genie TTS 引擎适配器"""

    def __init__(self, character_id: str, onnx_dir: str, language: str,
                 device: str = "cpu", is_half: bool = False):
        """
        Args:
            character_id: 角色ID，作为 Genie 的角色名
            onnx_dir: ONNX 模型文件夹路径（绝对或相对路径）
            language: 语言代码，Genie 接受 'Japanese', 'English', 'Chinese'
            device: 设备，实际由 ONNX Runtime 控制，保留参数
            is_half: 是否使用半精度，Genie 内部已使用 FP16 权重，保留参数
        """
        self.character_id = character_id
        self.language = language
        self.sample_rate = 32000  # Genie 固定输出采样率

        # 将路径转换为绝对路径，确保 Genie 能正确找到
        onnx_dir = os.path.abspath(onnx_dir)
        if not os.path.isdir(onnx_dir):
            raise FileNotFoundError(f"ONNX 模型目录不存在: {onnx_dir}")

        # 加载角色模型
        genie.load_character(
            character_name=character_id,
            onnx_model_dir=onnx_dir,
            language=language
        )
        self.ref_audio_set = False

    def set_ref_audio(self, audio_path: str, prompt_text: str = "",
                      prompt_lang: Optional[str] = None):
        """设置参考音频"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"参考音频文件不存在: {audio_path}")
        lang = prompt_lang or self.language
        genie.set_reference_audio(
            character_name=self.character_id,
            audio_path=audio_path,
            audio_text=prompt_text,
            language=lang
        )
        self.ref_audio_set = True

    def synthesize(self, text: str, text_lang: str = "zh", **kwargs) -> Tuple[int, np.ndarray]:
        """合成语音，返回 (采样率, 音频数组)。若失败返回 (sample_rate, 空数组)"""
        if not self.ref_audio_set:
            raise RuntimeError("必须先调用 set_ref_audio 设置参考音频")

        # 预处理文本：移除可能导致 Genie 崩溃的特殊字符
        # 保留常见标点，但去除可能影响分词的奇怪符号（可根据需要调整）
        import re
        # 移除控制字符和不可见字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # 替换英文括号为中文括号（如果Genie对英文括号处理不好）
        text = text.replace('(', '（').replace(')', '）')
        # 如果文本为空，返回空音频
        if not text.strip():
            print("[Genie] 警告：输入文本为空，返回空音频")
            return self.sample_rate, np.array([], dtype=np.int16)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        chunks = []

        async def collect():
            try:
                async for chunk in genie.tts_async(
                    character_name=self.character_id,
                    text=text,
                    play=False,
                    split_sentence=True,
                    save_path=None
                ):
                    chunks.append(chunk)
            except Exception as e:
                print(f"[Genie] TTS 异步合成出错: {e}")

        try:
            loop.run_until_complete(collect())
        except Exception as e:
            print(f"[Genie] TTS 合成执行异常: {e}")
            return self.sample_rate, np.array([], dtype=np.int16)

        if not chunks:
            print("[Genie] 警告：合成音频为空")
            return self.sample_rate, np.array([], dtype=np.int16)

        audio_bytes = b''.join(chunks)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        return self.sample_rate, audio_array