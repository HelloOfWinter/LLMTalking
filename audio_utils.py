# audio_utils.py
"""
音频处理工具函数
支持PCM格式转换和音频文件操作
"""

import base64
import wave
import numpy as np
from pathlib import Path
import io
import logging

logger = logging.getLogger(__name__)


def save_pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, 
                    channels: int = 1, output_path: str = "output.wav") -> bool:
    """
    将PCM数据保存为WAV文件
    
    Args:
        pcm_data: 原始PCM字节数据
        sample_rate: 采样率
        channels: 声道数
        output_path: 输出文件路径
        
    Returns:
        bool: 是否保存成功
    """
    try:
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit = 2字节
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        
        logger.info(f"✅ WAV文件已保存: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 保存WAV文件失败: {e}")
        return False


def load_wav_to_pcm(wav_path: str) -> tuple:
    """
    从WAV文件加载PCM数据
    
    Args:
        wav_path: WAV文件路径
        
    Returns:
        tuple: (pcm_bytes, sample_rate, channels)
    """
    try:
        with wave.open(wav_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            n_frames = wav_file.getnframes()
            pcm_bytes = wav_file.readframes(n_frames)
        
        logger.info(f"✅ 加载WAV文件: {wav_path}, 采样率: {sample_rate}, 声道: {channels}")
        return pcm_bytes, sample_rate, channels
        
    except Exception as e:
        logger.error(f"❌ 加载WAV文件失败: {e}")
        return None, None, None


def pcm_to_base64(pcm_bytes: bytes) -> str:
    """
    将PCM字节数据转换为Base64字符串
    
    Args:
        pcm_bytes: PCM字节数据
        
    Returns:
        str: Base64编码的字符串
    """
    try:
        return base64.b64encode(pcm_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"❌ PCM转Base64失败: {e}")
        return ""


def base64_to_pcm(base64_str: str) -> bytes:
    """
    将Base64字符串转换为PCM字节数据
    
    Args:
        base64_str: Base64编码的字符串
        
    Returns:
        bytes: PCM字节数据
    """
    try:
        return base64.b64decode(base64_str)
    except Exception as e:
        logger.error(f"❌ Base64转PCM失败: {e}")
        return b""


def numpy_to_pcm(audio_array: np.ndarray) -> bytes:
    """
    将numpy音频数组转换为PCM字节数据
    
    Args:
        audio_array: 浮点音频数组 (-1.0 到 1.0)
        
    Returns:
        bytes: PCM字节数据 (16-bit)
    """
    try:
        # 转换为16-bit PCM
        audio_int16 = (audio_array * 32768.0).astype(np.int16)
        
        # 转换为字节
        return audio_int16.tobytes()
        
    except Exception as e:
        logger.error(f"❌ numpy转PCM失败: {e}")
        return b""


def pcm_to_numpy(pcm_bytes: bytes, dtype: type = np.float32) -> np.ndarray:
    """
    将PCM字节数据转换为numpy数组
    
    Args:
        pcm_bytes: PCM字节数据
        dtype: 输出数据类型
        
    Returns:
        np.ndarray: 音频数组
    """
    try:
        # 转换为int16数组
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        
        # 转换为目标类型
        if dtype == np.float32:
            return audio_int16.astype(np.float32) / 32768.0
        else:
            return audio_int16
            
    except Exception as e:
        logger.error(f"❌ PCM转numpy失败: {e}")
        return np.array([])


def resample_audio(audio_array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    重采样音频（简单实现）
    
    Args:
        audio_array: 音频数组
        orig_sr: 原始采样率
        target_sr: 目标采样率
        
    Returns:
        np.ndarray: 重采样后的音频数组
    """
    try:
        # 计算重采样因子
        ratio = target_sr / orig_sr
        
        # 简单线性插值重采样
        orig_length = len(audio_array)
        new_length = int(orig_length * ratio)
        
        # 生成新的采样点
        orig_indices = np.arange(orig_length)
        new_indices = np.linspace(0, orig_length - 1, new_length)
        
        # 线性插值
        resampled = np.interp(new_indices, orig_indices, audio_array)
        
        logger.info(f"🔄 重采样: {orig_sr}Hz -> {target_sr}Hz")
        
        return resampled
        
    except Exception as e:
        logger.error(f"❌ 重采样失败: {e}")
        return audio_array


def convert_mono_to_stereo(audio_array: np.ndarray) -> np.ndarray:
    """
    将单声道音频转换为立体声
    
    Args:
        audio_array: 单声道音频数组
        
    Returns:
        np.ndarray: 立体声音频数组
    """
    try:
        return np.column_stack((audio_array, audio_array))
    except Exception as e:
        logger.error(f"❌ 单声道转立体声失败: {e}")
        return audio_array


def test_audio_conversion():
    """测试音频转换功能"""
    print("🧪 测试音频转换功能...")
    
    # 生成测试音频信号
    sample_rate = 16000
    duration = 1.0  # 1秒
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波
    
    # 测试转换流程
    pcm_bytes = numpy_to_pcm(test_audio)
    print(f"✅ PCM字节长度: {len(pcm_bytes)}")
    
    base64_str = pcm_to_base64(pcm_bytes)
    print(f"✅ Base64字符串长度: {len(base64_str)}")
    
    pcm_back = base64_to_pcm(base64_str)
    print(f"✅ 恢复的PCM长度: {len(pcm_back)}")
    
    audio_back = pcm_to_numpy(pcm_back)
    print(f"✅ 恢复的音频形状: {audio_back.shape}")
    
    print("✅ 音频转换测试完成")


if __name__ == "__main__":
    # 运行测试
    test_audio_conversion()