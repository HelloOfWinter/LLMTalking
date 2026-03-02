import os
import argparse
import genie_tts as genie

def convert_model(pth_path, ckpt_path, output_dir):
    """
    将 GPT-SoVITS 的 .pth 和 .ckpt 转换为 GENIE 格式的 ONNX 模型目录。
    """
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"找不到 SoVITS 权重文件: {pth_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 GPT 权重文件: {ckpt_path}")

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    print("开始转换模型...")
    print(f"  SoVITS 权重: {pth_path}")
    print(f"  GPT 权重   : {ckpt_path}")
    print(f"  输出目录   : {output_dir}")

    # 调用 GENIE 的转换函数
    genie.convert_to_onnx(
        torch_pth_path=pth_path,
        torch_ckpt_path=ckpt_path,
        output_dir=output_dir
    )

    print("转换完成！")
    print(f"生成的文件位于: {output_dir}")
    # 列出输出目录中的文件
    files = os.listdir(output_dir)
    for f in files:
        print(f"  - {f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 GPT-SoVITS 模型转换为 GENIE 格式")
    parser.add_argument("--pth", required=True, help="SoVITS 权重的 .pth 文件路径")
    parser.add_argument("--ckpt", required=True, help="GPT 权重的 .ckpt 文件路径")
    parser.add_argument("--output", required=True, help="输出文件夹路径")
    args = parser.parse_args()

    convert_model(args.pth, args.ckpt, args.output)