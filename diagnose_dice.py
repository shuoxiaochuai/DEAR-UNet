from PIL import Image
import os

# ========== 硬编码配置 ==========
input_paths = [
    r"D:\DeepLearning CV\IDRiD\Original Images\Testing Set\IDRiD_72.jpg",
    r"D:\DeepLearning CV\MyResults2026\MCA-DSP-FTL\0.91 dice=0.483\Pred18_lab.jpg",
    r"D:\DeepLearning CV\MyResults2026\UNet\0.99 dice=0.283\Pred18_predictCAUNet.jpg",
    r"D:\DeepLearning CV\MyResults2026\MCA-UNet\0.99 dice=0.381\Pred18_predictCAUNet.jpg",
    r"D:\DeepLearning CV\MyResults2026\UNet-DSP-FTL\0.78 dice=0.337\Pred18_predictCAUNet.jpg",
    r"D:\DeepLearning CV\MyResults2026\MCA-UNet-SPA\0.99 dice=0.422\Pred18_predictCAUNet.jpg",
    r"D:\DeepLearning CV\MyResults2026\MCA-DSP-FTL\0.91 dice=0.483\Pred18_predictCAUNet.jpg"
]

output_path = r"D:\DeepLearning CV\MyResults2026\output_combined4.jpg"

# 目标尺寸与间隔
TARGET_SIZE = 640
GAP = 15
# =================================

def main():
    images = []
    for path in input_paths:
        if not os.path.exists(path):
            print(f"警告：文件不存在，已跳过 -> {path}")
            continue
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"读取图片失败：{path}，错误：{e}")

    if not images:
        print("没有有效的图片，程序退出。")
        return

    processed = []
    for img in images:
        w, h = img.size
        # 计算缩放比例，使得短边刚好等于 TARGET_SIZE（保证完全覆盖 640x640）
        scale = TARGET_SIZE / min(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        # 等比缩放
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # 居中裁剪出 640x640
        left = (new_w - TARGET_SIZE) // 2
        top = (new_h - TARGET_SIZE) // 2
        cropped = resized.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))
        processed.append(cropped)

    # 拼接总宽度
    total_width = len(processed) * TARGET_SIZE + (len(processed) - 1) * GAP
    total_height = TARGET_SIZE
    combined = Image.new('RGB', (total_width, total_height), color='white')

    x_pos = 0
    for img in processed:
        combined.paste(img, (x_pos, 0))
        x_pos += TARGET_SIZE + GAP

    combined.save(output_path)
    print(f"拼接完成！输出文件：{output_path}")

if __name__ == "__main__":
    main()