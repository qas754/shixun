import rasterio
import numpy as np
from PIL import Image
import os

def create_rgb_image(tif_file, output_path="output_rgb.png", band_indices=(2, 1, 0)):

    # 输入验证
    if not os.path.isfile(tif_file):
        print(f"错误: 文件 {tif_file} 不存在")
        return None
    if not tif_file.lower().endswith(('.tif', '.tiff')):
        print(f"错误: 文件 {tif_file} 不是TIFF格式")
        return None

    # 打开TIFF文件
    try:
        with rasterio.open(tif_file) as src:
            print(f"波段数: {src.count}")
            if src.count < 3:
                print(f"错误: 文件 {tif_file} 只有 {src.count} 个波段，需至少3个。")
                return None
            # 读取指定波段（1-based indexing in rasterio）
            bands = src.read([i + 1 for i in band_indices])  # 读取红、绿、蓝波段
    except rasterio.errors.RasterioIOError as e:
        print(f"打开文件出错: {e}")
        return None

    # 分配波段
    try:
        red, green, blue = [band.astype(float) for band in bands]
    except IndexError as e:
        print(f"波段索引错误: {e}, 请检查波段数量或顺序。")
        return None

    # 真彩色正则化
    rgb_orign = np.dstack((red, green, blue))
    array_min, array_max = rgb_orign.min(), rgb_orign.max()

    # 防止除零错误
    if array_max == array_min:
        print("警告: 图像数据范围为0，输出图像将全为黑色")
        rgb_normalized = np.zeros_like(rgb_orign, dtype=np.uint8)
    else:
        rgb_normalized = ((rgb_orign - array_min) / (array_max - array_min)) * 255
        rgb_normalized = np.clip(rgb_normalized, 0, 255).astype(np.uint8)

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存RGB图像
    try:
        rgb_image = Image.fromarray(rgb_normalized, mode='RGB')
        rgb_image.save(output_path)
        print(f"RGB图像已保存到 {output_path}")
    except (OSError, ValueError) as e:
        print(f"保存图像出错: {e}")
        return None

    return rgb_normalized

# 示例用法
if __name__ == "__main__":
    input_tif = r"D:\tif_file\2019_1101_nofire_B2348_B12_10m_roi.tif"
    result = create_rgb_image(input_tif, band_indices=(2, 1, 0))  # B04, B03, B02
    if result is not None:
        print("处理完成")
    else:
        print("处理失败")