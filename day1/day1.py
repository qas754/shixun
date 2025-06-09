import rasterio
import numpy as np
from PIL import Image


def shuchu(tif_file, output_path="output_rgb.png"):
    """
    Process a TIFF file to create a normalized true-color RGB image.

    Args:
        tif_file (str): Path to input TIFF file
        output_path (str): Path to save the output RGB image
    """
    try:
        with rasterio.open(tif_file) as src:
            print(f"波段数: {src.count}")
            if src.count < 3:  # Check for minimum required bands
                print(f"错误: 文件 {tif_file} 只有 {src.count} 个波段，需至少3个。")
                return None
            bands = src.read()  # 读取所有波段，形状为 (波段数, 高度, 宽度)
            if src.count < 5:  # Warn if less than 5 bands
                print(f"警告: 文件 {tif_file} 只有 {src.count} 个波段，期望5个 (B02, B03, B04, B08, B12)。")
    except Exception as e:
        print(f"打开文件出错: {e}")
        return None

    # 分配波段（假设TIFF中的波段顺序为B02, B03, B04, B08, B12）
    try:
        blue = bands[0].astype(float)  # B02 - 蓝
        green = bands[1].astype(float)  # B03 - 绿
        red = bands[2].astype(float)  # B04 - 红
        nir = bands[3].astype(float)  # B08 - 近红外
        swir = bands[4].astype(float)  # B12 - 短波红外
    except IndexError as e:
        print(f"波段索引错误: {e}, 请检查波段数量或顺序。")
        return None

    # 真彩色正则化
    rgb_orign = np.dstack((red, green, blue))
    array_min, array_max = rgb_orign.min(), rgb_orign.max()

    # 防止除零错误
    if array_max == array_min:
        rgb_normalized = np.zeros_like(rgb_orign, dtype=np.uint8)
    else:
        rgb_normalized = ((rgb_orign - array_min) / (array_max - array_min)) * 255
        rgb_normalized = np.clip(rgb_normalized, 0, 255).astype(np.uint8)

    # 创建并保存RGB图像
    try:
        rgb_image = Image.fromarray(rgb_normalized, mode='RGB')
        rgb_image.save(output_path)
        print(f"RGB图像已保存到 {output_path}")
    except Exception as e:
        print(f"保存图像出错: {e}")
        return None

    # 返回归一化后的数组
    return rgb_normalized


# 示例用法
if __name__ == "__main__":
    input_tif = r"D:\tif_file\2019_1101_nofire_B2348_B12_10m_roi.tif"
    result = shuchu(input_tif)
    if result is not None:
        print("处理完成")
    else:
        print("处理失败")