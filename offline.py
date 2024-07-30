from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
import os
import time


def convert_to_jpg_and_delete_original(img_path):
    """
    将指定路径的图片转换为jpg格式，如果已经是jpg格式则不进行转换。
    返回转换后的jpg文件路径，并删除原始的非jpg格式图片。
    """
    img = Image.open(img_path)
    if img.format != 'JPEG':
        jpg_path = img_path.with_suffix('.jpg')
        img.convert('RGB').save(jpg_path)
        print(f"Converted {img_path} to {jpg_path}")

        # 删除原始非jpg格式图片
        os.remove(img_path)
        print(f"Deleted original image: {img_path}")

        return jpg_path
    else:
        return img_path


if __name__ == '__main__':
    fe = FeatureExtractor()

    img_dir = Path("./static/img")
    feature_dir = Path("./static/feature")

    while True:
        img_paths = sorted(img_dir.glob("*"))  # 获取所有图片路径，包括非jpg格式

        for img_path in img_paths:
            if img_path.suffix.lower() not in ['.jpg', '.jpeg']:
                # 如果不是jpg格式，先转换为jpg格式并删除原始非jpg格式图片
                jpg_path = convert_to_jpg_and_delete_original(img_path)
                img_path = jpg_path

            print(f"Processing image: {img_path}")

            # 构建特征文件路径
            feature_path = feature_dir / (img_path.stem + ".npy")

            # 如果特征文件已经存在，则跳过当前图像
            if feature_path.exists():
                print(f"Feature file {feature_path} already exists. Skipping.")
                continue

            # 提取特征并保存为 numpy 数组
            try:
                image = Image.open(img_path)
                feature = fe.extract(img=image)
                np.save(feature_path, feature)
                print(f"Feature saved to {feature_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        print("Waiting for 15 seconds before the next iteration...")
        time.sleep(15)  # 暂停10秒后再次执行循环
