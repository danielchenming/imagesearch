from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    fe = FeatureExtractor()

    img_dir = Path("./static/img")
    feature_dir = Path("./static/feature")
    
    img_paths = sorted(img_dir.glob("*.jpg"))
    
    for img_path in img_paths:
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
