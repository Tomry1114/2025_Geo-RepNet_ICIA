import sys
sys.path.append('/mnt/data1_hdd/wgk/PaddleClas')

from ppcls.data.dataloader.imagenet_dataset import ImageNetDataset
import matplotlib.pyplot as plt
import cv2
import io
from PIL import Image

if __name__ == "__main__":
   

      # 替换为你自己的路径
      image_root = "/mnt/data1_hdd/wgk/PaddleClas/tr/datasets_depth/"
      cls_label_path = "/mnt/data1_hdd/wgk/PaddleClas/tr/datasets_depth/train_list.txt"
      depth_root = "/mnt/data1_hdd/wgk/PaddleClas/tr/datasets_depth/"

      # 简单 transform 测试（你也可以传实际项目里的 transform_ops 配置）
      transform_ops = [{'DecodeImage': {}}, {'ResizeImage': {'size': 224}}, {'ToCHWImage': {}}]

      # 初始化数据集
      dataset = ImageNetDataset(
          image_root=image_root,
          cls_label_path=cls_label_path,
          depth_root=depth_root,
          transform_ops=transform_ops,
          delimiter=" ",
          relabel=False
      )

      # 随便读一张图试试
      rgb_img, depth_img, label = dataset[0]

      print(f"RGB shape: {rgb_img.shape}")       # 应为 [3, H, W]
      print(f"Depth shape: {depth_img.shape}")   # 应为 [1, H, W]
      print(f"Label: {label}")


