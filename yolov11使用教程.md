# 1.数据集构建

## 1.不同数据集转化

### 1.coco数据集构建

#### 1.先使用labelme进行标注

#### 2.使用以下代码将数据转化成coco数据集

```python
import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================== 参数解析 ==================
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default="C:\\Users\\lenovo\\Desktop\\新建文件夹\\labelme_result",
                    help="根目录路径（包含所有labelme标注的json文件）")
parser.add_argument('--save_path', type=str, default="C:\\Users\\lenovo\\Desktop\\新建文件夹\\coco.json",
                    help="保存路径（不划分数据集时使用）")
parser.add_argument('--random_split', action='store_true', default=True,
                    help="启用随机划分数据集（默认比例8:1:1）")
args = parser.parse_args()


# ================== 数据集划分函数 ==================
def split_dataset(data, test_size=0.1, val_size=0.1):
    """
    划分数据集为train/val/test (比例 8:1:1)
    """
    # 第一次划分：分出训练集
    train_data, temp_data = train_test_split(
        data,
        test_size=test_size + val_size,
        random_state=42
    )

    # 第二次划分：从剩余数据中分出验证集和测试集
    val_data, test_data = train_test_split(
        temp_data,
        test_size=test_size / (test_size + val_size),
        random_state=42
    )

    return train_data, val_data, test_data


# ================== 主转换函数 ==================
def labelme2coco():
    # 初始化数据结构
    coco_template = {
        "info": {"description": "COCO Dataset"},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    # 收集所有标注数据和类别
    all_data = []
    category_map = {}
    image_id_map = {}

    print("正在扫描标注文件...")
    # 遍历根目录下的所有json文件
    # 遍历根目录下的所有json文件
    for idx, filename in enumerate(tqdm(os.listdir(args.root_dir))):
        if filename.endswith(".json"):
            filepath = os.path.join(args.root_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # 记录图像路径映射
                    image_id_map[data["imagePath"]] = idx

                    # 收集类别信息
                    for shape in data["shapes"]:
                        label = shape["label"]
                        if label not in category_map:
                            # 新增颜色相关字段
                            category_map[label] = {
                                "id": len(category_map),
                                "keypoints": [f"point_{i + 1}" for i in range(len(shape["points"]))],
                                "skeleton": [[i + 1, (i + 1) % len(shape["points"]) + 1] for i in
                                             range(len(shape["points"]))],
                                "colors": {},  # 颜色名称到ID的映射
                                "color_list": []  # 颜色名称顺序列表
                            }
                        # 处理颜色信息
                        flags = shape.get("flags", {})
                        for flag_key in flags:
                            if flag_key.startswith("color_") and flags[flag_key]:
                                color_name = flag_key[6:]  # 去除"color_"前缀
                                # 如果颜色不存在则添加
                                if color_name not in category_map[label]["colors"]:
                                    color_id = len(category_map[label]["color_list"])
                                    category_map[label]["colors"][color_name] = color_id
                                    category_map[label]["color_list"].append(color_name)
                        # ...其他处理...
                    all_data.append(data)
            except Exception as e:
                print(f"错误加载文件 {filename}: {str(e)}")
                continue

    # 构建categories
    for label, info in category_map.items():
        # 转换颜色信息为列表格式
        color_info = [{"id": idx, "name": name} for idx, name in enumerate(info["color_list"])]
        coco_template["categories"].append({
            "id": info["id"],
            "name": label,
            "supercategory": "object",
            "keypoints": info["keypoints"],
            "skeleton": info["skeleton"],
            "colors": color_info  # 新增颜色字段
        })

    # 数据集划分
    if args.random_split:
        train_data, val_data, test_data = split_dataset(all_data)
        datasets = {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
    else:
        datasets = {"all": all_data}

    # 处理每个数据集划分
    for phase, data in datasets.items():
        coco_data = coco_template.copy()
        coco_data["images"] = []
        coco_data["annotations"] = []
        ann_id = 0

        print(f"正在处理 {phase} 数据集...")
        for img_data in tqdm(data):
            # 构建image信息
            image_info = {
                "id": image_id_map[img_data["imagePath"]],
                "file_name": os.path.basename(img_data["imagePath"]),
                "width": img_data["imageWidth"],
                "height": img_data["imageHeight"],
                "license": 0,
                "date_captured": ""
            }
            coco_data["images"].append(image_info)

            # 处理标注
            for shape in img_data["shapes"]:
                points = np.array(shape["points"])
                # 转换关键点格式 [x1,y1,v1,x2,y2,v2,...]
                keypoints = []
                for x, y in points:
                    keypoints.extend([float(x), float(y), 2])  # v=2表示可见

                # 计算边界框
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                bbox = [
                    float(np.min(x_coords)),  # x_min
                    float(np.min(y_coords)),  # y_min
                    float(np.max(x_coords) - np.min(x_coords)),  # width
                    float(np.max(y_coords) - np.min(y_coords))  # height
                ]

                # 计算多边形面积（使用shoelace公式）
                area = 0.5 * abs(np.sum(
                    x_coords * np.roll(y_coords, 1) -
                    np.roll(x_coords, 1) * y_coords)
                )
                # 处理颜色信息（新增）
                color_id = -1  # 默认值
                flags = shape.get("flags", {})
                for flag_key in flags:
                    if flag_key.startswith("color_") and flags[flag_key]:
                        color_name = flag_key[6:]
                        if color_name in category_map[label]["colors"]:
                            color_id = category_map[label]["colors"][color_name]
                            break
                # 构建annotation
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": image_info["id"],
                    "category_id": category_map[shape["label"]]["id"],
                    "color_id": color_id,  # 新增颜色ID字段
                    "segmentation": [points.flatten().tolist()],
                    "area": float(area),
                    "bbox": bbox,
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "num_keypoints": len(points)
                })
                ann_id += 1

                # 保存结果（修正保存逻辑）
                output_path = args.save_path
                if args.random_split:
                    base, ext = os.path.splitext(args.save_path)
                    output_path = f"{base}_{phase}{ext}"

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(coco_data, f, indent=2)
                print(f"已保存 {phase} 数据集到 {output_path}")

if __name__ == "__main__":
    labelme2coco()
    
```



### 2.coco-pose.yaml用yolov11训练的数据集

#### 1.先使用labelme进行标注

#### 2.用代码将labelme标注的数据集进行转化

```python
import os
import json
import argparse
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================== 参数解析 ==================
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default="C:\\Users\\lenovo\\Desktop\\tup\\labelme_result",
                    help="根目录路径（包含所有 labelme 标注的 json 文件）")
parser.add_argument('--image_dir', type=str, default="C:\\Users\\lenovo\\Desktop\\tup\\images",
                    help="图片所在目录（与 json 文件的 imagePath 对应）")
parser.add_argument('--random_split', action='store_true', default=True,
                    help="启用随机划分数据集（默认比例 8:1:1）")
parser.add_argument('--output_dir', type=str, default="C:\\Users\\lenovo\\Desktop\\tup\\dataset_output",
                    help="输出目录路径（用于保存分类后的数据集）")
args = parser.parse_args()


# ================== 数据集划分函数 ==================
def split_dataset(data, test_size=0.1, val_size=0.1):
    train_data, temp_data = train_test_split(
        data,
        test_size=test_size + val_size,
        random_state=42
    )
    val_data, test_data = train_test_split(
        temp_data,
        test_size=test_size / (test_size + val_size),
        random_state=42
    )
    return train_data, val_data, test_data


# ================== 创建输出目录结构 ==================
def create_output_dirs(output_dir, phases):
    for phase in phases:
        phase_dir = os.path.join(output_dir, phase)
        os.makedirs(os.path.join(phase_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(phase_dir, "labels"), exist_ok=True)


# ================== 主转换函数 ==================
def process_json_files():
    # 获取所有 json 文件名
    json_files = [f for f in os.listdir(args.root_dir) if f.endswith(".json")]

    # 划分数据集（根据文件名列表划分）
    if args.random_split:
        train_files, val_files, test_files = split_dataset(json_files)
        datasets = {"train": train_files, "val": val_files, "test": test_files}
    else:
        datasets = {"all": json_files}

    # 全局类别映射，保证类别号一致
    category_map = {}
    next_cat_id = 0

    for phase, file_list in datasets.items():
        phase_image_dir = os.path.join(args.output_dir, phase, "images")
        phase_label_dir = os.path.join(args.output_dir, phase, "labels")

        print(f"正在处理 {phase} 数据集，共 {len(file_list)} 个文件...")
        for json_file in tqdm(file_list, desc=f"Processing {phase}"):
            json_path = os.path.join(args.root_dir, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"读取文件 {json_file} 失败: {e}")
                continue

            img_filename = os.path.basename(data.get("imagePath", ""))
            src_img_path = os.path.join(args.image_dir, img_filename)

            # 获取图片尺寸
            image_width = data.get("imageWidth", None)
            image_height = data.get("imageHeight", None)
            if image_width is None or image_height is None:
                print(f"{json_file} 中缺少 imageWidth 或 imageHeight，跳过该文件。")
                continue

            label_lines = []
            for shape in data.get("shapes", []):
                label = shape.get("label")
                if label not in category_map:
                    category_map[label] = next_cat_id
                    next_cat_id += 1
                cat_id = category_map[label]

                points = np.array(shape.get("points", []))
                if points.size == 0:
                    continue

                # 计算 bbox，归一化
                x_min = np.min(points[:, 0]) / image_width
                y_min = np.min(points[:, 1]) / image_height
                x_max = np.max(points[:, 0]) / image_width
                y_max = np.max(points[:, 1]) / image_height
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                width = x_max - x_min
                height = y_max - y_min

                # 归一化关键点
                kp_list = []
                for pt in points:
                    kp_list.append(f"{pt[0] / image_width:.6f}")
                    kp_list.append(f"{pt[1] / image_height:.6f}")

                line_elements = [str(cat_id),
                                 f"{x_center:.6f}",
                                 f"{y_center:.6f}",
                                 f"{width:.6f}",
                                 f"{height:.6f}"]
                line_elements.extend(kp_list)
                label_line = " ".join(line_elements)
                label_lines.append(label_line)

            label_filename = os.path.splitext(json_file)[0] + ".txt"
            label_file_path = os.path.join(phase_label_dir, label_filename)
            with open(label_file_path, 'w') as lf:
                lf.write("\n".join(label_lines))

            if os.path.exists(src_img_path):
                dest_img_path = os.path.join(phase_image_dir, img_filename)
                try:
                    shutil.copy2(src_img_path, dest_img_path)
                except Exception as e:
                    print(f"复制图片 {img_filename} 失败: {e}")
            else:
                print(f"图片文件不存在: {src_img_path}")


if __name__ == "__main__":
    if args.random_split:
        phases = ["train", "val", "test"]
    else:
        phases = ["all"]
    create_output_dirs(args.output_dir, phases)
    process_json_files()

```







## 2.创建目录

### 1.coco-pose.yaml要求的目录格式

**ultralytics-main**在该目录下创建datasets文件夹，按照下面的结构进行分布

```
# datasets/
# └── coco8-pose/
#     ├── images/    #图像     
#     │   └── train
	  |	  |-- val
	  |   |-- test
#     └── labels/    # txt标注文件
#         ├── train
#         ├── val
#         └── test
```











# 2.基本参数配置

## 1.可以选的

### 1.**YOLO 训练/推理的全局配置**

**ultralytics-main\ultralytics\cfg\default.yaml**



### 2.模型配置

**ultralytics-main\ultralytics\cfg\models**



例如ultralytics-main\ultralytics\cfg\models\11\yolo11-pose.yaml

这个模型里面的kpt_shape: 可以进行修改

```yaml
nc: 80 # number of classes，检测类别的数量
kpt_shape: [5, 2]  # 5个关键点，每个点只有(x,y)坐标
```

## 2.必须要做的

**ultralytics-main\ultralytics\cfg\datasets**从这个文件夹下面选一个我们需要的数据集格式的.yaml文件，把它放到训练代码的同级目录(方便观察)

根据自己的需求具体修改

### 1.coco-pose.yaml

```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# COCO 2017 Keypoints dataset https://cocodataset.org by Microsoft
# Documentation: https://docs.ultralytics.com/datasets/pose/coco/
# Example usage: yolo train data=coco-pose.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── coco-pose  ← downloads here (20.1 GB)

#-----------这里就是对应的路径-------------------------------
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ./datasets/coco-pose # dataset root dir
train: annotations/coco_train.json # train images (relative to 'path') 56599 images
val: annotations/coco_val.json # val images (relative to 'path') 2346 images
test: annotations/coco_test.json # 20288 of 40670 images, submit to https://codalab.lisn.upsaclay.fr/competitions/7403

#-----------------------------------------------------------

#-----------关键点，和水平反转后对称方向--------------------------
# Keypoints
kpt_shape: [5, 2] #这个就代表着有5个关键点，并都是(x,y)的形式
flip_idx: [0,2,1,4,3] #这个代表着，当图像水平翻转后，0位置的点还是0位置的点，1位置的点变成了2位置的点以此类推
#--------------------------------------------------------


#-------------标签-------------------------------
# Classes
names:
  0: armor
#--------------------------------------------------


# Download script/URL (optional)
download: |
  from pathlib import Path

  from ultralytics.utils.downloads import download

  # Download labels
  dir = Path(yaml["path"])  # dataset root dir
  url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
  urls = [f"{url}coco2017labels-pose.zip"]
  download(urls, dir=dir.parent)
  # Download data
  urls = [
      "http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
      "http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
      "http://images.cocodataset.org/zips/test2017.zip",  # 7G, 41k images (optional)
  ]
  download(urls, dir=dir / "images", threads=3)

```



对于上面的flip_idx这个值可以利用下面代码，观察对应点反转后应该是如何对应

```python
import json
import os
from pathlib import Path
import numpy as np
import cv2

def check_annotations(json_path, image_dir):
    """检查COCO格式的标注文件"""
    print(f"开始检查标注文件: {json_path}")
    
    # 加载标注文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. 基本信息检查
    print("\n=== 基本信息 ===")
    print(f"图片数量: {len(data['images'])}")
    print(f"标注数量: {len(data['annotations'])}")
    print(f"类别数量: {len(data['categories'])}")
    
    # 2. 检查图片分辨率
    print("\n=== 图片分辨率检查 ===")
    resolutions = set()
    for img in data['images']:
        resolutions.add((img['width'], img['height']))
    print(f"数据集中的图片分辨率: {resolutions}")
    
    # 3. 检查标注格式
    print("\n=== 标注格式检查 ===")
    for i, ann in enumerate(data['annotations'][:5]):
        print(f"\n标注 {i+1}:")
        print(f"图片ID: {ann['image_id']}")
        print(f"bbox: {ann['bbox']}")
        print(f"关键点数量: {ann['num_keypoints']}")
        print(f"关键点坐标数: {len(ann['keypoints'])}")
        
        # 检查bbox是否在图片范围内
        img_info = next(img for img in data['images'] if img['id'] == ann['image_id'])
        x, y, w, h = ann['bbox']
        if x < 0 or y < 0 or x + w > img_info['width'] or y + h > img_info['height']:
            print(f"⚠️ 警告: bbox超出图片范围!")
            
    # 4. 验证图片文件
    print("\n=== 图片文件检查 ===")
    missing_files = []
    for img in data['images']:
        img_path = Path(image_dir) / img['file_name']
        if not img_path.exists():
            missing_files.append(img['file_name'])
    
    if missing_files:
        print("❌ 以下图片文件缺失:")
        for f in missing_files:
            print(f" - {f}")
    else:
        print("✅ 所有图片文件都存在")
        
    # 5. 检查关键点分布
    print("\n=== 关键点统计 ===")
    keypoints_count = [ann['num_keypoints'] for ann in data['annotations']]
    print(f"平均关键点数: {np.mean(keypoints_count):.2f}")
    print(f"最小关键点数: {min(keypoints_count)}")
    print(f"最大关键点数: {max(keypoints_count)}")

def visualize_annotations(json_path, image_dir, output_dir, num_samples=5):
    """可视化部分标注结果，包括关键点连线和顺序编号"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 随机选择几张图片进行可视化
    import random
    sample_imgs = random.sample(data['images'], min(num_samples, len(data['images'])))
    
    # 定义关键点连接关系（装甲板五个点的连接顺序）
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 外框四边形
        (0, 4), (1, 4), (2, 4), (3, 4)   # 中心点与四个角点的连接
    ]
    
    for img_info in sample_imgs:
        img_path = Path(image_dir) / img_info['file_name']
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        
        # 找到该图片的所有标注
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_info['id']]
        
        # 绘制bbox和关键点
        for ann in annotations:
            # 绘制边界框
            x, y, w, h = map(int, ann['bbox'])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 获取所有关键点坐标
            keypoints = []
            for i in range(0, len(ann['keypoints']), 3):
                x, y, v = ann['keypoints'][i:i+3]
                if v > 0:  # 只处理可见的关键点
                    keypoints.append((int(x), int(y)))
            
            # 绘制关键点之间的连线
            if len(keypoints) == 5:  # 确保有5个关键点
                for start_idx, end_idx in connections:
                    if start_idx < len(keypoints) and end_idx < len(keypoints):
                        cv2.line(img, 
                                keypoints[start_idx], 
                                keypoints[end_idx], 
                                (255, 0, 0),  # 蓝色连线
                                2)
            
            # 绘制关键点并添加编号
            for idx, point in enumerate(keypoints):
                # 绘制关键点（红色）
                cv2.circle(img, point, 5, (0, 0, 255), -1)
                # 添加关键点编号（白色背景，黑色文字）
                text_position = (point[0] + 10, point[1] - 10)
                # 先绘制白色背景矩形
                text_size = cv2.getTextSize(str(idx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(img, 
                            (text_position[0] - 2, text_position[1] - text_size[1] - 2),
                            (text_position[0] + text_size[0] + 2, text_position[1] + 2),
                            (255, 255, 255), -1)
                # 再绘制黑色文字
                cv2.putText(img, str(idx), text_position, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
        # 添加图例
        legend_y = 30
        cv2.putText(img, 'Red: Keypoints', (10, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, 'Blue: Connections', (10, legend_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(img, 'Green: Bounding Box', (10, legend_y + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, 'White: Keypoint Index', (10, legend_y + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存结果
        output_path = Path(output_dir) / f"vis_{img_info['file_name']}"
        cv2.imwrite(str(output_path), img)
        
    print(f"✅ 可视化结果已保存至: {output_dir}")

if __name__ == "__main__":
    json_path = "C:\\Users\\lenovo\\Desktop\\tup\\coco\\coco_train.json"
    image_dir = "C:\\Users\\lenovo\\Desktop\\tup\\images"
    output_dir = "D:\\tup_yolox\\visualization_results"
    
    check_annotations(json_path, image_dir)
    visualize_annotations(json_path, image_dir, output_dir)
```







# 3.使用代码训练

```python
from ultralytics import YOLO

#主要任务类型：
#-detect: 目标检测（矩形框）
#- pose: 姿态估计（关键点）
#- segment: 实例分割（像素级）
#- classify: 图像分类
#加载模型，第一个是预训练模型的位置，第二个是任务模式(这里是检测任务)
#model = YOLO("E:\\ultralytics-main\\yolo11n.pt",task="detect")

model = YOLO("E:\\ultralytics-main\\yolo11n.pt",task="pose")

#第一个参数就是训练用的数据集描述文件，epochs就是训练多少轮，workers必须要设置(windows设置成1就行)
#batch根据电脑显卡配置进行调整
results = model.train(data='./coco-pose.yaml',epochs=50,workers=1,batch=16)
```

















# 使用模型进行预测

```python
from ultralytics import YOLO

yolo = YOLO("训练好的模型",task="detect")

#测试图片
result = yolo(source="测试图片路径",show=True,save=True)
#测试视频
result = yolo.predict(source="测试视频路径",show=True,save=True)
```

这里建议在jupyter里进行

然后可以直接运行

```python
result[0]#result是列表，所以说如果测试的是图片的话，就可以只取第一个输出结果
#如果处理的视频的话，这里的意思就是输出的第一帧的输出结果
```

看到result里面的具体数据都有什么









# 导出模型为ONNX

```python
from ultralytics import YOLO

model = YOLO(r"runs\detect\train6\weights\best.pt")

model.export(
    format="onnx",
    dynamic=True,      # 动态输入尺寸 & batch
    simplify=True,     # 简化模型
    opset=17,          # ONNX opset
    imgsz=640,         # 输入分辨率
    batch=1            # 默认 batch
)

```