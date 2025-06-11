import json
import re
import os
from PIL import Image, ImageDraw

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # 验证数据格式是否为列表
            if not isinstance(data, list):
                raise ValueError("JSON文件内容不是列表格式")
            # 验证列表中的元素是否为字典
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError("列表中的元素不是字典格式")
            return data
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        return []
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式")
        return []
    except ValueError as ve:
        print(f"错误: {ve}")
        return []
    except Exception as e:
        print(f"发生未知错误: {e}")
        return []

json_path = 'tools/VRSBench/eval_result/FM9G4B-V_VRSBench_referring_eval_result.json'
data = load_json_file(json_path)

def parse_bbox_infer(bbox_str):
    # 匹配模式: 数字序列，可能包含小数点
    pattern = r'<box>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</box>'
    match = re.search(pattern, bbox_str)
    try:
        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            bbox =  [x1, y1, x2, y2]
            # 检查坐标是否在有效范围内
            if all(0 <= coord <= 1000 for coord in bbox):
                return bbox
            else:
                
                return 0
        else:
            
            return 1
    except Exception as e:
        
        return 2
    
def parse_bbox_gt(bbox_str):
    # 匹配模式: 数字序列，可能包含小数点
    pattern = r'<(\d+)><(\d+)><(\d+)><(\d+)>'
    match = re.search(pattern, bbox_str)
    try:
        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            bbox =  [x1, y1, x2, y2]
            # 检查坐标是否在有效范围内
            if all(0 <= coord <= 100 for coord in bbox):
                return bbox
            else:
                
                return 0
        else:
            
            return 1
    except Exception as e:
        
        return 2
    
data_base_dir = '/data/jr/VRSBench'
sava_path = 'tools/VRSBench/bbox_painting_result'
gt_range = 100
infer_range = 1000
for item in data:
    bbox_gt = parse_bbox_gt(item.get("ground_truth"))
    bbox_infer = parse_bbox_infer(item.get("inference_result"))
    if isinstance(bbox_gt, list) and isinstance(bbox_infer, list):
        img_base_dir = os.path.join(data_base_dir, "Images_val")
        img_name = item.get("image_id")
        img_path = os.path.join(img_base_dir, img_name)
        print(f"Processing image: {img_name}, GT bbox: {bbox_gt}, Inference bbox: {bbox_infer}")
        if os.path.exists(img_path):
            # 读取图片并绘制bbox           
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)

            # 0-range -> 0-1 -> 0-width/height 
            width, height = img.size  
            bbox_gt = [x / gt_range for x in bbox_gt]
            bbox_infer = [x / infer_range for x in bbox_infer]         
            bbox_gt = [bbox_gt[0] * width, bbox_gt[1] * height, bbox_gt[2] * width, bbox_gt[3] * height]
            bbox_infer = [bbox_infer[0] * width, bbox_infer[1] * height, bbox_infer[2] * width, bbox_infer[3] * height]
            # 绘制ground truth bbox
            draw.rectangle(bbox_gt, outline="green", width=2)
            # 绘制inference bbox
            draw.rectangle(bbox_infer, outline="red", width=2)
            # 保存或显示图片
            img.save(os.path.join(sava_path, img_name))

