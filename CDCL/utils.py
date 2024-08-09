import os
from sklearn.model_selection import train_test_split

def unlabel_data_load(unlabeled_dir):

    unlabeled_img_dir = os.path.join(unlabeled_dir,'img')
    unlabeled_mask_dir = os.path.join(unlabeled_dir,'mask')
    
    # 获取所有img和mask文件的路径
    unlabeled_img_files = [os.path.join(unlabeled_img_dir, f) for f in os.listdir(unlabeled_img_dir) if f.endswith('.tif')]
    unlabeled_mask_files = [os.path.join(unlabeled_mask_dir, f) for f in os.listdir(unlabeled_mask_dir) if f.endswith('.tif')]

    # 对文件路径列表进行排序
    unlabeled_img_files.sort()
    unlabeled_mask_files.sort()
    
    # 确保两个目录中的文件数一致
    assert len(unlabeled_img_files) == len(unlabeled_mask_files), "图片和标签文件数量不匹配"
    
    # 创建文件路径的元组列表，(img_path, mask_path)
    file_pairs = [(img_file, mask_file) for img_file, mask_file in zip(unlabeled_img_files, unlabeled_mask_files)]
    
    # 划分训练集和验证集（假设验证集占20%）
    train_files, val_files = train_test_split(file_pairs, test_size=0.8, random_state=42)
    
    # 分离训练和验证集的图像和掩码文件路径
    train_unlabeled_img_files = [f[0] for f in train_files]
    val_unlabeled_img_files = [f[0] for f in val_files]
    val_unlabeled_mask_files = [f[1] for f in val_files]
    
    return train_unlabeled_img_files, val_unlabeled_img_files, val_unlabeled_mask_files
    
    
def label_data_load(labeled_dir):

    labeled_img_dir = os.path.join(labeled_dir,'img')
    labeled_mask_dir = os.path.join(labeled_dir,'mask')
    
    # 获取所有img和mask文件的路径
    labeled_img_files = [os.path.join(labeled_img_dir, f) for f in os.listdir(labeled_img_dir) if f.endswith('.tif')]
    labeled_mask_files = [os.path.join(labeled_mask_dir, f) for f in os.listdir(labeled_mask_dir) if f.endswith('.tif')]

    # 确保两个目录中的文件数一致
    assert len(labeled_img_files) == len(labeled_mask_files), "图片和标签文件数量不匹配"
    
    # 创建唯一标识符到文件路径的映射
    img_file_map = {extract_identifier_label(os.path.basename(f)): f for f in labeled_img_files}
    mask_file_map = {extract_identifier_label(os.path.basename(f)): f for f in labeled_mask_files}

    # 确保每个唯一标识符在img和mask中都有对应的文件
    assert set(img_file_map.keys()) == set(mask_file_map.keys()), "某些唯一标识符在img和mask中不匹配"

    # 获取所有唯一标识符
    identifiers = list(img_file_map.keys())
    
    # 生成完整的文件列表
    all_labeled_img_files = [img_file_map[id] for id in identifiers]
    all_labeled_mask_files = [mask_file_map[id] for id in identifiers]
    
    return all_labeled_img_files, all_labeled_mask_files

# 提取唯一标识符
def extract_identifier(filename):
    parts = filename.split('_')
    identifier = '_'.join(parts[0:5])
    return identifier

def extract_identifier_label(filename):
    if filename.endswith('-img.tif'):
        return filename.replace('-img.tif', '')
    elif filename.endswith('-mask.tif'):
        return filename.replace('-mask.tif', '')
    else:
        raise ValueError("文件名格式不正确")
    
    