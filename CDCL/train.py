import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import backbone
from model.segnet import Extractor, Classifier, Projector  
from model.losses import CrossEntropy, EntropyMinimization, ConsistencyWeight, ContrastiveLoss  
from model.ema_model import CDCL_ema
from model.model import CDCL
import glob
import os
from utils import unlabel_data_load,label_data_load
from dataset_my import two_dataset, three_dataset
from tqdm import tqdm
import numpy as np

def dice_coefficient(pred, target, smooth=1e-5):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()

def data_load(labeled_dir,unlabeled_dir, batchsize_train=4, batchsize_val=4):
    
    train_unlabeled_img_files,val_unlabeled_img_files,val_unlabeled_mask_files = unlabel_data_load(unlabeled_dir)
    train_labeled_img_files,train_labeled_mask_files = label_data_load(labeled_dir)
    
    silver_dataset = two_dataset(val_unlabeled_img_files,val_unlabeled_mask_files,is_val=True)
    
    train_main_dataset = three_dataset(train_labeled_img_files,train_labeled_mask_files,train_unlabeled_img_files)
    
    val_silver_loader = DataLoader(silver_dataset, batch_size=batchsize_val, shuffle=False, num_workers=4)

    train_main_loader = DataLoader(train_main_dataset, batch_size=batchsize_train, shuffle=True, num_workers=4)
    
    return val_silver_loader,train_main_loader

    
    
# 定义 EMA 更新函数
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.extractor.parameters(), model.extractor.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    for ema_param, param in zip(ema_model.projector.parameters(), model.projector.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    for ema_param, param in zip(ema_model.classifier.parameters(), model.classifier.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    

# 定义选项类
class Options:
    def __init__(self):
        self.mode = 'semi'
        self.ignore_index = 255
        self.reduction = 'mean'
        self.epoch_semi = 10            # 前20个epoch只进行监督学习，从第21个epoch开始进行半监督学习
        self.in_dim = 512               # 特征提取器输出的通道数
        self.out_dim = 128              # 投影器输出的维度
        self.downsample = 1             # 投影器的降采样率（无降采样）
        self.capacity = 5000            # 特征库的最大容量
        self.count = 0                  # 当前特征库的计数，初始化为0
        self.feature_bank = []          # 特征库，存储特征向量
        self.label_bank = []            # 标签库，存储伪标签
        self.FC_bank = []               # 存储特征中心向量
        self.patch_num = 8              # 每个投影特征块的数量，将特征图划分为8x8的块
        self.bdp_threshold = 0.2        # 对比损失的下限阈值
        self.fdp_threshold = 0.8        # 对比损失的上限阈值
        self.weight_contr = 0.1         # 对比损失的权重
        self.weight_ent = 0.01          # 熵最小化损失的权重
        self.weight_cons = 1.0          # 一致性损失的权重
        self.max_epoch = 100            # 最大训练轮次数
        self.ramp = 5                   # 一致性权重的增长参数
        self.threshold = 0.95           # 置信度掩码的阈值
        self.temp = 0.5                 # 对比损失的温度参数


opts = Options()

# 初始化模型和 EMA 模型
model = CDCL(opts).cuda()
ema_model = CDCL_ema(opts).cuda()

# 定义优化器
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)


labeled_dir = '/media/Data/spx/train_mouse_brain_project/CELL_SEG_DATASET0718'
unlabeled_dir = '/media/Data/spx/train_mouse_brain_project/train_he_162/cut/A02078C4_after_tc_regist'

val_silver_loader,train_main_loader = data_load(labeled_dir,unlabeled_dir, batchsize_train=4, batchsize_val=4)


# 模拟训练过程 需要加上算acc，dice
num_epochs = 50
alpha = 0.99
train_dice_scores = []
val_dice_scores = []
for epoch in range(num_epochs):
    model.train()
    ema_model.train()
    total_train_loss = 0
    
    for batch_idx, ((x_l, target), (x_ul, _)) in enumerate(zip(labeled_loader, unlabeled_loader)):
        x_l, target = x_l.cuda(), target.cuda()
        x_ul = x_ul.cuda()

        input_var = torch.autograd.Variable(x_l)
        target_var = torch.autograd.Variable(target)
        ema_input_var = torch.autograd.Variable(x_ul, volatile=True)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(opts['ignore_index']).sum()
        assert labeled_minibatch_size > 0

        ema_model_out = ema_model(ema_input_var)
        model_out = model(input_var)

        if epoch < opts['epoch_semi']:
            # 仅有监督学习
            loss_sup, output = model(x_l=input_var, y_l=target_var)
            loss = loss_sup
        else:
            # 半监督学习
            x_ul = {'strong_aug': x_ul}
            proj_ul_ema, z_ul_ema = ema_model(x_ul={'weak_aug': x_ul})

            loss, output = model(x_l=input_var, y_l=target_var, x_ul=x_ul, epoch=epoch, proj_ul_ema=proj_ul_ema, z_ul_ema=z_ul_ema)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            update_ema_variables(model, ema_model, opts['ema_decay'], global_step)

            total_train_loss += loss.item()
            
            # 计算Dice系数
            pred = torch.argmax(output, dim=1)
            dice_score = dice_coefficient(pred, y_l)
            train_dice_scores.append(dice_score)
            
    avg_train_loss = total_train_loss / len(train_main_loader)
    avg_train_dice_score = np.mean(train_dice_scores)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}, Train Dice Score: {avg_train_dice_score}")

    
    
print("Training complete.")
