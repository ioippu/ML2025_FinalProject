"""
Ensemble Inference Script
支援多模型融合 (Ensemble) 推論，提升準確率。
原理：將多個模型的預測機率 (Softmax probabilities) 加總平均。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
import os
import argparse
import warnings
from collections import defaultdict
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import csv

# 引用 inference.py 中的共用函數
from inference import build_model, safe_rgb_loader, DEFAULT_CLASS_NAMES

warnings.filterwarnings("ignore", message="Palette images with Transparency")

parser = argparse.ArgumentParser(description='Ensemble Inference')
parser.add_argument('--data-dir', type=str, required=True, help='Data directory (ImageFolder format)')
parser.add_argument('--models', type=str, nargs='+', required=True, help='List of model architectures (e.g. swin_t vit_b_16)')
parser.add_argument('--checkpoints', type=str, nargs='+', required=True, help='List of checkpoint paths (must match order of --models)')
parser.add_argument('--num-classes', type=int, default=100, help='Number of classes')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--save-results', type=str, default='ensemble_results.csv', help='Save per-class results to CSV')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

def load_models(archs, checkpoint_paths, num_classes):
    """載入所有指定的模型"""
    models = []
    print(f'\nLoading {len(archs)} models for ensemble...')
    
    for arch, ckpt_path in zip(archs, checkpoint_paths):
        print(f'  - Model: {arch}')
        print(f'    Checkpoint: {ckpt_path}')
        
        # 建立模型架構
        net = build_model(arch, num_classes)
        
        # 載入權重
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
            if 'net' in checkpoint:
                net.load_state_dict(checkpoint['net'])
                acc = checkpoint.get('acc', 'N/A')
                print(f'    Loaded successfully (Val Acc: {acc})')
            else:
                net.load_state_dict(checkpoint)
                print('    Loaded successfully (state_dict)')
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        net = net.cuda()
        net.eval()
        models.append(net)
        
    return models

def ensemble_evaluate(models, data_dir, batch_size=32, num_workers=4):
    """對資料夾進行 Ensemble 評估"""
    
    # 圖片前處理 (所有模型都使用標準 ImageNet 設定: Resize 224 -> Normalize)
    # 注意：如果有模型使用不同的 img_size (如 384)，這裡需要針對每個模型做不同 transform
    # 目前假設所有 checkpoint 都是 224x224 訓練的
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = torchvision.datasets.ImageFolder(
        data_dir,
        transform=transform,
        loader=safe_rgb_loader
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f'\nEvaluating on: {data_dir}')
    print(f'Total images: {len(dataset)}')
    
    folder_classes = dataset.classes
    
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Ensemble Eval')
        for inputs, targets in pbar:
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            # 融合預測：將所有模型的 Softmax 機率相加
            ensemble_probs = torch.zeros(inputs.size(0), args.num_classes).cuda()
            
            for net in models:
                logits = net(inputs)
                probs = F.softmax(logits, dim=1)
                ensemble_probs += probs
            
            # 取平均 (其實不除以 N 也不影響 argmax 結果，但為了數值意義正確)
            ensemble_probs /= len(models)
            
            # 計算 Top-1
            _, pred = ensemble_probs.max(1)
            top1_correct += pred.eq(targets).sum().item()
            
            # 計算 Top-5
            _, pred_top5 = ensemble_probs.topk(5, dim=1)
            for i in range(targets.size(0)):
                if targets[i] in pred_top5[i]:
                    top5_correct += 1
            
            # 每個類別統計
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if pred[i] == targets[i]:
                    class_correct[label] += 1
            
            total += targets.size(0)
            current_acc = top1_correct / total
            pbar.set_postfix({'Ens Acc': f'{current_acc:.4f}'})
            
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    
    print('\n' + '='*60)
    print('Ensemble Results')
    print('='*60)
    print(f'Models used: {args.models}')
    print(f'Top-1 Accuracy: {top1_acc*100:.2f}% ({top1_correct}/{total})')
    print(f'Top-5 Accuracy: {top5_acc*100:.2f}% ({top5_correct}/{total})')
    
    # 整理類別準確率
    class_acc_list = []
    for idx in range(len(folder_classes)):
        if class_total[idx] > 0:
            acc = class_correct[idx] / class_total[idx]
            class_name = folder_classes[idx]
            chinese_name = DEFAULT_CLASS_NAMES.get(class_name, '')
            class_acc_list.append((class_name, chinese_name, acc, class_correct[idx], class_total[idx]))
            
    class_acc_list.sort(key=lambda x: x[2])
    
    print('\n最差的 10 個類別 (Ensemble):')
    for name, cn_name, acc, correct, total_cls in class_acc_list[:10]:
        print(f'  {name} ({cn_name}): {acc*100:.1f}% ({correct}/{total_cls})')
        
    print('\n最好的 10 個類別 (Ensemble):')
    for name, cn_name, acc, correct, total_cls in class_acc_list[-10:]:
        print(f'  {name} ({cn_name}): {acc*100:.1f}% ({correct}/{total_cls})')
        
    print('='*60)
    
    return class_acc_list

def main():
    if len(args.models) != len(args.checkpoints):
        print("Error: Number of models must match number of checkpoints")
        return
        
    # 1. 載入所有模型
    models = load_models(args.models, args.checkpoints, args.num_classes)
    
    # 2. 執行融合評估
    results = ensemble_evaluate(models, args.data_dir, args.batch_size, args.num_workers)
    
    # 3. 存檔
    if args.save_results:
        with open(args.save_results, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'Chinese Name', 'Accuracy', 'Correct', 'Total'])
            for name, cn_name, acc, correct, total_cls in results:
                writer.writerow([name, cn_name, f'{acc:.4f}', correct, total_cls])
        print(f'Results saved to {args.save_results}')

if __name__ == '__main__':
    main()

