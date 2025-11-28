"""
Inference script for Herbal Classification
支援：
  1. 單張圖片預測
  2. 整個資料夾（ImageFolder 格式）批次評估並計算準確率
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

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Palette images with Transparency")

parser = argparse.ArgumentParser(description='Herbal Classification Inference')
parser.add_argument('--image-path', type=str, default=None, help='Input image path (for single image inference)')
parser.add_argument('--data-dir', type=str, default=None, help='Data directory for batch evaluation (ImageFolder format)')
parser.add_argument('--arch', default='swin_t', type=str, help='network architecture')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth.tar)')
parser.add_argument('--num-classes', type=int, default=100, help='Number of classes')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--topk', type=int, default=5, help='Show top-k predictions')
parser.add_argument('--save-results', type=str, default=None, help='Save results to CSV file')

# args = parser.parse_args()  <-- 移到 main()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# 100 類中藥材的類別名稱（拼音 -> 中文對照）
DEFAULT_CLASS_NAMES = {
    'Anxixiang': '安息香', 'Baibiandou': '白扁豆', 'Baifan': '白矾', 'Bailian': '白莲',
    'Baimaogen': '白茅根', 'Baiqian': '白前', 'Baishao': '白芍', 'Baizhi': '白芷',
    'Baiziren': '柏子仁', 'Beishashen': '北沙参', 'Bibo': '荜拨', 'Bichengqie': '荜澄茄',
    'Biejia': '鳖甲', 'Binglang': '槟榔', 'Cangzhu': '苍术', 'Caodoukou': '草豆蔻',
    'Chenxiang': '沉香', 'Chuanlianzi': '川楝子', 'Chuanmuxiang': '川木香', 'Chuanniuxi': '川牛膝',
    'Dafupi': '大腹皮', 'Dandouchi': '淡豆豉', 'Daoya': '稻芽', 'Dilong': '地龙',
    'Dongchongxiacao': '冬虫夏草', 'Fangfeng': '防风', 'Fanxieye': '番泻叶', 'Fengfang': '蜂房',
    'Gancao': '甘草', 'Ganjiang': '干姜', 'Gansong': '甘松', 'Gaoben': '藁本',
    'Ghishizhi': '赤石脂', 'Gouqizi': '枸杞子', 'Guizhi': '桂枝', 'Gujingcao': '骨筋草',
    'Guya': '谷芽', 'HaiIong': '海龙', 'Haipiaoxiao': '海螵蛸', 'Hehuanpi': '合欢皮',
    'Huangbo': '黄柏', 'Huangqi': '黄芪', 'Huangqin': '黄芩', 'Hubeibeimu': '湖北贝母',
    'Jiangcan': '僵蚕', 'Jiezi': '芥子', 'Jiguanhua': '鸡冠花', 'Jindenglong': '金灯笼',
    'Jineijin': '鸡内金', 'Jingjiesui': '荆芥穗', 'Jinguolan': '金果榄', 'Jinqianbaihuashe': '金钱白花蛇',
    'Jiuxiangchong': '九香虫', 'Juhe': '橘核', 'Kudiding': '苦地丁', 'Laifuzi': '莱菔子',
    'Lianfang': '莲房', 'Lianxu': '莲须', 'Lianzi': '莲子', 'Lianzixin': '莲子心',
    'Lingzhi': '灵芝', 'Lizhihe': '荔枝核', 'Longyanrou': '龙眼肉', 'Lugen': '芦根',
    'Lulutong': '路路通', 'Maidong': '麦冬', 'Mudingxiang': '木丁香', 'Qianghuo': '羌活',
    'Qiannianjian': '千年健', 'Qinpi': '秦皮', 'Quanxie': '全蝎', 'Rendongteng': '忍冬藤',
    'Renshen': '人参', 'Roudoukou': '肉豆蔻', 'Sangjisheng': '桑寄生', 'Sangpiaoxiao': '桑螵蛸',
    'Sangshen': '桑葚', 'Shancigu': '山慈菇', 'Shannai': '山奈', 'Shanzhuyu': '山茱萸',
    'Shayuanzi': '沙苑子', 'Shiliupi': '石榴皮', 'Sigualuo': '丝瓜络', 'Suanzaoren': '酸枣仁',
    'Sumu': '苏木', 'Taizishen': '太子参', 'Tianhuafen': '天花粉', 'Tianma': '天麻',
    'Tujingpi': '土荆皮', 'Walengzi': '瓦楞子', 'Wujiapi': '五加皮', 'Xixin': '细辛',
    'Yinchaihu': '银柴胡', 'Yiyiren': '薏苡仁', 'Yujin': '郁金', 'Zhebeimu': '浙贝母',
    'Zhiqiao': '枳壳', 'Zhuru': '竹茹', 'Zhuyazao': '猪牙皂', 'Zirantong': '自然铜',
}


def safe_rgb_loader(path):
    """安全載入圖片並轉換為 RGB，處理損壞的圖片"""
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.load()  # 強制載入完整圖片，提前發現損壞
            return img.convert('RGB')
    except (OSError, IOError) as e:
        print(f'\nWarning: Skipping corrupted image: {path}')
        print(f'  Error: {e}')
        # 回傳一個空白圖片，之後會被過濾掉
        return Image.new('RGB', (224, 224), (128, 128, 128))


def build_model(arch, num_classes):
    """根據架構名稱建立模型"""
    if arch == 'convnext_tiny':
        from torchvision.models import convnext_tiny
        model = convnext_tiny(num_classes=num_classes)
    elif arch == 'resnet50':
        from torchvision.models import resnet50
        model = resnet50(num_classes=num_classes)
    elif arch == 'densenet161':
        from torchvision.models import densenet161
        model = densenet161(num_classes=num_classes)
    elif arch == 'efficientnet_b0':
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(num_classes=num_classes)
    elif arch == 'efficientnet_b1':
        from torchvision.models import efficientnet_b1
        model = efficientnet_b1(num_classes=num_classes)
    elif arch == 'efficientnet_b2':
        from torchvision.models import efficientnet_b2
        model = efficientnet_b2(num_classes=num_classes)
    elif arch == 'efficientnet_b3':
        from torchvision.models import efficientnet_b3
        model = efficientnet_b3(num_classes=num_classes)
    elif arch == 'efficientnet_b4':
        from torchvision.models import efficientnet_b4
        model = efficientnet_b4(num_classes=num_classes)
    elif arch == 'regnet_x_3_2gf':
        from torchvision.models import regnet_x_3_2gf
        model = regnet_x_3_2gf(num_classes=num_classes)
    elif arch == 'vit_b_16':
        from torchvision.models import vit_b_16
        model = vit_b_16(num_classes=num_classes)
    elif arch == 'mobilenet_v3_large':
        from torchvision.models import mobilenet_v3_large
        model = mobilenet_v3_large(num_classes=num_classes)
    elif arch == 'shufflenet_v2_x1_5':
        from torchvision.models import shufflenet_v2_x1_5
        model = shufflenet_v2_x1_5(num_classes=num_classes)
    elif arch == 'swin_b':
        from torchvision.models import swin_b
        model = swin_b(num_classes=num_classes)
    elif arch == 'swin_s':
        from torchvision.models import swin_s
        model = swin_s(num_classes=num_classes)
    elif arch == 'swin_t':
        from torchvision.models import swin_t
        model = swin_t(num_classes=num_classes)
    else:
        raise NotImplementedError(f"Architecture '{arch}' not supported")
    return model


def evaluate_folder(net, data_dir, transform, class_names, batch_size=32, num_workers=4):
    """對整個資料夾進行評估，計算 Top-1 和 Top-5 準確率"""
    
    # 使用 ImageFolder 載入資料
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
    print(f'Number of classes: {len(dataset.classes)}')
    
    # 確認類別順序
    folder_classes = dataset.classes
    print(f'Classes (first 5): {folder_classes[:5]}...')
    
    # 統計
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    # 每個類別的統計
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    # 錯誤預測記錄
    wrong_predictions = []
    
    net.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for inputs, targets in pbar:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = net(inputs)
            
            # Top-1
            _, pred = logits.max(1)
            top1_correct += pred.eq(targets).sum().item()
            
            # Top-5
            _, pred_top5 = logits.topk(5, dim=1)
            for i in range(targets.size(0)):
                if targets[i] in pred_top5[i]:
                    top5_correct += 1
            
            # 每個類別統計
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if pred[i] == targets[i]:
                    class_correct[label] += 1
                else:
                    # 記錄錯誤預測
                    wrong_predictions.append({
                        'true_label': folder_classes[label],
                        'pred_label': folder_classes[pred[i].item()],
                        'confidence': F.softmax(logits[i], dim=0)[pred[i]].item()
                    })
            
            total += targets.size(0)
            
            # 更新進度條
            current_acc = top1_correct / total
            pbar.set_postfix({'Top-1 Acc': f'{current_acc:.4f}'})
    
    # 計算總體準確率
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    
    print('\n' + '='*60)
    print('Evaluation Results')
    print('='*60)
    print(f'Total images: {total}')
    print(f'Top-1 Accuracy: {top1_acc*100:.2f}% ({top1_correct}/{total})')
    print(f'Top-5 Accuracy: {top5_acc*100:.2f}% ({top5_correct}/{total})')
    
    # 顯示每個類別的準確率（按準確率排序）
    print('\n' + '-'*60)
    print('Per-class Accuracy (sorted by accuracy):')
    print('-'*60)
    
    class_acc_list = []
    for idx in range(len(folder_classes)):
        if class_total[idx] > 0:
            acc = class_correct[idx] / class_total[idx]
            class_name = folder_classes[idx]
            chinese_name = DEFAULT_CLASS_NAMES.get(class_name, '')
            class_acc_list.append((class_name, chinese_name, acc, class_correct[idx], class_total[idx]))
    
    # 按準確率排序（最低的在前）
    class_acc_list.sort(key=lambda x: x[2])
    
    # 顯示最差的 10 個類別
    print('\n最差的 10 個類別:')
    for name, cn_name, acc, correct, total_cls in class_acc_list[:10]:
        print(f'  {name} ({cn_name}): {acc*100:.1f}% ({correct}/{total_cls})')
    
    # 顯示最好的 10 個類別
    print('\n最好的 10 個類別:')
    for name, cn_name, acc, correct, total_cls in class_acc_list[-10:]:
        print(f'  {name} ({cn_name}): {acc*100:.1f}% ({correct}/{total_cls})')
    
    print('='*60)
    
    return {
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'total': total,
        'class_acc': class_acc_list,
        'wrong_predictions': wrong_predictions[:100]  # 只保留前 100 個錯誤
    }


def inference_single(net, image_path, transform, class_names, topk=5):
    """對單張圖片進行推論"""
    print(f'\nProcessing image: {image_path}')
    raw_image = Image.open(image_path).convert("RGB")
    image = transform(raw_image)
    image = torch.unsqueeze(image, 0).cuda()
    
    net.eval()
    with torch.no_grad():
        logits = net(image)
        probs = F.softmax(logits, dim=1)
        topk_probs, topk_indices = probs.topk(topk, dim=1)
    
    print('\n' + '='*60)
    print(f'Top-{topk} Predictions:')
    print('='*60)
    
    for i in range(topk):
        idx = topk_indices[0, i].item()
        prob = topk_probs[0, i].item()
        class_key = class_names[idx]
        chinese_name = DEFAULT_CLASS_NAMES.get(class_key, class_key)
        
        rank_marker = "★" if i == 0 else " "
        print(f'{rank_marker} {i+1}. {class_key} ({chinese_name}): {prob*100:.2f}%')
    
    print('='*60)
    
    top1_idx = topk_indices[0, 0].item()
    top1_prob = topk_probs[0, 0].item()
    top1_class = class_names[top1_idx]
    top1_chinese = DEFAULT_CLASS_NAMES.get(top1_class, top1_class)
    
    print(f'\n預測結果: {top1_class} ({top1_chinese})')
    print(f'信心度: {top1_prob*100:.2f}%')
    
    return top1_class, top1_prob


def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print('='*60)
    print('Herbal Classification Inference')
    print('='*60)
    
    # 檢查參數
    if args.image_path is None and args.data_dir is None:
        print("Error: Please provide either --image-path or --data-dir")
        return
    
    # 建立模型
    print(f'Building model: {args.arch}')
    net = build_model(args.arch, args.num_classes)
    
    # 載入 checkpoint
    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    
    if 'net' in checkpoint:
        net.load_state_dict(checkpoint['net'])
        if 'acc' in checkpoint:
            print(f"Model accuracy (from checkpoint): {checkpoint['acc']}")
        if 'epoch' in checkpoint:
            print(f"Trained epochs: {checkpoint['epoch']}")
    else:
        net.load_state_dict(checkpoint)
    
    net = net.cuda()
    net.eval()
    cudnn.benchmark = True
    print('Model loaded successfully!')
    
    # 圖片前處理
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 類別名稱（按字母順序，與 ImageFolder 一致）
    class_names = sorted(DEFAULT_CLASS_NAMES.keys())
    
    if args.data_dir:
        # 批次評估整個資料夾
        results = evaluate_folder(
            net, args.data_dir, transform, class_names,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # 儲存結果到 CSV
        if args.save_results:
            import csv
            with open(args.save_results, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Class', 'Chinese Name', 'Accuracy', 'Correct', 'Total'])
                for name, cn_name, acc, correct, total_cls in results['class_acc']:
                    writer.writerow([name, cn_name, f'{acc:.4f}', correct, total_cls])
            print(f'\nResults saved to: {args.save_results}')
    
    elif args.image_path:
        # 單張圖片推論
        inference_single(net, args.image_path, transform, class_names, args.topk)


if __name__ == '__main__':
    main()
