import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import shutil
import argparse
import numpy as np
import re
import socket
import warnings

# 忽略 Palette 圖片透明度警告
warnings.filterwarnings("ignore", message="Palette images with Transparency")

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, correct_num, set_logger
from tqdm import tqdm

import wandb
import time
import math


def safe_rgb_loader(path):
    """安全載入圖片並轉換為 RGB，處理 Palette/透明度問題"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--traindir', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--valdir', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet50', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight deacy')
parser.add_argument('--lr-type', default='cosine', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[150, 225], type=int, nargs='+', help='milestones for lr-multistep')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--hybridmix', action='store_true', help='using data augmentation hybridmix')

parser.add_argument('--imagenet-pretrained', default='', type=str, help='imagenet pretrained')
parser.add_argument('--resume-checkpoint', default='./checkpoint/resnet32.pth.tar', type=str, help='resume checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--evaluate-checkpoint', default='./checkpoint/resnet32_best.pth.tar', type=str, help='evaluate checkpoint')
parser.add_argument('--checkpoint-dir', default='./checkpoint_baseline_new_336', type=str, help='checkpoint directory')
parser.add_argument('--optimizer', default='auto', type=str, choices=['sgd', 'adamw', 'auto'], help='optimizer type (default: auto)')

# wandb 相關參數
parser.add_argument('--wandb-project', default='herbal-classification', type=str, help='wandb project name')
parser.add_argument('--wandb-entity', default=None, type=str, help='wandb entity/team name')
parser.add_argument('--no-wandb', action='store_true', help='disable wandb logging')

    

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

args.log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'arch' + '_' +  args.arch + '_'+\
          'seed'+ str(args.manual_seed)
args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.log_dir)
if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

log_txt =  os.path.join(args.checkpoint_dir, args.log_dir +'.txt')

logger = set_logger(log_txt)
logger.info("==========\nArgs:{}\n==========".format(args))

# 初始化 wandb
use_wandb = not args.no_wandb and not args.evaluate
if use_wandb:
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.arch}_seed{args.manual_seed}_{socket.gethostname()}",
        config={
            "arch": args.arch,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "init_lr": args.init_lr,
            "weight_decay": args.weight_decay,
            "lr_type": args.lr_type,
            "hybridmix": args.hybridmix,
            "seed": args.manual_seed,
        }
    )
    logger.info("wandb initialized: project={}, entity={}".format(args.wandb_project, args.wandb_entity))


np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)


##  Data Augmentation (資料增強)
## 使用 torchvision.datasets.ImageFolder 讀取 traindir、valdir，訓練端包含 RandomResizedCrop、RandomHorizontalFlip，驗證端使用 Resize+CenterCrop。若 --hybridmix，則在 DataLoader 內啟用 CutMix/MixUp 組合的 collate_fn。


img_size = 224 # For resnet default input size

## 使用 torchvision.datasets.ImageFolder 讀取 traindir、valdir
## 訓練端加入更豐富的 Augmentation 以提升泛化能力
trainset = torchvision.datasets.ImageFolder(
    args.traindir,
    transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip()
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)), # 隨機裁切並縮放，增加對物體大小變化的魯棒性
        transforms.RandomHorizontalFlip(),                        # 隨機水平翻轉
        transforms.RandomRotation(20),                            # 隨機旋轉 +/- 20度，模擬不同拍攝角度
        transforms.ColorJitter(0.2, 0.2, 0.2),                    # 隨機調整亮度、對比、飽和度，模擬不同光照條件
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
    loader=safe_rgb_loader)  # 使用安全載入器處理 Palette 圖片

## 驗證/測試端通常只做 Resize 和 Normalize，保持圖片原貌
testset = torchvision.datasets.ImageFolder(
    args.valdir, 
    transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.Resize((img_size, img_size)), # 直接縮放至指定大小 (有些做法是 Resize(256) + CenterCrop(224))
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ]),
    loader=safe_rgb_loader)  # 使用安全載入器處理 Palette 圖片

NUM_CLASSES = len(set(trainset.classes))

cutmix = v2.CutMix(num_classes=NUM_CLASSES)
mixup = v2.MixUp(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

from torch.utils.data import default_collate

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))


if args.hybridmix:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn)
else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=args.num_workers)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers,
                                          pin_memory=True)

logger.info("Number of train dataset: {}".format(len(trainloader.dataset)))
logger.info("Number of validation dataset: {}".format(len(testloader.dataset)))
logger.info("Number of classes: {}".format(len(set(trainloader.dataset.classes))))
# num_classes = len(set(trainloader.dataset.classes))
logger.info('==> Building model..')
# model = getattr(models, args.arch)
if args.arch == 'convnext_tiny':
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    model = convnext_tiny(num_classes=NUM_CLASSES)
    weights = ConvNeXt_Tiny_Weights.verify(ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.geargst_state_dict(check_hash=True)
    del imagenet_model_dict['classifier.2.weight']
    del imagenet_model_dict['classifier.2.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
elif args.arch == 'resnet50':
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(num_classes=NUM_CLASSES)
    weights = ResNet50_Weights.verify(ResNet50_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    del imagenet_model_dict['fc.weight']
    del imagenet_model_dict['fc.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
elif args.arch == 'densenet161':
    from torchvision.models import densenet161, DenseNet161_Weights
    model = densenet161(num_classes=NUM_CLASSES)
    weights = DenseNet161_Weights.verify(DenseNet161_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    del imagenet_model_dict['classifier.weight']
    del imagenet_model_dict['classifier.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
elif args.arch == 'efficientnet_b0':
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(num_classes=NUM_CLASSES)
    weights = EfficientNet_B0_Weights.verify(EfficientNet_B0_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    del imagenet_model_dict['classifier.1.weight']
    del imagenet_model_dict['classifier.1.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
# === 新增以下區塊 ===
elif args.arch == 'efficientnet_b1':
    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    model = efficientnet_b1(num_classes=NUM_CLASSES)
    weights = EfficientNet_B1_Weights.verify(EfficientNet_B1_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    del imagenet_model_dict['classifier.1.weight']
    del imagenet_model_dict['classifier.1.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)

elif args.arch == 'efficientnet_b2':
    from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
    model = efficientnet_b2(num_classes=NUM_CLASSES)
    weights = EfficientNet_B2_Weights.verify(EfficientNet_B2_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    del imagenet_model_dict['classifier.1.weight']
    del imagenet_model_dict['classifier.1.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)

elif args.arch == 'efficientnet_b3':
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    model = efficientnet_b3(num_classes=NUM_CLASSES)
    weights = EfficientNet_B3_Weights.verify(EfficientNet_B3_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    del imagenet_model_dict['classifier.1.weight']
    del imagenet_model_dict['classifier.1.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)

elif args.arch == 'efficientnet_b4':
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
    model = efficientnet_b4(num_classes=NUM_CLASSES)
    weights = EfficientNet_B4_Weights.verify(EfficientNet_B4_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    del imagenet_model_dict['classifier.1.weight']
    del imagenet_model_dict['classifier.1.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)

elif args.arch == 'regnet_x_3_2gf':
    from torchvision.models import regnet_x_3_2gf, RegNet_X_3_2GF_Weights
    model = regnet_x_3_2gf(num_classes=NUM_CLASSES)
    weights = RegNet_X_3_2GF_Weights.verify(RegNet_X_3_2GF_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    del imagenet_model_dict['fc.weight']
    del imagenet_model_dict['fc.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
elif args.arch == 'vit_b_16':
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    model = vit_b_16(num_classes=NUM_CLASSES)
    weights = ViT_B_16_Weights.verify(ViT_B_16_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    del imagenet_model_dict['heads.head.weight']
    del imagenet_model_dict['heads.head.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
elif args.arch == 'mobilenet_v3_large':
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    model = mobilenet_v3_large(num_classes=NUM_CLASSES)
    weights = MobileNet_V3_Large_Weights.verify(MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    del imagenet_model_dict['classifier.3.weight']
    del imagenet_model_dict['classifier.3.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
elif args.arch == 'shufflenet_v2_x1_5':
    from torchvision.models import shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_Weights
    model = shufflenet_v2_x1_5(num_classes=NUM_CLASSES)
    weights = ShuffleNet_V2_X1_5_Weights.verify(ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    del imagenet_model_dict['fc.weight']
    del imagenet_model_dict['fc.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
elif args.arch == 'swin_b':
    from torchvision.models import swin_b, Swin_B_Weights
    model = swin_b(num_classes=NUM_CLASSES)
    weights = Swin_B_Weights.verify(Swin_B_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    # print(imagenet_model_dict.keys())
    del imagenet_model_dict['head.weight']
    del imagenet_model_dict['head.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
elif args.arch == 'swin_s':
    from torchvision.models import swin_s, Swin_S_Weights
    model = swin_s(num_classes=NUM_CLASSES)
    weights = Swin_S_Weights.verify(Swin_S_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    # print(imagenet_model_dict.keys())
    del imagenet_model_dict['head.weight']
    del imagenet_model_dict['head.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
elif args.arch == 'swin_t':
    from torchvision.models import swin_t, Swin_T_Weights
    model = swin_t(num_classes=NUM_CLASSES)
    weights = Swin_T_Weights.verify(Swin_T_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    # print(imagenet_model_dict.keys())
    del imagenet_model_dict['head.weight']
    del imagenet_model_dict['head.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
elif args.arch == 'convnext_tiny_acmix':
    from acmix_model import convnext_tiny_acmix
    logger.info("Creating ConvNeXt-Tiny + ACMix model...")
    model = convnext_tiny_acmix(num_classes=NUM_CLASSES)
    # 嘗試載入官方 ConvNeXt-Tiny 權重到共用的部分（加速收斂）
    try:
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        weights = ConvNeXt_Tiny_Weights.verify(ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        imagenet_model_dict = weights.get_state_dict(check_hash=True)
        
        # 過濾掉不匹配的層 (因為 ACMix 結構不同，Block 內的權重無法直接對應)
        # 但 downsample layers 和 stem 是可以共用的
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in imagenet_model_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} shared layers from ConvNeXt-Tiny ImageNet weights")
    except Exception as e:
        logger.info(f"Could not load pretrained weights: {e}. Training from scratch.")
        
else:
    raise NotImplementedError
# net = model(num_classes=NUM_CLASSES)
net = model
net.eval()
resolution = (1, 3, 224, 224)
logger.info('Arch: %s, Params: %.2fM'
        % (args.arch, cal_param_size(net)/1e6))
del(net)

net = model.cuda()
#net = torch.nn.DataParallel(net)
if len(args.imagenet_pretrained) !=0 and not 'vit' in args.arch and not 'swin' in args.arch:
    initalized_model_dict = net.state_dict()
    imagenet_model_dict = torch.load(args.imagenet_pretrained, map_location=torch.device('cpu'))
    if 'densenet' in args.arch:
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        state_dict = imagenet_model_dict
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        imagenet_model_dict = state_dict
    for key in initalized_model_dict.keys():
        if 'fc' in key or 'classifier' in key or 'num_batches_tracked' in key:
            continue
        initalized_model_dict[key] = imagenet_model_dict[key]
    net.load_state_dict(initalized_model_dict)
    logger.info("Load imagenet pretrained weights!")
cudnn.benchmark = True


def train(epoch, criterion_list, optimizer):
    """訓練一個 epoch，回傳 (train_loss, lr)"""
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')

    lr = adjust_lr(optimizer, epoch, args)
    start_time = time.time()
    criterion_ce = criterion_list[0]

    net.train()
    
    # 使用 tqdm 顯示進度條，不要每個 batch 都 print
    pbar = tqdm(trainloader, desc=f'Train Epoch {epoch}', leave=False)
    for inputs, targets in pbar:
        inputs = inputs.float().cuda()
        targets = targets.cuda()

        optimizer.zero_grad()
        logits = net(inputs)
        loss_cls = criterion_ce(logits, targets)
        loss = loss_cls
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
        train_loss_cls.update(loss_cls.item(), inputs.size(0))
        
        # 更新進度條顯示
        pbar.set_postfix({'loss': f'{train_loss.avg:.4f}', 'lr': f'{lr:.5f}'})

    duration = time.time() - start_time
    logger.info('Epoch:{}\t lr:{:.4f}\t Duration:{:.1f}s\t Train_loss:{:.5f}'
                .format(epoch, lr, duration, train_loss.avg))
    
    return train_loss.avg, lr


def test(epoch, criterion_ce):
    """驗證一個 epoch，回傳 (acc, val_loss)"""
    net.eval()
    test_loss_cls = AverageMeter('test_loss_cls', ':.4e')

    top1_num = 0
    total = 0
    
    with torch.no_grad():
        # 使用 tqdm 顯示進度條
        pbar = tqdm(testloader, desc=f'Val Epoch {epoch}', leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = net(inputs)

            loss_cls = criterion_ce(logits, targets)
            test_loss_cls.update(loss_cls, inputs.size(0))

            top1 = correct_num(logits, targets, topk=(1,))[0]
            top1_num += top1
            total += targets.size(0)
            
            # 更新進度條
            current_acc = (top1_num / total).item()
            pbar.set_postfix({'acc': f'{current_acc:.4f}'})

        acc1 = round((top1_num/total).item(), 4)
        val_loss = test_loss_cls.avg.item() if hasattr(test_loss_cls.avg, 'item') else test_loss_cls.avg

        logger.info('Val Epoch:{}\t Val_loss:{:.5f}\t Val_Acc:{:.4f}'
                    .format(epoch, val_loss, acc1))

    return acc1, val_loss


if __name__ == '__main__':
    
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion_ce = nn.CrossEntropyLoss()

    if args.evaluate:      
        logger.info('Evaluate pre-trained weights from: {}'.format(args.evaluate_checkpoint))
        checkpoint = torch.load(args.evaluate_checkpoint, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['net'])
        acc, val_loss = test(start_epoch, criterion_ce)
        logger.info('Evaluation Acc: {:.4f}'.format(acc))
    else:
        trainable_list = nn.ModuleList([])
        trainable_list.append(net)

        data = torch.randn(1, 3, 224, 224).cuda()
        net.eval()
        logits = net(data)


        ## 為什麼選 AdamW？
        ## 收斂性：Transformer 架構對梯度的變異非常敏感，SGD 往往較難收斂或需要極精細的參數調整。AdamW 能自適應調整每個參數的學習率，效果通常更好。
        ## 標準配置：Swin Transformer 的原始論文和官方實作都是使用 AdamW。
        # 根據模型架構選擇優化器與初始學習率建議

        # 判斷是否應該使用 AdamW
        use_adamw = False
        if args.optimizer == 'adamw':
            use_adamw = True
        elif args.optimizer == 'auto':
             if 'swin' in args.arch or 'vit' in args.arch or 'convnext' in args.arch:
                 use_adamw = True

        if use_adamw:
            # Transformer 和 ConvNeXt 類模型通常建議使用 AdamW，或者使用者強制指定
            logger.info(f"Using AdamW optimizer for {args.arch}")
            
            # 如果使用者沒有特別指定 init-lr (若是預設 0.1)，則自動調整為適合 AdamW 的數值 (如 5e-4)
            if args.init_lr == 0.1: 
                args.init_lr = 5e-4
                logger.info(f"Auto-adjusting init_lr to {args.init_lr} for AdamW")

            optimizer = optim.AdamW(trainable_list.parameters(),
                                    lr=args.init_lr,
                                    weight_decay=args.weight_decay)
        else:
            # CNN 類模型 (ResNet, DenseNet 等) 預設使用 SGD
            logger.info(f"Using SGD optimizer for {args.arch}")
            optimizer = optim.SGD(trainable_list.parameters(),
                                  lr=args.init_lr, 
                                  momentum=0.9, 
                                  weight_decay=args.weight_decay, 
                                  nesterov=True)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_ce)
        criterion_list.cuda()

        if args.resume:
            logger.info('Resume pre-trained weights from: {}'.format(args.resume_checkpoint))
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1

        # 訓練主迴圈
        logger.info('='*60)
        logger.info('Start Training...')
        logger.info('='*60)
        
        for epoch in range(start_epoch, args.epochs):
            # Train
            train_loss, lr = train(epoch, criterion_list, optimizer)
            
            # Validation
            acc, val_loss = test(epoch, criterion_ce)

            # 檢查是否為最佳模型
            is_best = acc > best_acc
            if is_best:
                best_acc = acc
                
            # wandb log：每個 epoch 記錄所有 metrics
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/lr": lr,
                    "val/loss": val_loss,
                    "val/acc": acc,
                    "val/best_acc": best_acc,
                })

            # 顯示清楚的 epoch 摘要（含是否更新 best）
            best_marker = " ★ NEW BEST!" if is_best else ""
            print(f'\n[Epoch {epoch}/{args.epochs-1}] '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Acc: {acc:.4f} | '
                  f'Best: {best_acc:.4f}{best_marker}\n')

            # 儲存 checkpoint
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, args.arch + '.pth.tar'))

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, args.arch + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'))
                logger.info('*** New best model saved! Acc: {:.4f} ***'.format(acc))

        # 訓練結束，載入最佳模型做最終評估
        logger.info('='*60)
        logger.info('Training finished! Evaluating best model...')
        logger.info('='*60)
        logger.info('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar')))
        
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['net'])
        best_epoch = checkpoint['epoch']
        top1_acc, _ = test(best_epoch, criterion_ce)

        logger.info('='*60)
        logger.info('Final Best Accuracy: {:.4f} (Epoch {})'.format(top1_acc, best_epoch))
        logger.info('='*60)
        
        # wandb 記錄最終結果並結束
        if use_wandb:
            wandb.run.summary["best_acc"] = top1_acc
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.finish()
