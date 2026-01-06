# 忽略非关键警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# === NPU 核心组件导入 ===
import torch_npu
from torch_npu.npu import amp
from torch_npu.optim import NpuFusedSGD # 使用昇腾专用融合优化器

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training on Ascend 910B')
    parser.add_argument('--data', default='../../dataset/', help='数据集存放路径 (会自动下载)')
    parser.add_argument('--epochs', default=50, type=int, help='训练轮数')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch Size')
    parser.add_argument('--num-workers', default=4, type=int, help='数据加载线程数')
    parser.add_argument('--lr', default=0.1, type=float, help='初始学习率')
    parser.add_argument('--device-id', default=0, type=int, help='NPU设备ID')
    args = parser.parse_args()

    # 1. 环境初始化
    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)
    print(f"🚀 Running on Ascend 910B (Device: {device})")

    # 2. 数据准备 (CIFAR-10)
    print("⏳ Preparing Data...")
    
    # CIFAR-10 的标准数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 下载并加载数据集 (num_workers 根据 CPU 核数调整，建议 4-8)
    trainset = torchvision.datasets.CIFAR10(
        root=args.data, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, drop_last=True) # pin_memory=True 加速 host 到 device 传输

    testset = torchvision.datasets.CIFAR10(
        root=args.data, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True)

    # 3. 模型定义 (魔改版 ResNet50)
    print("🧠 Building Model...")
    # num_classes=10 对应 CIFAR-10 的类别数
    model = models.resnet50(num_classes=10)
    
    # === 关键修改 ===
    # 标准 ResNet50 第一层是 7x7 卷积 + 2倍下采样，适合 224x224 大图。
    # 对于 CIFAR-10 (32x32)，我们需要把第一层改成 3x3 卷积，去掉 stride，
    # 否则特征图在进入网络深层前就变得太小了，导致精度无法提升。
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() # 去掉第一层的 MaxPool
    
    model = model.to(device)

    # 4. 优化器与 Loss
    criterion = nn.CrossEntropyLoss().to(device)
    # 使用 NPU 专用融合 SGD，比原生 SGD 快
    optimizer = NpuFusedSGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # 学习率调整策略
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 混合精度 Scaler
    scaler = amp.GradScaler()

    # 5. 训练与测试循环
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # --- 训练阶段 ---
        train_loss, train_acc, fps = train_one_epoch(
            train_loader, model, criterion, optimizer, scaler, device)
        
        # --- 测试阶段 ---
        val_acc = validate(test_loader, model, device)
        
        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # 打印日志
        print(f"Epoch [{epoch+1}/{args.epochs}] | "
              f"Time: {epoch_time:.1f}s | FPS: {fps:.1f} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")

        # --- 保存最佳模型 ---
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"🎉 New Best Accuracy: {best_acc:.2f}%! Saving checkpoint...")
            state = {
                'net': model.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_best.pth')

    print(f"\n✅ Training Finished. Best Accuracy: {best_acc:.2f}%")

def train_one_epoch(train_loader, model, criterion, optimizer, scaler, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    total_samples = 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 异步传输数据到 NPU
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 混合精度前向计算 (FP16)
        with amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # 反向传播与优化
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 统计数据
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)

    end_time = time.time()
    fps = total_samples / (end_time - start_time)
    
    return train_loss / (batch_idx + 1), 100. * correct / total, fps

def validate(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # 推理时也可以开启混合精度加速
            with amp.autocast():
                outputs = model(inputs)
                
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total

if __name__ == '__main__':
    main()