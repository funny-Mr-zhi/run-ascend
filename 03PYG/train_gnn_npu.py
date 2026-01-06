import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# === NPU 核心组件 ===
import torch_npu
from torch_npu.optim import NpuFusedAdam  # GNN 这里的神器

# 屏蔽警告
import warnings
warnings.filterwarnings('ignore')

def main():
    # 1. 环境初始化
    device = torch.device('npu:0')
    torch.npu.set_device(device)
    print(f"🚀 Running GNN on Ascend 910B (Device: {device})")

    # 2. 准备数据 (Cora)
    print("⏳ Loading Cora Dataset...")
    dataset = Planetoid(root='../dataset/Cora', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    
    # 将整个图数据搬运到 NPU
    # 注意：GNN 通常是一次性把整个图(x, edge_index)放到显存里
    data = data.to(device)

    # 3. 定义 GCN 模型
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # PyG 的 GCNConv 底层会自动调用 torch.scatter 等算子
            # 在 NPU 上会被映射到高性能的 AICore 算子上
            self.conv1 = GCNConv(dataset.num_features, 16)
            self.conv2 = GCNConv(16, dataset.num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    model = GCN().to(device)
    
    # 4. 优化器 (使用 NPU 融合 Adam)
    # GNN 参数通常比较稀疏，FusedAdam 能显著减少 CPU 下发指令的开销
    optimizer = NpuFusedAdam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 5. 训练循环
    print("🔥 Start Training...")
    model.train()
    
    # 记录时间
    times = []
    
    for epoch in range(200):
        start = time.time()
        
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # 仅计算训练集的 Loss
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        torch.npu.synchronize() # 准确计时需要
        end = time.time()
        times.append(end - start)

        if epoch % 20 == 0:
            # 简单验证一下精度
            model.eval()
            pred = model(data.x, data.edge_index).argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            acc = int(correct) / int(data.test_mask.sum())
            model.train()
            print(f'Epoch {epoch:03d}: Loss: {loss.item():.4f}, Test Acc: {acc:.4f}')

    avg_time = sum(times) / len(times) * 1000 # 转为毫秒
    print(f"\n✅ Training Finished!")
    print(f"Average Epoch Time: {avg_time:.2f} ms")
    print("If this runs without error, your PyG environment is ready!")

if __name__ == '__main__':
    main()