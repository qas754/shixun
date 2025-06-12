# 完整的模型训练套路(以CIFAR10为例)
import time
import torch
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import os
from dataset import ImageTxtDataset
from model import Chen

# 准备数据集
train_data = ImageTxtDataset(r"D:\python.learm\shixun\day3\train.txt",
                             r"D:\python.learm\shixun\day3\images\train",
                             transforms.Compose([transforms.Resize(32),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ]))

test_data = ImageTxtDataset(r"D:\python.learm\shixun\day3\val.txt",
                            r"D:\python.learm\shixun\day3\val",
                            transforms.Compose([transforms.Resize(32),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
                                                ]))

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度: {train_data_size}")
print(f"测试数据集的长度: {test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# 创建网络模型
chen = Chen()
print("模型结构:")
print(chen)  # 打印模型结构，确认输出层维度

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optim = torch.optim.SGD(chen.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0  # 记录训练的次数
total_test_step = 0   # 记录测试的次数
epoch = 10            # 训练的轮数
num_classes = 10      # 假设 CIFAR-10，有 10 个类别，需根据实际数据集调整

# 添加 TensorBoard
writer = SummaryWriter("../logs_train")

# 添加开始时间
start_time = time.time()

for i in range(epoch):
    print(f"-----第 {i+1} 轮训练开始-----")
    # 训练步骤
    chen.train()  # 设置模型为训练模式
    for data in train_loader:
        imgs, targets = data
        # 检查标签是否在有效范围内
        if targets.max() >= num_classes or targets.min() < 0:
            print(f"Invalid targets found in batch: {targets}")
            print(f"Max target: {targets.max()}, Min target: {targets.min()}")
            break
        outputs = chen(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optim.zero_grad()  # 梯度清零
        loss.backward()    # 反向传播
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第 {total_train_step} 次训练的 loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    end_time = time.time()
    print(f"训练时间: {end_time - start_time}")

    # 测试步骤（以测试数据上的正确率来评估模型）
    chen.eval()  # 设置模型为评估模式
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            # 检查测试集标签
            if targets.max() >= num_classes or targets.min() < 0:
                print(f"Invalid targets found in test batch: {targets}")
                print(f"Max target: {targets.max()}, Min target: {targets.min()}")
                break
            outputs = chen(imgs)
            # 损失
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            # 正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的 loss: {total_test_loss}")
    print(f"整体测试集上的正确率: {total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # 保存每一轮训练模型
    model_save_path = os.path.join(r"D:\python.learm\shixun\day4\model_save", f"chen_{i}.pth")
    torch.save(chen, model_save_path)
    print(f"模型已保存: {model_save_path}")

writer.close()