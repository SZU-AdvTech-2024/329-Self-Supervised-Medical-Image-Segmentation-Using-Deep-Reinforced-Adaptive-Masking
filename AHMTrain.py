import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from unet import UNet


# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image


# 定义图像预处理变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载数据
data_dir = 'data/皮肤镜'
dataset = CustomDataset(root_dir=data_dir, transform=transform)
n_val = int(len(dataset) * 0.4)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=True)


# 定义网络结构
class ConvNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ConvNet, self).__init__()
        vgg16 = models.vgg16(pretrained=True)

        self.feature_extractor = nn.Sequential(
            *list(vgg16.features.children())[:17],
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.policy_network = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.value_network = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        policy = self.policy_network(features)
        value = self.value_network(features)
        # print(policy.shape)
        # print(value.shape)
        return policy, value


# 定义训练过程
def train_model(conv_model, unet_model, train_loader, val_loader, fea_optimizer, pol_optimizer, val_optimizer, unet_optimizer, num_epochs=25):
    best_loss = float('inf')

    for epoch in range(num_epochs):
        conv_model.train()
        unet_model.train()
        running_loss_policy1 = 0.0
        running_loss_value1 = 0.0
        running_loss_feature1 = 0.0
        running_loss_unet1 = 0.0
        running_loss_policy2 = 0.0
        running_loss_value2 = 0.0
        running_loss_feature2 = 0.0
        running_loss_unet2 = 0.0

        for inputs in train_loader:
            inputs = inputs.to(device)
            inputs.requires_grad = True

            # 更新策略网络和价值网络的梯度
            fea_optimizer.zero_grad()
            pol_optimizer.zero_grad()
            value_optimizer.zero_grad()

            policy, value = conv_model(inputs)

            policyC = policy.to('cpu')
            valueC = value.to('cpu')
            policyC = policyC.detach().numpy()
            valueC = valueC.detach().numpy()
            blocks_per_row = 16
            blocks_per_col = 16
            block_size = 14

            # 创建一个空的 224x224 的掩码
            mask = np.zeros((224, 224))

            # 将每个 14x14 的特征图块放置到 224x224 的掩码中
            for i in range(blocks_per_row):
                for j in range(blocks_per_col):
                    block = policyC[0][i * blocks_per_col + j]
                    mask[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = block

            threshold = 0.5  # 设定阈值
            binary_mask = (mask > threshold).astype(np.uint8)

            # 生成一个与掩码相同大小的随机矩阵
            random_matrix = np.random.rand(224, 224)

            # 设定30%的概率不进行掩码
            probability_threshold = 0.3
            binary_mask[random_matrix < probability_threshold] = 1

            inputsC = inputs.to('cpu')
            inputsC = inputsC.detach().numpy()
            image_array = np.array(inputsC[0])

            for i in range(3):  # 对RGB三个通道分别处理
                image_array[i,:,:] = image_array[i,:,:] * binary_mask
            masked_image = image_array

            masked_image = np.expand_dims(masked_image, axis=0)
            masked_image = torch.from_numpy(masked_image.astype(np.float32))

            masked_image = masked_image.to(device=device)

            # 使用 U-Net 重建网络
            unet_optimizer.zero_grad()
            reconstructed_inputs = unet_model(masked_image)

            unet_loss = nn.MSELoss()(reconstructed_inputs, inputs)
            unet_loss.backward()
            unet_optimizer.step()

            reconstructed_inputs = reconstructed_inputs.to('cpu')
            reconstructed_inputs = reconstructed_inputs.detach().numpy()
            # print(f"reconstructed_inputs shape {reconstructed_inputs.shape}")
            L_rec = np.linalg.norm(reconstructed_inputs - inputsC, axis=1)
            r = np.mean(L_rec)
            r_n = np.full_like(valueC, r)

            # print(f"r_n:{r_n.shape}")

            A = r_n - valueC

            delta_theta_v = torch.mean(torch.from_numpy(A ** 2))
            delta_theta_v.requires_grad = True

            # print(f"deltaV:{delta_theta_v}")

            delta_theta_p = -torch.mean(torch.from_numpy(np.log(policyC) * A))
            delta_theta_p.requires_grad = True

            # print(f"deltaP:{delta_theta_p}")

            delta_theta_f = delta_theta_p+delta_theta_v

            delta_theta_v.backward(retain_graph=True)
            delta_theta_p.backward(retain_graph=True)
            delta_theta_f.backward()


            # 总损失
            fea_optimizer.step()
            pol_optimizer.step()
            value_optimizer.step()

            running_loss_policy1 += delta_theta_v
            running_loss_value1 += delta_theta_p
            running_loss_feature1 += delta_theta_f
            running_loss_unet1 += unet_loss

        epoch_loss_policy1 = running_loss_policy1 / len(train_loader.dataset)
        epoch_loss_value1 = running_loss_value1 / len(train_loader.dataset)
        epoch_loss_feature1 = running_loss_feature1 / len(train_loader.dataset)
        epoch_loss_unet1 = running_loss_unet1 / len(train_loader.dataset)

        print(f'train: Epoch {epoch+1}/{num_epochs} Policy Loss: {epoch_loss_policy1:.4f}, Value Loss: {epoch_loss_value1:.4f},'
              f'feature Loss: {epoch_loss_feature1:.4f}, unet Loss: {epoch_loss_unet1:.4f}')

        conv_model.eval()
        unet_model.eval()

        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device)

                policy, value = conv_model(inputs)
                policyC = policy.to('cpu')
                valueC = value.to('cpu')
                policyC = policyC.detach().numpy()
                valueC = valueC.detach().numpy()
                blocks_per_row = 16
                blocks_per_col = 16
                block_size = 14

                # 创建一个空的 224x224 的掩码
                mask = np.zeros((224, 224))

                # 将每个 14x14 的特征图块放置到 224x224 的掩码中
                for i in range(blocks_per_row):
                    for j in range(blocks_per_col):
                        block = policyC[0][i * blocks_per_col + j]
                        mask[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = block

                threshold = 0.5  # 设定阈值
                binary_mask = (mask > threshold).astype(np.uint8)

                # 生成一个与掩码相同大小的随机矩阵
                random_matrix = np.random.rand(224, 224)

                # 设定30%的概率不进行掩码
                probability_threshold = 0.3
                binary_mask[random_matrix < probability_threshold] = 1

                inputsC = inputs.to('cpu')
                inputsC = inputsC.detach().numpy()
                image_array = np.array(inputsC[0])

                for i in range(3):  # 对RGB三个通道分别处理
                    image_array[i, :, :] = image_array[i, :, :] * binary_mask
                masked_image = image_array

                masked_image = np.expand_dims(masked_image, axis=0)
                masked_image = torch.from_numpy(masked_image.astype(np.float32))

                masked_image = masked_image.to(device=device)

                # 使用 U-Net 重建网络
                reconstructed_inputs = unet_model(masked_image)

                unet_loss = nn.MSELoss()(reconstructed_inputs, inputs)

                reconstructed_inputs = reconstructed_inputs.to('cpu')
                reconstructed_inputs = reconstructed_inputs.detach().numpy()
                # print(f"reconstructed_inputs shape {reconstructed_inputs.shape}")
                L_rec = np.linalg.norm(reconstructed_inputs - inputsC, axis=1)
                r = np.mean(L_rec)
                r_n = np.full_like(valueC, r)
                A = r_n - valueC

                delta_theta_v = np.mean((r_n - valueC) ** 2)
                delta_theta_p = -np.mean(np.log(policyC) * A)
                delta_theta_f = delta_theta_p + delta_theta_v

                running_loss_policy2 += delta_theta_v
                running_loss_value2 += delta_theta_p
                running_loss_feature2 += delta_theta_f
                running_loss_unet2 += unet_loss

        epoch_loss_policy2 = running_loss_policy2 / len(val_loader.dataset)
        epoch_loss_value2 = running_loss_value2 / len(val_loader.dataset)
        epoch_loss_feature2 = running_loss_feature2 / len(val_loader.dataset)
        epoch_loss_unet2 = running_loss_unet2 / len(val_loader.dataset)


        print(f'eval: Epoch {epoch+1}/{num_epochs} Policy Loss: {epoch_loss_policy2:.4f}, Value Loss: {epoch_loss_value2:.4f},'
              f'feature Loss: {epoch_loss_feature2:.4f}, unet Loss: {epoch_loss_unet2:.4f}')

        # 保存最好的模型参数
        if epoch_loss_feature1 < best_loss:
            best_loss = epoch_loss_feature1
            torch.save(unet_model.state_dict(), 'checkpoints/result.pth')


# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

unet = UNet(n_channels=3, n_classes=3).to(device)

# 实例化特征提取网络、策略网络和价值网络
conv_model = ConvNet(input_channels=3, num_classes=3).to(device)
fea_optimizer = optim.Adam(conv_model.feature_extractor.parameters(), lr=0.001)
pol_optimizer = optim.Adam(conv_model.policy_network.parameters(), lr=0.001)
value_optimizer = optim.Adam(conv_model.value_network.parameters(), lr=0.001)
unet_optimizer = optim.Adam(unet.parameters(), lr=0.001)

# 开始训练模型
train_model(conv_model, unet, train_loader, val_loader, fea_optimizer, pol_optimizer, value_optimizer, unet_optimizer, num_epochs=100)
