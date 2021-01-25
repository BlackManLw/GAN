import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torch

#其实GAN的本质上是学习到了一个数据的分布问题

#随机产生一个正态分布
#np.random.normal(loc,scale,size)  生成一个正态分布：loc代表均值，scale——代表标准差；size代表数据生成的尺寸
X = np.random.normal(size=(1000, 2))
A = np.array([[1, 2], [-0.1, 0.5]])
b = np.array([1, 2])
data = X.dot(A)+b

# plt.figure(figsize=(3.5,2.5))
# plt.scatter(X[:100,0],X[:100,1],color='red')
# plt.show()
# plt.figure(figsize=(3.5,2.5))
# plt.scatter(data[:100,0],data[:100,1],color='blue')
# plt.show()

#生成数据并且定义batch_size和dataloader
batch_size = 8
data_iter = DataLoader(data, batch_size=batch_size)

#生成器
class net_G(nn.Module):
    def __init__(self):
        super(net_G, self).__init__()
        #一个linear层，生成的数据是2维
        self.model = nn.Sequential(
            nn.Linear(2, 2),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    #初始化服从高斯分布
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

#鉴别器，三个linear层+激活函数Sigmoid
class net_D(nn.Module):
    def __init__(self):
        super(net_D, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 5),
            nn.Tanh(),
            nn.Linear(5, 3),
            nn.Tanh(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

#更新鉴别器的参数
#X真实数据；Z随机产生的参数，送到生成器中将会得到虚假数据
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    batch_size = X.shape[0]
    #生成全1数据和全0数据这写都是标签
    ones = torch.ones(batch_size).view(batch_size, 1)
    zeros = torch.zeros(batch_size).view(batch_size, 1)
    real_Y = net_D(X)
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X)
    #让真数据和真数据的标签尽可能想接近，假数据和假数据的标签尽可能接近
    loss_D = (loss(real_Y, ones)+loss(fake_Y, zeros))/2
    loss_D.backward()
    trainer_D.step()
    return loss_D.sum()

#更新生成器：需要注意的是损失函数是不一样的
def update_G(Z, net_D, net_G, loss, trainer_G):
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,)).view(batch_size, 1)
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X)
    #需要让生成器的假数据和真数据标签尽可能的接近
    loss_G = loss(fake_Y, ones)
    #进行反向传播之后
    loss_G.backward()
    #更新参数
    trainer_G.step()
    return loss_G.sum()

#训练GAN
#每次先迭代更新鉴别器再更新生成器
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = nn.BCELoss()  # 二分类
    #采用adam优化器
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    plt.figure(figsize=(7, 4))
    d_loss_point = []
    g_loss_point = []
    d_loss = 0
    g_loss = 0
    for epoch in range(1, num_epochs+1):
        print('epoch:',epoch)
        d_loss_sum = 0
        g_loss_sum = 0
        batch = 0
        for X in data_iter:
            batch += 1
            X = torch.tensor(X, dtype=torch.float32)
            batch_size = X.shape[0]
            #随机初始化噪声数据,生成的数据是服从正太分布的
            Z = torch.tensor(np.random.normal(
                0, 1, (batch_size, latent_dim)), dtype=torch.float32)
            
            #更新鉴别器
            trainer_D.zero_grad()
            d_loss = update_D(X, Z, net_D, net_G, loss, trainer_D)
            d_loss_sum += d_loss
            
            #更新生成器
            trainer_G.zero_grad()
            g_loss = update_G(Z, net_D, net_G, loss, trainer_G)
            g_loss_sum += g_loss
        
        #将每个epoch的loss都添加到相应的列表中去
        d_loss_point.append(d_loss_sum/batch)
        g_loss_point.append(g_loss_sum/batch)

    plt.ylabel('Loss', fontdict={'size': 14})
    plt.xlabel('epoch', fontdict={'size': 14})
    #指定刻度
    plt.xticks(range(0, num_epochs+1, 3))
    plt.plot(range(1, num_epochs+1), d_loss_point,
             color='orange', label='discriminator')
    plt.plot(range(1, num_epochs+1), g_loss_point,
             color='blue', label='generator')
    #加上图例
    plt.legend()
    plt.show()
    print(d_loss, g_loss)
    
    #经过训练之后，然后随机生成一个噪声，然后开始迭代通过生成器生成一个fake的数据
    Z = torch.tensor(np.random.normal(
        0, 1, size=(100, latent_dim)), dtype=torch.float32)
    fake_X = net_G(Z).detach().numpy()
    plt.figure(figsize=(5, 4))
    plt.scatter(data[:, 0], data[:, 1], color='blue', label='real')
    plt.scatter(fake_X[:, 0], fake_X[:, 1], color='orange', label='generated')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 50
    generator = net_G()
    discriminator = net_D()
    train(discriminator, generator, data_iter, num_epochs, lr_D, lr_G, latent_dim, data)