import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision import transforms
from PIL import Image


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc1 = nn.Linear(345744, 120)
        # self.fc1 = nn.Linear(160, 120)
        self.fc1 = nn.Linear(55696, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # self.fc4 = nn.Linear(10, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#
# transform = transforms.Compose([
#         # transforms.CenterCrop(512),
#         # transforms.Resize(448),
#         transforms.Resize(20),
#         transforms.ToTensor()
#     ])

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(250),
        # transforms.Resize(448),
        transforms.ToTensor()
    ])

train_losses=[]

classes = ('www.theguardian.com',
            'www.spiegel.de',
            'www.cnn.com',
            'www.bbc.com',
            'www.amazon.com',
            'www.ebay.com',
            'www.njuskalo.hr',
            'www.google.com',
            'www.github.com',
            'www.youtube.com')

def train(net):
    # net = Net()
    optimizer = Adam(net.parameters(), lr=0.0001)
    criterion = CrossEntropyLoss()
    if torch.cuda.is_available():
        net = net.cuda()
        criterion = criterion.cuda()

    sample_submission = pd.read_csv("train_sample.csv")
    # print(sample_submission.label)

    for epoch in range(15):
        net.train()
        # ids = sample_submission.id.to_string(index=False)
        # print(ids)
        # for i in tqdm(sample_submission.id):
        for tuple in sample_submission.itertuples():
            i = tuple.id
            path = os.path.join('data', 'train', str(i) + '.jpg')
            # print(path)
            img = Image.open(path)
            # print(img)
            img = transform(img)
            # print(img.shape)
            # img = img.reshape(1, 3, 600, 600)
            img = img.reshape(1, 3, 250, 250)
            if torch.cuda.is_available():
                img = img.cuda()
            label = sample_submission.label[i-1]
            # print(label)

            optimizer.zero_grad()
            output_train = net(img)
            if torch.cuda.is_available():
                output_real = torch.tensor([label], device="cuda:0")
            else:
                output_real = torch.tensor([label])
            # b = np.zeros(2)
            # b[label] = 1
            # output_real = torch.tensor(b, device="cuda:0")
            # output_real = Variable(label)
            # print(output_real)
            # print(output_train)
            loss_train = criterion(output_train, output_real)
            loss_train.backward()
            optimizer.step()
            train_losses.append(loss_train.cpu().detach().numpy())
            if epoch % 10 == 0 or i % 50 == 0:
                print(f'Epoch: {epoch} \t loss: {loss_train}')
    img = Image.open("data/train/test.jpg")
    # print(img)
    img = transform(img)
    # print(img.shape)
    img = img.reshape(1, 3, 250, 250)
    # img = img.cuda()
    outputs = net(img)
    _, predicted = torch.max(outputs, 1)
    print("KAK KLASIFICIRA OVO? (njuskalo)", F.softmax(net(img)), classes[predicted])

    img = Image.open("data/train/test2.jpg")
    # print(img)
    img = transform(img)
    # print(img.shape)
    img = img.reshape(1, 3, 250, 250)
    if torch.cuda.is_available():
        img = img.cuda()
    outputs = net(img)
    _, predicted = torch.max(outputs, 1)
    print("KAK KLASIFICIRA OVO? (ebay)", F.softmax(net(img)), classes[predicted])

def save_model(net, path):
    torch.save(net.state_dict(), path)

if __name__ == '__main__':
    # for clas in classes:
    #     fq = FQDN(clas)
    #     print(fq, fq.is_valid, fq.is_valid_absolute, fq.absolute, fq.relative)
    save_path = "trained_model"
    net = Net()
    try:
        train(net)
        save_model(net, save_path)
    except KeyboardInterrupt:
        save_model(net, save_path)

    plt.plot(train_losses, label='Training loss')
    plt.legend()
    plt.show()