import os

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import util

NUM_OF_EPOCHS = 1
LR = 0.0001


def train(net):
    optimizer = Adam(net.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    if torch.cuda.is_available():
        net = net.cuda()
        criterion = criterion.cuda()

    sample_submission = pd.read_csv("train_sample.csv")

    for epoch in range(NUM_OF_EPOCHS):
        net.train()
        for tuple in sample_submission.itertuples():
            i = tuple.id
            path = os.path.join('data', 'train', str(i) + '.jpg')
            img = util.open_image(path)
            label = sample_submission.label[i - 1]

            optimizer.zero_grad()
            output_train = net(img)
            if torch.cuda.is_available():
                output_real = torch.tensor([label], device="cuda:0")
            else:
                output_real = torch.tensor([label])

            loss_train = criterion(output_train, output_real)
            loss_train.backward()
            optimizer.step()
            if epoch % 10 == 0 or i % 50 == 0:
                print(f'Epoch: {epoch} \t loss: {loss_train}')


def save_model(net, path):
    torch.save(net.state_dict(), path)


if __name__ == '__main__':
    net = util.Net()
    try:
        train(net)
        save_model(net, util.MODEL_SAVE_PATH)
    except KeyboardInterrupt:
        save_model(net, util.MODEL_SAVE_PATH)
