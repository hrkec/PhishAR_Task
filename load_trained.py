import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

import main

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

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(250),
        # transforms.Resize(448),
        transforms.ToTensor()
    ])

if __name__ == '__main__':
    device = torch.device('cuda')
    model = main.Net()
    save_path = "trained_model"
    model.load_state_dict(torch.load(save_path, map_location="cuda:0"))
    model.to(device)

    img = Image.open("data/train/test_mob.jpg")
    # print(img)
    img = transform(img)
    # print(img.shape)
    img = img.reshape(1, 3, 250, 250)
    img = img.cuda()
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    print("KAK KLASIFICIRA OVO?", F.softmax(model(img)), classes[predicted])