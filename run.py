import sys

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

def classify(image):
    device = torch.device('cpu')
    model = main.Net()
    save_path = "trained_model"
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)

    img = Image.open(image)
    if not img.mode == 'RGB':
        img = img.convert('RGB')
    # img.save("website.jpg")
    # img = Image.open("website.jpg")
    img = transform(img)
    img = img.reshape(1, 3, 250, 250)
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    print(f'{classes[predicted]}')
    return classes[predicted]

if __name__ == '__main__':
    device = torch.device('cpu')
    model = main.Net()
    save_path = "trained_model"
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)

    img = Image.open(sys.argv[1])
    img.save("website.jpg")
    img = Image.open("website.jpg")
    img = transform(img)
    img = img.reshape(1, 3, 250, 250)
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    print(f'{classes[predicted]}')