import torch.cuda
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(250),
    transforms.ToTensor()
])


MODEL_SAVE_PATH = "trained_model"


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(55696, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def open_image(image_path):
    image = Image.open(image_path)
    if not image.mode == 'RGB':
        image = image.convert('RGB')
    image = transform(image)
    image = image.reshape(1, 3, 250, 250)
    if torch.cuda.is_available():
        image = image.cuda()
    return image

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

def classify(image_path):
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device('cuda')
    model = Net()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    img = open_image(image_path)

    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    return classes[predicted]