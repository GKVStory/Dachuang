import os.path
from os.path import join
from pathlib import Path
from torch import nn
from django.shortcuts import render
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# 这部分用于上传等功能的实现
def upload_file(request):
    if request.method == "POST":  # 请求方法为POST时，进行处理
        myFile = request.FILES.get("myfile", None)  # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return "no files for upload!"
        tmp_path = os.path.join(BASE_DIR, 'static/data/JPEGImages')
        destination = open(os.path.join(tmp_path, myFile.name), 'wb+')  # 打开特定的文件进行二进制的写操作
        for i in myFile.chunks():  # 分块写入文件
            destination.write(i)
        destination.close()
        return "已成功标注！"


# 这部分用于路由功能的实现
def test(request):
    result = upload_file(request)
    return render(request, 'test.html', {'result': result})


def home(request):
    return render(request, 'home.html')


def compare(request):
    return render(request, 'compare.html')


def auto(request):
    if request.method == "POST":
        result = upload_file(request)
        auto_function()
        return render(request, 'auto.html', {'result': result})
    return render(request, 'auto.html')

def sort(request):
    return render(request, 'sort.html')


def auto_function():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet().to(device)

    weights = 'static/params/unet.pth'
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully')
    else:
        print('no loading')

    test_path = 'static/data/JPEGImages'  # 测试图片存放路径
    result_path = 'static/results'  # 结果保存路径
    test = os.listdir(test_path)
    test.sort(key=lambda x: int(x[:-4]))
    test_name_list = [x.split('.')[0] for x in test]
    test_img = [join(test_path, x + ".png") for x in test_name_list]

    for ind in range(len(test_img)):
        img = Keep_image_size_open(test_img[ind])
        img_data = transform(img).to(device)
        img_data = torch.unsqueeze(img_data, dim=0)
        out = net(img_data)
        save_image(out, f'{result_path}/{ind}.png')


transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))
        self.name.sort(key=lambda x: int(x[:-4]))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # png
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))
        segment_image = Keep_image_size_open(segment_path)
        image = Keep_image_size_open(image_path)
        return transform(image), transform(segment_image)


def data():
    data = MyDataset('./data')
    print(data[0][0].shape)
    print(data[0][1].shape)


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):  # 下采样
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):  # 上采样
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)

        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)

        self.out = nn.Conv2d(64, 3, 3, 1, 1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))

        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.Th(self.out(O4))


def data():
    x = torch.randn(2, 3, 256, 256)
    net = UNet()
    print(net(x).shape)


from PIL import Image


def Keep_image_size_open(path, size=(256, 256)):  # 等比缩放
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask
# # compare part
#
# weight_path = os.path.join(BASE_DIR, 'static\\params\\unet.pth')
# data_path = os.path.join(BASE_DIR, 'static\\data')
# save_path = os.path.join(BASE_DIR, 'static\\predict')
# seg_path = os.path.join(BASE_DIR, 'static\\segment')
# pred_path = os.path.join(BASE_DIR, 'static\\predict')
# img_path = os.path.join(BASE_DIR, 'static\\image')
#
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# class MyDataset(Dataset):
#     def __init__(self, path):
#         self.path = path
#         self.name = os.listdir(os.path.join(path, 'SegmentationClass'))
#         self.name.sort(key=lambda x: int)
#
#     def __len__(self):
#         return len(self.name)
#
#     def __getitem__(self, index):
#         segment_name = self.name[index]  # png
#         segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
#         image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))
#         segment_image = Keep_image_size_open(segment_path)
#         image = Keep_image_size_open(image_path)
#         return transform(image), transform(segment_image)
#
#
# class Conv_Block(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(Conv_Block, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.Dropout2d(0.3),
#             nn.LeakyReLU(),
#
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.Dropout2d(0.3),
#             nn.LeakyReLU()
#         )
#
#     def forward(self, x):
#         return self.layer(x)
#
#
# class DownSample(nn.Module):  # 下采样
#     def __init__(self, channel):
#         super(DownSample, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
#             nn.BatchNorm2d(channel),
#             nn.LeakyReLU()
#         )
#
#     def forward(self, x):
#         return self.layer(x)
#
#
# class UpSample(nn.Module):  # 上采样
#     def __init__(self, channel):
#         super(UpSample, self).__init__()
#         self.layer = nn.Conv2d(channel, channel // 2, 1, 1)
#
#     def forward(self, x, feature_map):
#         up = F.interpolate(x, scale_factor=2, mode='nearest')
#         out = self.layer(up)
#         return torch.cat((out, feature_map), dim=1)
#
#
# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         self.c1 = Conv_Block(3, 64)
#         self.d1 = DownSample(64)
#         self.c2 = Conv_Block(64, 128)
#         self.d2 = DownSample(128)
#         self.c3 = Conv_Block(128, 256)
#         self.d3 = DownSample(256)
#         self.c4 = Conv_Block(256, 512)
#         self.d4 = DownSample(512)
#         self.c5 = Conv_Block(512, 1024)
#
#         self.u1 = UpSample(1024)
#         self.c6 = Conv_Block(1024, 512)
#         self.u2 = UpSample(512)
#         self.c7 = Conv_Block(512, 256)
#         self.u3 = UpSample(256)
#         self.c8 = Conv_Block(256, 128)
#         self.u4 = UpSample(128)
#         self.c9 = Conv_Block(128, 64)
#
#         self.out = nn.Conv2d(64, 3, 3, 1, 1)
#         self.Th = nn.Sigmoid()
#
#     def forward(self, x):
#         R1 = self.c1(x)
#         R2 = self.c2(self.d1(R1))
#         R3 = self.c3(self.d2(R2))
#         R4 = self.c4(self.d3(R3))
#         R5 = self.c5(self.d4(R4))
#
#         O1 = self.c6(self.u1(R5, R4))
#         O2 = self.c7(self.u2(O1, R3))
#         O3 = self.c8(self.u3(O2, R2))
#         O4 = self.c9(self.u4(O3, R1))
#
#         return self.Th(self.out(O4))
#
#
# def data():
#     data = MyDataset('../static/data')
#     print(data[0][0].shape)
#     print(data[0][1].shape)
#
#
# def Keep_image_size_open(path, size=(256, 256)):  # 等比缩放
#     img = Image.open(path)
#     temp = np.max(torch.tensor(img.size))
#     mask = Image.new('RGB', (temp, temp), (0, 0, 0))
#     mask.paste(img, (0, 0))
#     mask = mask.resize(size)
#     return mask
#
#
# def net():
#     x = torch.randn(2, 3, 256, 256)
#     net = UNet()
#     print(net(x).shape)
#
#
# def predict():
#     data_loader = DataLoader(MyDataset(data_path), batch_size=1, shuffle=True)
#     net = UNet().to(device)
#     if os.path.exists(weight_path):
#         net.load_state_dict(torch.load(weight_path, map_location='cpu'))
#         print('successful load weight！')
#     else:
#         print('not successful load weight')
#
#     opt = optim.Adam(net.parameters())
#     loss_fun = nn.BCELoss()
#
#     epoch = 1
#     while True:
#         for i, (image, segment_image) in enumerate(data_loader):
#             image, segment_image = image.to(device), segment_image.to(device)
#
#             out_image = net(image)
#             train_loss = loss_fun(out_image, segment_image)
#
#             opt.zero_grad()
#             train_loss.backward()
#             opt.step()
#
#             if i % 5 == 0:
#                 print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')
#
#             if i % 50 == 0:
#                 torch.save(net.state_dict(), weight_path)
#
#             _segment_image = segment_image[0]
#             _out_image = out_image[0]
#             seg = torch.stack([_segment_image], dim=0)
#             img = torch.stack([_out_image], dim=0)
#             save_image(img, f'{save_path}/{i}.png')
#             save_image(seg, f'{seg_path}/{i}.png')
#         epoch += 1
#         if epoch >= 21:
#             break
#
#
# def train():
#     data_loader = DataLoader(MyDataset(data_path), batch_size=2, shuffle=True)
#     net = UNet().to(device)
#     if os.path.exists(weight_path):
#         net.load_state_dict(torch.load(weight_path, map_location='cpu'))
#         print('successful load weight！')
#     else:
#         print('not successful load weight')
#
#     opt = optim.Adam(net.parameters())
#     loss_fun = nn.BCELoss()
#
#     epoch = 1
#     while True:
#         for i, (image, segment_image) in enumerate(data_loader):
#             image, segment_image = image.to(device), segment_image.to(device)
#
#             out_image = net(image)
#             train_loss = loss_fun(out_image, segment_image)
#
#             opt.zero_grad()
#             train_loss.backward()
#             opt.step()
#
#             if i % 5 == 0:  # 隔5次打印一次权重
#                 print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')
#
#             if i % 50 == 0:  # 隔50次保存一次权重
#                 torch.save(net.state_dict(), weight_path)
#
#             _image = image[0]
#             _segment_image = segment_image[0]
#             _out_image = out_image[0]
#
#             img = torch.stack([_image, _segment_image, _out_image], dim=0)
#             pred = torch.stack([_out_image], dim=0)
#             seg = torch.stack([_segment_image], dim=0)
#             image = torch.stack([_image], dim=0)
#             save_image(image, f'{img_path}/{i}.png')
#             save_image(seg, f'{seg_path}/{i}.png')
#             save_image(img, f'{save_path}/{i}.png')
#             save_image(pred, f'{pred_path}/{i}.png')
#         epoch += 1
#         if epoch >= 101:
#             break
