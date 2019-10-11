from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from config import opt


class IcvDataset(Dataset):
    def __init__(self, total_data, transforms=None, train=True, test=False):
        self.train = train
        self.test = test
        imgs = []
        if self.test:
            for index, row in total_data.iterrows():
                imgs.append((row['filename']))
            self.imgs = imgs
        else:
            for index, row in total_data.iterrows():
                # print((row["filename"], row["label"]))
                imgs.append((row["filename"], row["label"]))
            self.imgs = imgs
        if transforms is None:
            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            if self.test or not self.train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.ToTensor(),
                    normalize
                ])
        else:
            self.transforms = transforms

    # 对numpy加载的图片数据有时候会出现异常,使用PIL
    def __getitem__(self, index):
        if self.test:
            filename = self.imgs[index]
            img = Image.open(filename)
            img = img.convert("RGB")
            img = self.transforms(img)
            return img, filename
        else:
            filename, label = self.imgs[index]
            img = Image.open(filename)
            # print(filename, img.mode)
            img = img.convert("RGB")
            img = self.transforms(img)
            return img, opt.label_index_dict[label]

    def __len__(self):
        return len(self.imgs)
