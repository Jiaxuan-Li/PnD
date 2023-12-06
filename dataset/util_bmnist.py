import os
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from glob import glob
from PIL import Image

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


class BmnistDataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None):
        super(BmnistDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.data  = glob(os.path.join(root, split,"*","*"))
            print(len(self.data))

        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*","*"))  
                   
        elif split=='test':
            print(os.path.join(root, split))
            self.data = glob(os.path.join(root, split,"*","*"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('/')[-2])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, self.data[index]


transforms = {
    "bmnist": {
        "train": T.Compose([T.ToTensor()]), 
        "valid": T.Compose([T.ToTensor()]), 
        "test": T.Compose([T.ToTensor()]),
        },
    }


def get_dataset(dataset, data_dir, dataset_split, transform_split, percent, use_preprocess=None, image_path_list=None):
    dataset_category = dataset.split("-")[0]
    transform = transforms[dataset_category][transform_split]
    root = data_dir + f"/bmnist/{percent}"
    dataset = BmnistDataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list)

    return dataset

