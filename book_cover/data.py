from lightning import LightningDataModule
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms


def delete_alpha_chanel(x: torch.Tensor):
    if x.shape[2] > 3:
        return x[:, :, :3]
    elif x.shape[2] < 3:
        return x.expand(x.shape[0], x.shape[1], 3)
    return x


def permute_dimensions(x: torch.Tensor):
    return x.permute(2, 1, 0)  # w, h, c -> c, h, w


class Preprocessing:
    def __init__(self, image_size: int, grayscale: bool, augmentation: bool):
        super(Preprocessing, self).__init__()
        self.image_size = image_size
        self.grayscale = grayscale
        self.augmentation = augmentation
        t = [
            delete_alpha_chanel,
            permute_dimensions,
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ConvertImageDtype(torch.float)
        ]
        if self.grayscale:
            t.append(transforms.Grayscale())
        t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        if self.augmentation:
            t = t + [transforms.RandomChoice([
                transforms.RandomRotation((-10, 10), center=(self.image_size//2, self.image_size//2)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomErasing(p=0.1, scale=(0.1, 0.1)),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5)
            ], p=(1, 1, 1, 1, 1))]
        self.transform = transforms.Compose(t)

    def collate_fn(self, batch):
        images = []
        labels = []
        titles = []

        for datapoint in batch:
            labels.append(datapoint['label'])
            titles.append(datapoint['title'])
            img = self.transform(datapoint['image'])
            images.append(img)

        images = torch.stack(images)
        labels = torch.stack(labels)
        return {'pixel_values': images, 'labels': labels}


class BookCovers(LightningDataModule):
    def __init__(
            self,
            token: str = None,
            image_size: int = 256,
            grayscale: bool = False,
            augmentation: bool = False,
            debug: bool = False,
            cache_dir: str = None,
            batch_size: int = 16
    ):
        super(BookCovers, self).__init__()
        self.token = token
        self.image_size = image_size
        self.grayscale = grayscale
        self.augmentation = augmentation
        self.debug = debug
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.dataset = None
        self.preprocessing = Preprocessing(
            image_size=self.image_size,
            grayscale=self.grayscale,
            augmentation=self.augmentation
        )

    def prepare_data(self):
        dataset = load_dataset('tomaviktor/amazon-book-cover',
                               token=self.token,
                               split='train[:1024]' if self.debug else 'train',
                               cache_dir=self.cache_dir)
        dataset = DatasetDict({'train': dataset})
        train_size = len(dataset['train'])
        dataset = dataset['train'].train_test_split(train_size=0.8, shuffle=True, seed=0)
        ds = dataset['train'].train_test_split(train_size=int(train_size*0.7), shuffle=True, seed=0)
        dataset['train'] = ds['train']
        dataset['validation'] = ds['test']
        dataset.set_format('torch')
        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'],
            shuffle=True,
            pin_memory=True,
            batch_size=self.batch_size,
            collate_fn=self.preprocessing.collate_fn,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset['validation'],
            shuffle=True,
            pin_memory=True,
            batch_size=self.batch_size,
            collate_fn=self.preprocessing.collate_fn,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset['test'],
            shuffle=True,
            pin_memory=True,
            batch_size=self.batch_size,
            collate_fn=self.preprocessing.collate_fn,
            drop_last=True
        )
