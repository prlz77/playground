import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, ImageNet, ImageFolder
import torchvision.transforms as tt

def get_dataset(hyperparams, args):
    if hyperparams['dataset'] == 'mnist':
        return MNISTDataModule(batch_size=hyperparams['batch_size'], num_workers=args.num_workers)
    elif hyperparams['dataset'] == 'double_mnist':
        return DoubleMNISTDataModule(batch_size=hyperparams['batch_size'], num_workers=args.num_workers)
    elif hyperparams['dataset'] == 'imagenet':
        return ImageNetDataModule(batch_size=hyperparams['batch_size'], num_workers=args.num_workers)
    else:
        raise ValueError(hyperparams['dataset'])

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):
        # self.mnist_test = MNIST(self.data_dir, train=False)
        self.mnist_val = MNIST(self.data_dir, train=False, transform=tt.ToTensor())
        self.mnist_train = MNIST(self.data_dir, train=True, transform=tt.ToTensor())
        # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
        # return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

class DoubleMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data/double_mnist_seed_123_image_size_64_64", batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):
        # self.mnist_test = MNIST(self.data_dir, train=False)
        self.mnist_val = ImageFolder(f"{self.data_dir}/val", transform=tt.ToTensor())
        self.mnist_train = ImageFolder(f"{self.data_dir}/train", transform=tt.ToTensor())
        # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):
        self.train = ImageNet(self.data_dir, split='train', transform=tt.ToTensor())
        self.eval = ImageNet(self.data_dir, split='val', transform=tt.ToTensor())

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)
