import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from vars import NUM_CLASSES


def get_datalist(ids: str, real_data_root_path: str, fake_data_root_path: str) -> list[dict]:
    df = pd.read_csv(ids, sep="\t")

    data_dicts = [
        {
            'flair': f'{real_data_root_path}/{row["flair"]}',
            'seg': f'{real_data_root_path}/{row["seg"]}'
        } if fake_data_root_path is None or row['is_real'] else {
            'flair': f'{fake_data_root_path}/{row["flair"]}',
            'seg': f'{fake_data_root_path}/{row["seg"]}'
        }
        for _, row in df.iterrows()
    ]

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def get_data_loader(
        data: list[dict], transforms: transforms.Compose, batch_size: int, evaluation: bool = False
) -> torch.utils.data.DataLoader:
    data = SegDataset(data=data, transform=transforms)
    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=not evaluation,
        pin_memory=False
    )


class SegDataset(Dataset):
    def __init__(self, data: list[dict], transform: transforms.Compose) -> None:
        self.transform = transform
        self.data = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self) -> int:
        length = len(self.data)
        return length

    def __getitem__(self, index: int) -> tuple:
        x = Image.open(self.data[index]['flair'])
        y = Image.open(self.data[index]['seg'])

        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)

        x = transforms.ToTensor()(x)
        y = transforms.ToTensor()(y)

        x = x.to(self.device)
        y = torch.ceil(y.to(self.device) * NUM_CLASSES).to(torch.long)

        return x, torch.where(y == NUM_CLASSES, NUM_CLASSES - 1, y)
