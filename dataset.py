from torch.utils.data import Dataset
import h5py
import torch

class BrainTumorDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image, target = self.load_and_preprocess_slice(row['slice_path'])
        return image, target

    def load_and_preprocess_slice(self, slice_path):
        with h5py.File(slice_path, 'r') as f:
            image = f['image'][:]
            target = f['mask'][:]

        image = torch.from_numpy(image).float()
        target = torch.from_numpy(target).float()

        return image, target