from torch.utils.data import Dataset
from pathlib import Path
import h5py
import numpy as np

class OpenMICDataset(Dataset):

    def __init__(self, db_path: Path, split: str, lim: int = 0, in_mem: bool = False):
        
        self.__db_filenames: list = list((db_path / split).glob(r"*[.]h5"))
        if lim != 0:
            self.__db_filenames = self.__db_filenames[:lim]

        self.__in_mem = in_mem

        if in_mem:
            self.__data = []
            
            for filename in self.__db_filenames:
                with h5py.File(filename, 'r') as f:
                    self.__data.append((f["melspec"][()], f["labels"][()].astype(np.float32)))
            

    def __len__(self):
        return len(self.__db_filenames)

    def __getitem__(self, idx: int):

        if not self.__in_mem:
            with h5py.File(self.__db_filenames[idx], 'r') as f:
                return (f["melspec"][()], f["labels"][()].astype(np.float32))
        else:
            return self.__data[idx]
        

if __name__ == "__main__":
    ds = OpenMICDataset(Path("./openmic-2018/h5"), "training")