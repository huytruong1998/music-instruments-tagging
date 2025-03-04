import librosa
import numpy as np
from pathlib import Path
import h5py
import csv
import os
import json
import pandas as pd
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, 
        data_root_path: str,
        waveforms_path: str,
        npz_path: str,
        db_path: str,
        sr: int, 
        n_fft: int, 
        win_length: int, 
        hop_length: int, 
        window: str, 
        n_mels: int,
    ):
        self.__sr = sr
        self.__n_fft = n_fft
        self.__win_length = win_length
        self.__hop_length = hop_length
        self.__window = window
        self.__n_mels = n_mels

        self.__data_root_path = Path(data_root_path)
        self.__waveforms_path = Path(data_root_path, waveforms_path)
        self.__db_path = Path(data_root_path, db_path)

        npz = np.load(Path(data_root_path, npz_path), allow_pickle=True)

        y_true = npz["Y_true"]
        y_mask = npz["Y_mask"]

        valid_labels = np.where(y_mask, y_true, 0)

        keys = npz["sample_key"]
         
        self.__labels = []
        for idx in range(len(valid_labels)):
            self.__labels.append((keys[idx], (valid_labels[idx] >= 0.5).astype(int)))


    def extract(self):
        # wave_files = self.__waveforms_path.glob("*.ogg")
        wave_files = list(self.__waveforms_path.rglob("*.ogg"))

        #Split train and Test
        split_train = pd.read_csv(self.__data_root_path / 'partitions' /'split01_train.csv', 
                          header=None).squeeze("columns")
        split_test = pd.read_csv(self.__data_root_path / 'partitions' / 'split01_test.csv', 
                                header=None).squeeze("columns")
        
        train_set = set(split_train)
        test_set = set(split_test)
        
        #Extract feature and save h5
        for idx, wave_file in tqdm(enumerate(wave_files), desc="Extracting pairs of melspectrograms and labels...", total=len(wave_files)):
            wave, sr_ = librosa.load(wave_file, sr=None)

            if sr_ != self.__sr:
                wave = librosa.resample(y=wave, orig_sr=sr_, target_sr=self.__sr)
        
            melspec = np.log10(
                1.0 + librosa.feature.melspectrogram(
                    y=wave, 
                    sr=self.__sr,
                    n_fft=self.__n_fft,
                    hop_length=self.__hop_length,
                    win_length=self.__win_length,
                    window=self.__window,
                    center=True,
                    pad_mode='reflect',
                    power=2.0,
                    n_mels=self.__n_mels
                ),
                dtype=np.float32
            )
            
            if wave_file.stem != self.__labels[idx][0]:
                print("not equal __labels",self.__labels[idx][0])
                print("not equal wave_file.stem",wave_file.stem)
    
            self.__db_path.mkdir(666, exist_ok=True)

            if wave_file.stem in train_set:
                h5_file_path = self.__db_path / "training" 
            elif wave_file.stem in test_set:
                h5_file_path = self.__db_path / "testing"
            else:
                # This should never happen, but better safe than sorry.
                raise RuntimeError('Unknown sample key={}! Abort!'.format(wave_file.stem))
            
            h5_file_path.mkdir(666, exist_ok=True)
            
            with h5py.File(h5_file_path / f"{wave_file.stem}.h5", "w") as f:
                f.create_dataset("melspec", data=melspec)
                f.create_dataset("labels", data=self.__labels[idx][1])


if __name__ == "__main__":

    c = {}
    with open("data_config.json", "r") as conf:
        c = json.load(conf)

    ext = FeatureExtractor(
        c["data_root_path"],
        c["waveforms_path"],
        c["npz_path"],
        c["db_path"],
        c["sample_rate"],
        c["n_fft"],
        c["win_length"],
        c["hop_length"],
        c["window"],
        c["n_mels"]
    )

    ext.extract()