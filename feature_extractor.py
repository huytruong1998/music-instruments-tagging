import librosa
import numpy as np
from pathlib import Path
import h5py
import csv
import os
import json

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

        gt = npz["Y_true"]
        keys = npz["sample_key"]
         
        self.__labels = []

        for idx in range(len(gt)):
            self.__labels.append((keys[idx], gt[idx] >= 0.5))


    def extract(self):
        print("ssss")
        wave_files = self.__waveforms_path.glob(r"[.]ogg")

        for wave_file in wave_files:
            print(wave_file)



        for idx, wave_file in enumerate(wave_files):
            print(wave_files)    
            
            wave, sr_ = librosa.load(wave_file, sr=None)

            if sr_ != self.__sr:
                wave = librosa.resample(y=wave, orig_sr=sr_, target_sr=self.__sr)
        
            melspec = librosa.feature.melspectrogram(
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
            )

            self.__db_path.mkdir(666, exist_ok=True)

            with h5py.File(f"{wave_file.stem}.h5", "w") as f:
                f.create_dataset("melspec", data=melspec)
                f.create_dataset("labels", data=self.__labels[idx])

                

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