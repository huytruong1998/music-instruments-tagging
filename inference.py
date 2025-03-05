from torch import load, cuda, from_numpy, no_grad, float32
from model import Cnn14
import json
import numpy as np
import librosa
from pathlib import Path
from feature_extractor import FeatureExtractor
import pandas as pd 
import shutil
from tqdm import tqdm 

def vec_to_labels(vec, labels):
    return labels[vec == 1]

if __name__ == "__main__":

    device = "cuda" if cuda.is_available() else "cpu"
    print(device)

    model = Cnn14(20)
    
    model.load_state_dict(load("./checkpoints/ckpt-30-0.092-71.16-2025-03-04T22.53.58.pt", weights_only=False))

    model = model.to(device)

    model.eval()

    labels = ['' for _ in range(20)]

    with open("./openmic-2018/class-map.json") as f:
        js = json.load(f)

        for k, v in js.items():
            labels[v] = k
    
    labels = np.array(labels)

    c = {}
    with open("data_config.json", "r") as conf:
        c = json.load(conf)

    target_sr = c["sample_rate"]
    n_fft = c["n_fft"]
    win_length = c["win_length"]
    hop_length = c["hop_length"]
    window = c["window"]
    n_mels = c["n_mels"]

    split_test = set(pd.read_csv("./openmic-2018/partitions/split01_test.csv").squeeze("columns"))
    ds_path = Path("./openmic-2018/audio")
    tgt_path = Path("./resources/source")
    results_path = Path("./resources/results")

    results_file = open(results_path / "results.txt", mode="+at")

    # print(split_test)
    # for filename in list(ds_path.rglob(r"*[.]ogg")):

    #     if filename.stem in split_test:
    #         shutil.copy(filename, tgt_path / filename.name)

    with no_grad():

        for filename in tqdm(tgt_path.glob(r"*[.]ogg"), desc="Running inference on test samples..."):

            wave, sr = librosa.load(filename, sr=None)

            if sr != target_sr:
                librosa.resample(wave, orig_sr=sr, target_sr=target_sr)
            
            melspec = np.log10(
                1.0 + librosa.feature.melspectrogram(
                    y=wave, 
                    sr=target_sr,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    center=True,
                    pad_mode='reflect',
                    power=2.0,
                    n_mels=n_mels
                ),
                dtype=np.float32
            )
            
            melspec = from_numpy(melspec).unsqueeze(0)

            # dB feats
            melspec = melspec.unsqueeze(1).to(device)

            # OG feats
            # melspec = melspec.unsqueeze(1).transpose(2, 3).to(device)

            pred_vec = model(melspec)["clipwise_output"].cpu().squeeze(0)

            pred_vec_thr = (pred_vec > 0.5).to(float32)

            pred_lb = vec_to_labels(pred_vec_thr.numpy(), labels)

            line = filename.stem.rjust(20) + "      " + str(pred_lb) + "\n"

            results_file.write(line)
        
    
