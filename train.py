from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from dataset import OpenMICDataset
import json
from pathlib import Path
from model import Cnn14
from torch import cuda, no_grad, Tensor, sum, float32, uint8, unsqueeze, save, load, logical_and, nan_to_num
from tqdm import tqdm
import numpy as np
import shutil
from datetime import datetime


def train(dl_train, dl_test, max_epochs, criterion, optimizer, model, device, num_classes, ckpt_path, ckpt_period, prediction_threshold, MIN_TEST_LOSS, MAX_TEST_ACC):
    model.train()

    print("TRAINING STARTED".center(shutil.get_terminal_size().columns, "="))

    for epoch in range(1, max_epochs):
        epoch_loss = []

        for x, y in tqdm(dl_train, desc=f"Epoch {str(epoch).rjust(4, '0')} / {str(max_epochs).rjust(4, '0')}"):
            optimizer.zero_grad()

            x: Tensor = x.to(device)

            # dB feats
            x = x.unsqueeze(1)

            # OG feats
            # x = x.unsqueeze(1).transpose(2, 3)

            y: Tensor = y.to(device)

            y_pred: Tensor = model(x)['clipwise_output'] 

            loss = criterion(y_pred, y)

            loss.backward()

            optimizer.step()

            epoch_loss.append(loss.item())

        epoch_loss = np.mean(epoch_loss)
        print(f"Loss: {epoch_loss}")

        if epoch % ckpt_period == 0:
            test(dl_test, criterion, model, device, num_classes, ckpt_path, prediction_threshold, MIN_TEST_LOSS, MAX_TEST_ACC, epoch)

            model.train()


def test(dl_test, criterion, model, device, num_classes, ckpt_path: Path, prediction_threshold, MIN_TEST_LOSS, MAX_TEST_ACC, epoch):
    model.eval()

    with no_grad():
        
        test_loss = []
        test_acc = np.array([], dtype=np.float32)
        test_precision = np.array([], dtype=np.float32)
        test_recall = np.array([], dtype=np.float32)

        for x, y in tqdm(dl_test, desc="Running inference on test samples..."):
            x: Tensor = x.to(device)

            # dB feats
            x = x.unsqueeze(1)

            # OG feats
            # x = x.unsqueeze(1).transpose(2, 3)

            y: Tensor = y.to(device)

            y_pred: Tensor = model(x)['clipwise_output'] 

            loss = criterion(y_pred, y)

            test_loss.append(loss.item())

            y_pred_thr = (y_pred > prediction_threshold).to(float32)

            acc = (y_pred_thr == y).to(uint8)

            TP = (logical_and(y_pred_thr == 1, y == 1)).to(uint8)
            FN = (logical_and(y_pred_thr == 0, y == 1)).to(uint8)
            FP = (logical_and(y_pred_thr == 1, y == 0)).to(uint8)

            count_TP = sum(TP, dim=1).cpu()
            count_FN = sum(FN, dim=1).cpu()
            count_FP = sum(FP, dim=1).cpu()

            R = nan_to_num(count_TP / (count_TP + count_FN), nan=1)
            P = nan_to_num(count_TP / (count_TP + count_FP), nan=1)

            acc = sum(acc, dim=1) / num_classes

            test_acc = np.concatenate([test_acc, acc.cpu().numpy()])
            test_precision = np.concatenate([test_precision, P.cpu().numpy()])
            test_recall = np.concatenate([test_recall, R.cpu().numpy()])

        test_loss = np.mean(test_loss)

        min_test_precision = test_precision.min() * 100.0
        avg_test_precision = test_precision.mean() * 100.0

        min_test_recall = test_recall.min() * 100.0
        avg_test_recall = test_recall.mean() * 100.0

        min_test_acc = test_acc.min() * 100.0
        avg_test_acc = test_acc.mean() * 100.0

        print(f"Loss: {test_loss:.4f}")

        print(f"Precision: MIN = {min_test_precision:.2f}% AVG = {avg_test_precision:.2f}%")
        print(f"Recall: MIN = {min_test_recall:.2f}% AVG = {avg_test_recall:.2f}%")
        print(f"Accuracy: MIN = {min_test_acc:.2f}% AVG = {avg_test_acc:.2f}%")

        if avg_test_acc > MAX_TEST_ACC or (avg_test_acc == MAX_TEST_ACC and test_loss < MIN_TEST_LOSS):
            MAX_TEST_ACC = avg_test_acc
            MIN_TEST_LOSS = test_loss

            ckpt_path.mkdir(666, exist_ok=True)

            save(
                model.state_dict(),
                ckpt_path / f"ckpt-{epoch}-{test_loss:.3f}-{avg_test_recall:.2f}-{datetime.now().replace(microsecond=0).isoformat().replace(':', '.')}.pt"
            )


if __name__ == "__main__":
    MIN_TEST_LOSS = 1e10
    MAX_TEST_ACC = 0.
    
    device = "cuda" if cuda.is_available() else "cpu"
    print(device)

    config = {}

    with open("./train_config.json") as js:
        config = json.load(js)

    lr = float(config["learning_rate"])
    batch_size = int(config["batch_size"])
    max_epochs = int(config["max_epochs"])
    db_path = Path(config["db_path"])
    train_split = config["train_split"]
    test_split = config["test_split"]
    num_classes = int(config["num_classes"])
    ckpt_path = Path(config["checkpoint_path"])
    prediction_threshold = float(config["prediction_threshold"])
    ckpt_period = int(config["checkpoint_period"])

    ds_train = OpenMICDataset(db_path, train_split, lim=batch_size*50, in_mem=True)
    ds_test = OpenMICDataset(db_path, test_split, lim=batch_size*16, in_mem=False)

    dl_train = DataLoader(
        ds_train, 
        batch_size, 
        shuffle=True
    )

    dl_test = DataLoader(
        ds_test,
        batch_size,
        shuffle=False
    )

    model = Cnn14(num_classes)
    model_dict = model.state_dict()

    pretrained_dict = load("./Cnn14_mAP=0.431.pth", map_location=device)['model']
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)

    model = model.to(device)

    criterion = BCELoss()
    
    optimizer = Adam(model.parameters(), lr)

    train(dl_train, dl_test, max_epochs, criterion, optimizer, model, device, num_classes, ckpt_path, ckpt_period, prediction_threshold, MIN_TEST_LOSS, MAX_TEST_ACC)

