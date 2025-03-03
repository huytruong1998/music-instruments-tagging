from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from dataset import OpenMICDataset
import json
from pathlib import Path
from model import Cnn14
from torch import cuda, no_grad, Tensor, sum, float32, uint8, unsqueeze, save
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

            x = x.unsqueeze(1).transpose(2, 3)

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

        for x, y in dl_test:
            x: Tensor = x.to(device)

            x = x.unsqueeze(1).transpose(2, 3)

            y: Tensor = y.to(device)

            y_pred: Tensor = model(x)['clipwise_output'] 

            loss = criterion(y_pred, y)

            test_loss.append(loss.item())

            y_pred_thr = (y_pred > prediction_threshold).to(float32)

            acc = (y_pred_thr == y).to(uint8)

            acc = sum(acc, dim=1) / num_classes

            test_acc = np.concatenate([test_acc, acc.cpu().numpy()])

        test_loss = np.mean(test_loss)
        test_acc = test_acc.mean() * 100.0

        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_acc:.2f}%")

        if test_acc > MAX_TEST_ACC or (test_acc == MAX_TEST_ACC and test_loss < MIN_TEST_LOSS):
            MAX_TEST_ACC = test_acc
            MIN_TEST_LOSS = test_loss

            ckpt_path.mkdir(666, exist_ok=True)

            save(
                model.state_dict(),
                ckpt_path / f"ckpt-{epoch}-{test_acc:.2f}-{datetime.now().replace(microsecond=0).isoformat().replace(':', '.')}.pt"
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

    ds_train = OpenMICDataset(db_path, train_split, lim=500, in_mem=True)
    ds_test = OpenMICDataset(db_path, test_split, lim=50, in_mem=True)

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
    model = model.to(device)

    criterion = BCELoss()
    
    optimizer = Adam(model.parameters(), lr)

    train(dl_train, dl_test, max_epochs, criterion, optimizer, model, device, num_classes, ckpt_path, ckpt_period, prediction_threshold, MIN_TEST_LOSS, MAX_TEST_ACC)
