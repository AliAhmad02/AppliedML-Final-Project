import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import io
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
from torchvision.transforms.functional import convert_image_dtype, pil_to_tensor
from torch import optim
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import torchvision.transforms.functional as transform

from concurrent.futures import ThreadPoolExecutor

# Paths to data
path_kp = "data_files/kp_data.txt"
path_img_specs = "data_files/image_specs.csv"
img_dir = "data_files/solar_images"

# Load kp data
data_kp = pd.read_csv(path_kp)
data_kp["datetime"] = pd.to_datetime(data_kp["datetime"])

# Load image specs data
img_specs = pd.read_csv(path_img_specs)
img_specs["datetime"] = pd.to_datetime(img_specs["datetime"])

image_filenames = img_specs["filename"].values
image_dates = img_specs["datetime"].values


# Converting datetimes to integer timestamps in seconds
image_timestamps = image_dates.astype("int64") // 10**9
kp_timestamps = data_kp["datetime"].values.astype("int64") // 10**9
kp_index_interpolated = CubicSpline(kp_timestamps, data_kp["Kp"].values)(
    image_timestamps
)

data_merged = pd.DataFrame(
    {
        "Timestamp": image_timestamps,
        "Image_filename": image_filenames,
        "Kp": kp_index_interpolated,
    }
)

day = 24 * 60 * 60
year = 365.2425 * day
# Synodic carrington rotation of sun
cycle = 27.2753 * day

data_merged["day_sin"] = np.sin(image_timestamps * (2 * np.pi / day))
data_merged["day_cos"] = np.cos(image_timestamps * (2 * np.pi / day))
data_merged["cycle_sin"] = np.sin(image_timestamps * (2 * np.pi / cycle))
data_merged["cycle_cos"] = np.cos(image_timestamps * (2 * np.pi / cycle))
data_merged["year_sin"] = np.sin(image_timestamps * (2 * np.pi / year))
data_merged["year_cos"] = np.cos(image_timestamps * (2 * np.pi / year))

# Scaling the numerical data
kp_scaler = MinMaxScaler(feature_range=(-1, 1))
timestamp_scaler = MinMaxScaler(feature_range=(-1, 1))

data_merged["Kp"] = kp_scaler.fit_transform(data_merged["Kp"].values.reshape(-1, 1))
data_merged["Timestamp"] = timestamp_scaler.fit_transform(
    data_merged["Timestamp"].values.reshape(-1, 1)
)


def create_sequence(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        feature = data.iloc[i : i + seq_length]
        target = data.iloc[i + seq_length]["Kp"]
        sequences.append((feature, target))
    return sequences

class ImageAndKpDataset(Dataset):
    def __init__(self, sequences, img_dir, img_transform):
        self.sequences = sequences
        self.img_transform = img_transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.sequences)
    
    def read_and_transform_image(self, path):
        return self.img_transform(read_image(path).cuda().float())

    def __getitem__(self, idx):
        features, target = self.sequences[idx]
        image_paths = [
            os.path.join(self.img_dir, path)
            for path in features["Image_filename"].values
        ]
        features = features.drop(columns=["Image_filename"])
        numerical_features = torch.tensor(features.values, dtype=torch.float32).cuda()
        target = torch.tensor(target, dtype=torch.float32).cuda().unsqueeze(-1)
        with ThreadPoolExecutor() as executor:
            images = list(executor.map(self.read_and_transform_image, image_paths))
        images = torch.stack(images)
        return images, numerical_features, target

seq_length = 7
sequences = create_sequence(data_merged, seq_length)
img_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)
dataset = ImageAndKpDataset(sequences, img_dir, img_transform)

# Split training data into training and validation data:
full_len = len(dataset)
train_frac = 0.8
train_size = int(full_len * train_frac)
train_data = Subset(dataset, range(0, train_size))
val_data = Subset(dataset, range(train_size, full_len))
batch_size = 8

# Create PyTorch dataloaders for data:

train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(val_data, batch_size=batch_size)


class SolarImageKpModel(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN for the solar images
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 64 * 32, 4)

        # RNN for the image data
        self.lstm_img = nn.LSTM(
            input_size=4, hidden_size=32, num_layers=1, batch_first=True
        )

        # Fully-connected layers for the numerical data (8 input features)
        self.fc_num1 = nn.Linear(8, 16)
        self.fc_num2 = nn.Linear(16, 32)

        # RNN for the numerical data
        self.lstm_num = nn.LSTM(
            input_size=32, hidden_size=32, num_layers=1, batch_first=True
        )

        # Fully-connected layer that combines the image and numerical data to make the final prediction
        self.fc_final = nn.Linear(64, 1)

    def forward(self, x_img, x_num):
        batch_size, seq_length, _, _, _ = x_img.size()

        # Perform CNN
        cnn_features = []
        for i in range(seq_length):
            img = x_img[:, i, :, :, :]
            img = self.pool(F.relu(self.conv1(img)))
            img = self.pool(F.relu(self.conv2(img)))
            img = self.pool(F.relu(self.conv3(img)))
            img = img.view(batch_size, -1)
            img = F.relu(self.fc1(img))
            cnn_features.append(img)

        # Putting the CNN features together so we can pass them to the RNN
        cnn_features = torch.stack(cnn_features, dim=1)

        rnn_out_img, _ = self.lstm_img(cnn_features)

        # We take the last time step from the RNN
        rnn_out_img = rnn_out_img[:, -1, :]

        # The two fully-connected layers for the numerical features
        x_num = F.relu(self.fc_num1(x_num))
        x_num = F.relu(self.fc_num2(x_num))

        # RNN for the numerical features
        rnn_out_num, _ = self.lstm_num(x_num)
        rnn_out_num = rnn_out_num[:, -1, :]

        # Combining the CNN and numerical features RNN output
        combined_rnn = torch.cat((rnn_out_img, rnn_out_num), dim=1)

        # Final fully-connected layer
        out = self.fc_final(combined_rnn)

        return out

n_epochs = 50
loss_fn = nn.MSELoss(reduction="mean")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SolarImageKpModel()
optimizer = optim.Adam(model.parameters(), lr=5*10**(-4))
model.to(device)


def train_valid(n_epochs, model, optimizer, loss_fn, train_loader, test_loader, device):
    train_hist = []
    test_hist = []

    # Training loop
    for epoch in tqdm(range(n_epochs)):
        total_loss = 0.0

        # Training
        model.train()
        for batch_img, batch_num, batch_target in train_loader:
            predictions = model(batch_img, batch_num)
            loss = loss_fn(predictions, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
         #Calculate average training loss and accuracy
        average_loss = total_loss / len(train_loader)
        train_hist.append(average_loss)

        # Validation on test data
        model.eval()
        with torch.no_grad():
            total_test_loss = 0.0

            for batch_img_test, batch_num_test, batch_target_test in test_loader:
                predictions_test = model(batch_img_test, batch_num_test)
                test_loss = loss_fn(predictions_test, batch_target_test)

                total_test_loss += test_loss.item()

            # Calculate average test loss and accuracy
            average_test_loss = total_test_loss / len(test_loader)
            test_hist.append(average_test_loss)
        print(
            f"Epoch [{epoch+1}/{n_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}"
        )
    torch.save(model, "data_files/image_model.pth")
    return train_hist, test_hist


train_hist, test_hist = train_valid(
    n_epochs, model, optimizer, loss_fn, train_loader, test_loader, device
)

pd.Series(train_hist).to_csv("data_files/train_history.csv")
pd.Series(test_hist).to_csv("data_files/test_history.csv")
