import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
from torchvision.transforms.functional import convert_image_dtype
from torch import optim
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

# Paths to data
path_kp = "AppliedML-Final-Project/data_files/kp_data.txt"
path_img_specs = "AppliedML-Final-Project/data_files/image_specs.csv"
img_dir = "AppliedML-Final-Project/data_files/solar_images"

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
    def __init__(self, sequences, img_dir, img_transform=None):
        self.sequences = sequences
        self.img_transform = img_transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        features, target = self.sequences[idx]
        images = []
        numerical_features = []
        for _, row in features.iterrows():
            img_filename = row.pop("Image_filename")
            img_path = os.path.join(self.img_dir, img_filename)
            image = read_image(img_path)
            image = convert_image_dtype(image, torch.float32)
            if self.img_transform:
                image = self.img_transform(image)
            images.append(image)
            numerical_features.append(row.values)
        images = torch.stack(images)
        numerical_features = torch.tensor(
            np.array(numerical_features).astype(float), dtype=torch.float32
        )
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
        return images, numerical_features, target


seq_length = 7
sequences = create_sequence(data_merged, seq_length)
# img_transform = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
img_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
dataset = ImageAndKpDataset(sequences, img_dir, img_transform)

# Split training data into training and validation data:
full_len = len(dataset)
# train_frac = 0.9
train_frac = 0.005
train_size = int(full_len * train_frac)
train_data = Subset(dataset, range(0, train_size))
val_data = Subset(dataset, range(train_size, full_len))
# batch_size = 1
batch_size = 4

# Create PyTorch dataloaders for data:
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(val_data, batch_size=batch_size)


class SolarImageKpModel(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN for the solar images
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.fc1 = nn.Linear(128 * 128 * 128, 4)
        self.fc1 = nn.Linear(64 * 64 * 128, 4)

        # RNN for the image data
        self.lstm_img = nn.LSTM(
            input_size=4, hidden_size=128, num_layers=1, batch_first=True
        )

        # Fully-connected layers for the numerical data (8 input features)
        self.fc_num1 = nn.Linear(8, 16)
        self.fc_num2 = nn.Linear(16, 128)

        # RNN for the numerical data
        self.lstm_num = nn.LSTM(
            input_size=128, hidden_size=128, num_layers=1, batch_first=True
        )

        # Fully-connected layer that combines the image and numerical data to make the final prediction
        self.fc_final = nn.Linear(256, 1)

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

n_epochs = 10
loss_fn = nn.MSELoss(reduction="mean")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SolarImageKpModel()
optimizer = optim.Adam(model.parameters())
model.to(device)


def train_valid(n_epochs, model, optimizer, loss_fn, train_loader, test_loader, device):
    train_hist = []
    test_hist = []

    # Training loop
    for epoch in range(n_epochs):
        total_loss = 0.0

        # Training
        model.train()
        for batch_img, batch_num, batch_target in train_loader:
            batch_img, batch_num, batch_target = (
                batch_img.to(device),
                batch_num.to(device),
                batch_target.to(device),
            )
            predictions = model(batch_img, batch_num)
            loss = loss_fn(predictions, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print("Finished batch")
        # Calculate average training loss and accuracy
        average_loss = total_loss / len(train_loader)
        train_hist.append(average_loss)

        # Validation on test data
        model.eval()
        with torch.no_grad():
            total_test_loss = 0.0

            for batch_img_test, batch_num_test, batch_target_test in test_loader:
                batch_img_test, batch_num_test, batch_target_test = (
                    batch_img_test.to(device),
                    batch_num_test.to(device),
                    batch_target_test.to(device),
                )
                predictions_test = model(batch_img_test, batch_num_test)
                test_loss = loss_fn(predictions_test, predictions_test)

                total_test_loss += test_loss.item()

            # Calculate average test loss and accuracy
            average_test_loss = total_test_loss / len(test_loader)
            test_hist.append(average_test_loss)
        print(
            f"Epoch [{epoch+1}/{n_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}"
        )
    return train_hist, test_hist


train_hist, test_hist = train_valid(
    n_epochs, model, optimizer, loss_fn, train_loader, test_loader, device
)
