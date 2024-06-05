import torch
import torch.nn.functional as F
from torch import nn


class SolarImageKpModel(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN for the solar images
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
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
            img = self.pool(F.relu(self.bn1(self.conv1(img))))
            img = self.pool(F.relu(self.bn2(self.conv2(img))))
            img = self.pool(F.relu(self.bn3(self.conv3(img))))
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
