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
from tqdm import tqdm

path_img_specs = "AppliedML-Final-Project/data_files/image_specs.csv"
img_dir = "AppliedML-Final-Project/data_files/solar_images"

# Load image specs data
img_specs = pd.read_csv(path_img_specs)
img_labels = img_specs["filename"].values

class ImageDataset(Dataset):
    def __init__(self, img_labels, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = img_labels
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = read_image(img_path)
        image = convert_image_dtype(image, torch.float32)
        return image
    

transform = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
image_dataset = ImageDataset(img_labels, img_dir, transform)
full_len = len(image_dataset)

# Split data into training and validation data
train_frac = 0.9
train_size = int(full_len * train_frac)
train_data = Subset(image_dataset, range(0, train_size))
val_data = Subset(image_dataset, range(train_size, full_len))

# Create PyTorch dataloaders for data
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(val_data, batch_size=batch_size)

class Encoder(nn.Module):
    def __init__(self, image_size, embedding_dim):
        super(Encoder, self).__init__()
        # define convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # variable to store the shape of the output tensor before flattening
        # the features, it will be used in decoders input while reconstructing
        self.shape_before_flattening = None
        # compute the flattened size after convolutions
        flattened_size = (image_size // 8) * (image_size // 8) * 128
        # define fully connected layer to create embeddings
        self.fc = nn.Linear(flattened_size, embedding_dim)
    def forward(self, x):
        # apply ReLU activations after each convolutional layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # store the shape before flattening
        self.shape_before_flattening = x.shape[1:]
        # flatten the tensor
        x = x.view(x.size(0), -1)
        # apply fully connected layer to generate embeddings
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim, shape_before_flattening):
        super(Decoder, self).__init__()
        # define fully connected layer to unflatten the embeddings
        self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flattening))
        # store the shape before flattening
        self.reshape_dim = shape_before_flattening
        # define transpose convolutional layers
        self.deconv1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # define final convolutional layer to generate output image
        self.conv1 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # apply fully connected layer to unflatten the embeddings
        x = self.fc(x)
        # reshape the tensor to match shape before flattening
        x = x.view(x.size(0), *self.reshape_dim)
        # apply ReLU activations after each transpose convolutional layer
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        # apply sigmoid activation to the final convolutional layer to generate output image
        x = torch.sigmoid(self.conv1(x))
        return x

embedding_dim = 12
image_size = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

shape_before_flattening = [128, 128, 128]

encoder = Encoder(
    image_size,
    embedding_dim,
).to(device)

decoder = Decoder(embedding_dim, shape_before_flattening).to(device)

### Training function
def train_epoch(encoder, decoder, dataloader, loss_fn, optimizer, device):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
#         print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


### Testing function
def test_epoch(encoder, decoder, dataloader, loss_fn, device):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch in dataloader:
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

def train_test(n_epochs, encoder, decoder, optimizer, loss_fn, train_loader, test_loader, device):
    train_hist = []
    test_hist = []
    for epoch in tqdm(range(n_epochs)):
        train_loss = train_epoch(encoder, decoder, train_loader, loss_fn, optimizer, device)
        test_loss = test_epoch(encoder, decoder, test_loader, loss_fn, device)
        train_hist.append(train_loss)
        test_hist.append(test_loss)
        print(f"Epoch [{epoch+1}/{n_epochs}] - Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        return train_hist, test_hist
    
n_epochs = 10
loss_fn = nn.MSELoss()
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]
optimizer = torch.optim.Adam(params_to_optimize)

train_hist, test_hist = train_test(n_epochs, encoder, decoder, optimizer, loss_fn, train_loader, test_loader, device)