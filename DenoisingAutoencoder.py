import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda, Compose
import numpy as np
import pandas as pd


# Returns CUDA device if available
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# Defines a Dataset object with given single-cell DataFrame
class SingleCellDataset(Dataset):
  def __init__(self, df, transform=None, shuffle=True):
    self.df = df
    if shuffle:
      self.df = self.df.sample(frac=1)
    self.transform = transform

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    sample = np.array(self.df.iloc[idx])
    if self.transform:
      sample = self.transform(sample)
    return sample

  def get_index(self):
    return self.df.index.values


# Sets elements of data to 0 with given probability p
def dropout(data, p, device):
    return (torch.rand(data.shape) < (1-p)).to(device).int() * data


# Defines a Module object for the denoising autoencoder archetecture
class DenoisingAE(nn.Module):
    def __init__(self):
        super(DenoisingAE, self).__init__()
        self.dropout = dropout
        self.encode = nn.Sequential(
            nn.Linear(5000, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 3),
            nn.BatchNorm1d(3, eps=1e-10, momentum=0.1)
        )
        self.decode = nn.Sequential(
            nn.Linear(3, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 5000),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.dropout(x, 0.5, device())
        x = self.encode(x)
        x = self.decode(x)
        return x


# Trains a given model and dataloader with specified loss function and optimizer
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, X in enumerate(dataloader):
        X = X.to(device())
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, X)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Calculates test error of given model and dataloader with specified loss function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device())
            pred = model(X)
            test_loss += loss_fn(pred, X).item()
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")


# Trains the autoencoder model with given AnnData object and returns a full 3D embedding of the data.
# 'train_split' specifies the proportion of data used for training with the remaining proportion used for
# calculating test error
# 'train_batch_size' specifies the number of training samples used per epoch
# 'test_batch_size' specifies the number of testing samples used per epoch
# 'embed_batch_size' specifies the chunk size when applying the embedding.
# A higher value will use more memory but less time, a smaller value will use less memory but take longer.
def embed(adata, train_split=0.85, train_batch_size=1000, test_batch_size=1000, epochs=1000,
          embed_batch_size=1000):
    df = adata.to_df()
    data = SingleCellDataset(
        df=df,
        shuffle=True,
        transform=ToTensor()
    )
    split_indx = int(len(data) * train_split)
    training = data[:split_indx][0].float()
    testing = data[split_indx:][0].float()

    train_dataloader = DataLoader(training, batch_size=train_batch_size)
    test_dataloader = DataLoader(testing, batch_size=test_batch_size)

    model = DenoisingAE().to(device())

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for t in range(epochs):
        print(f"Epoch {t + 1}")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    all_data = data[:][0]
    N = len(all_data)
    encodings = np.zeros((N, 3))
    with torch.no_grad():
        model = model.to('cpu')
        for i in range(int(N / embed_batch_size)):
            encodings[i * embed_batch_size:(i + 1) * embed_batch_size] = model.encode(all_data[i * embed_batch_size:(i + 1) * embed_batch_size]).numpy()
        encodings[-N % embed_batch_size:] = model.encode(all_data[-N % embed_batch_size:]).numpy()

    out_df = pd.DataFrame(encodings)
    out_df.index = data.get_index()

    return out_df
