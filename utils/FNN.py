import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import numpy as np

epochs = 1000
batch_size = 562
device ='cpu'
# labels are scaled before training 
scaler = StandardScaler()  


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('/utils/state_dictionaries/tate_dictionary_'))
loss_fn = nn.L1Loss()
MSE_loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),  weight_decay=0.001)

# Train function
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, sample_batch in enumerate(dataloader):
        X, y = sample_batch['features'].to(device), sample_batch['labels'].to(device)
        y = y.unsqueeze(1)
        # Compute prediction error
        X = X.float()
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()       

# Test function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)

    model.eval()
    L1_loss = 0
    MSE_loss = 0
    max_error = torch.Tensor(np.zeros(1))
    
    with torch.no_grad():
        for batch, sample_batch in enumerate(dataloader, 1):
            X, y = sample_batch['mordreds'].to(device), sample_batch['label'].to(device)
            X = X.float()
            y = y.unsqueeze(1)
            pred = model(X)
            
            #inverse scaling is applied before computing test scores
            yr = torch.Tensor(scaler.inverse_transform(y.numpy()))
            predr = torch.Tensor(scaler.inverse_transform(pred.numpy()))

            maximum_batch_error = torch.max(torch.abs(predr-yr))
            max_error = torch.max(max_error, maximum_batch_error)
            L1_loss +=  torch.sum(torch.abs(predr - yr))
            MSE_loss += torch.sum(torch.abs(predr - yr)**2)
            
    L1_loss = L1_loss/size
    MSE_loss = MSE_loss/size
    return L1_loss, np.sqrt(MSE_loss), max_error




