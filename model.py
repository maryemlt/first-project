import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Définition des couches cachées
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Couche de sortie
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Propagation avant à travers les couches cachées avec activation ReLU
        out = F.relu(self.fc1(x))
        out = self.dropout(out)  # Dropout pour la régularisation
        for _ in range(self.num_layers - 1):
            out = F.relu(self.fc2(out))
            out = self.dropout(out)
        
        # Couche de sortie avec activation log-softmax
        out = self.fc3(out)
        return F.log_softmax(out, dim=1)
