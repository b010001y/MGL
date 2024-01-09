import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class BuildingGNN(nn.Module):
    """Graph Neural Network model for building classification."""
    
    def __init__(self, num_features, num_classes):
        super(BuildingGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 256)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Apply a global mean pooling to get graph-level representation
        # x = global_mean_pool(x, batch)
        
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

# This model will be instantiated and trained in the main training script.
