# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_max_pool
# from torch.optim import Adam

# class ChessGNN(nn.Module):
#     def __init__(self, num_node_features, hidden_channels, num_classes):
#         super(ChessGNN, self).__init__()
#         self.conv1 = GCNConv(num_node_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.fc = nn.Linear(hidden_channels, num_classes)

#     def forward(self, x, edge_index, edge_attr):
#         x = self.conv1(x, edge_index, edge_attr)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index, edge_attr)
#         x = F.relu(x)
#         x = global_max_pool(x, torch.zeros(x.size(0), dtype=torch.long), x.size(0))
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)

# # Example usage
# # Model creation
# num_node_features = 10  # Number of node features (piece and team)
# hidden_channels = 32  # Number of hidden channels in GCN layers
# num_classes = 3  # Number of classes for graph classification
# model = ChessGNN(num_node_features, hidden_channels, num_classes)

# # Example input data
# x = torch.randn(4, num_node_features)  # Node features
# edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)  # Edge indices
# edge_attr = torch.randn(4, 1)  # Edge attributes
# batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)  # Batch indices for graph classification
# y = torch.tensor([0, 1, 2, 0])  # Labels for graph classification

# # Optimizer definition
# optimizer = Adam(model.parameters(), lr=0.01)

# # Loss function
# criterion = nn.NLLLoss()

# # Training the model
# model.train()

# for epoch in range(10):
#     optimizer.zero_grad()
#     output = model(x, edge_index, edge_attr)
#     loss = criterion(output[batch], y)
#     loss.backward()
#     optimizer.step()

#     print(f"Epoch {epoch+1} - Loss: {loss.item()}")

# # Evaluating the model
# model.eval()
# with torch.no_grad():
#     output = model(x, edge_index, edge_attr)
#     predicted_labels = output.argmax(dim=1)
#     accuracy = (predicted_labels == y).sum().item() / len(y)
#     print(f"Accuracy: {accuracy}")


# # Sauvegarde du model
# save_path = 'trained_model.pt'

# state = {
#     'model': model.state_dict(),
#     'optimizer': optimizer.state_dict(),
#     'epoch': num_epochs,
#     'train_acc': train_acc,
#     'test_acc': test_acc
# }

# torch.save(state, save_path)



# Loading part
# load_path = 'trained_model.pt'

# state = torch.load(load_path)

# model = GNNClassifier(num_features=dataset.num_features, hidden_dim=64, num_classes=dataset.num_classes)
# model.load_state_dict(state['model'])

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer.load_state_dict(state['optimizer'])
# num_epochs = state['epoch']
# train_acc = state['train_acc']
# test_acc = state['test_acc']