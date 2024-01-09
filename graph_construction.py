import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def build_superpixel_graph(image_embeddings, edge_threshold=0.5):
    """Builds a graph where nodes are superpixels and edges are based on similarity."""
    num_nodes = image_embeddings.shape[0]
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))

    # Calculate pairwise cosine similarity and create edges based on a threshold
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            similarity = cos(image_embeddings[i].unsqueeze(0), image_embeddings[j].unsqueeze(0))
            if similarity > edge_threshold:
                adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
    
    return adjacency_matrix

def get_label_embeddings(labels):
    """Converts text labels into a one-hot encoded sparse matrix."""
    # Here, we're assuming that labels are in plain text format.
    vectorizer = CountVectorizer()
    label_embeddings = vectorizer.fit_transform(labels)
    
    return label_embeddings

# def combine_embeddings(image_embeddings, label_embeddings):
#     """Combines image and label embeddings to form the feature vector for each node."""
#     # This function needs to handle the conversion to the appropriate tensor format.
#     # Here, we will concatenate normalized embeddings as an example.
#     old_label_embeddings = label_embeddings.clone()
#     label_embeddings = label_embeddings.cpu().numpy()
#     label_embeddings = OneHotEncoder().fit_transform(label_embeddings.reshape(-1, 1)).toarray()
#     label_embeddings = torch.tensor(label_embeddings).float().to('cuda')
    
#     # Normalize image embeddings
#     image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
    
#     combined_embeddings = torch.cat((image_embeddings, label_embeddings), dim=1)
    
#     return combined_embeddings


def combine_embeddings(image_embeddings, label_embeddings):
    """Combines image and label embeddings to form the feature vector for each node."""

    # 确保label_embeddings与image_embeddings在批处理维度上的大小相同
    if label_embeddings.shape[0] != image_embeddings.shape[0]:
        raise ValueError("The number of image embeddings and label embeddings must be the same.")

    # Normalize image embeddings
    image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
    
    # 如果label_embeddings不是浮点型，转换成浮点型
    if not label_embeddings.is_floating_point():
          label_embeddings = label_embeddings.float()

    # 拼接图像嵌入和标签嵌入
    combined_embeddings = torch.cat((image_embeddings, label_embeddings), dim=1)
    
    return combined_embeddings
# In the main training loop, we'll use these functions to construct the graph and its features
