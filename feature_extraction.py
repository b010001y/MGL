import torch
import torchvision.models as models
import torch.nn as nn
import jieba  # 用于中文分词
from gensim.models import KeyedVectors
import numpy as np



def get_visual_embeddings(image_tensor, pretrained_model='resnet'):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    """Extracts feature embeddings from images using a pre-trained CNN."""
    if pretrained_model == 'resnet':
        model = models.resnet50(pretrained=True)
    # You can add more pre-trained models here.
    model = nn.Sequential(*list(model.children())[:-1])  # Remove the last classification layer
    with torch.no_grad():
        visual_embeddings = model(image_tensor)
    return visual_embeddings.squeeze()  # Assuming that we are not using batches here



def load_word2vec_model(word2vec_file_path):
    """Loads the Word2Vec model from a file."""
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_file_path, binary=False)
    return word2vec_model

def get_text_embeddings(labels, word2vec_model):
    """Converts Chinese text labels into embeddings using Word2Vec model."""
    embeddings = []

    for label in labels:
        words = jieba.lcut(label)  # 中文分词
        word_embeddings = [word2vec_model[word] for word in words if word in word2vec_model]
        
        if word_embeddings:
            avg_embedding = np.mean(word_embeddings, axis=0)
            embeddings.append(avg_embedding)
        else:
            embeddings.append(np.zeros(word2vec_model.vector_size))  # Fallback for no words found

    return torch.tensor(embeddings).float()

# These functions will be used in the main training loop to get the embeddings
