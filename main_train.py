import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from data_loader import get_data_loaders
from feature_extraction import get_visual_embeddings, get_text_embeddings, load_word2vec_model
from graph_construction import build_superpixel_graph, combine_embeddings
from gnn_model import BuildingGNN
from torchvision import transforms


# 测试和评估模型的部分将在这里进行
# 测试和评估模型
def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            image_embeddings = get_visual_embeddings(images).to(device)
            text_embeddings = get_text_embeddings(labels, word2vec_model).to(device)

            combined_embeddings = combine_embeddings(image_embeddings, text_embeddings)
            adjacency_matrix = build_superpixel_graph(image_embeddings)

            graph_data = Data(x=combined_embeddings, edge_index=adjacency_matrix.nonzero().t().contiguous()).to(device)
            
            outputs = model(graph_data)
            predicted = torch.argmax(outputs.data, 1)
            # predicted = outputs.data

            map_label = map_labels_to_int(labels)
            map_label = torch.tensor(map_label).to(device)

            total += len(labels)
            correct += (predicted == map_label).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')


# 假设参数
NUM_CLASSES = 20  # 假设有10个类别
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 20
IMAGE_DIR = '/data4/lvxin/butianci/butianci/data/Pattern_Recoginition'
LABEL_DIR = '/data4/lvxin/butianci/butianci/data/Pattern_Recoginition'

# 图像和文本转换
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

MAP_LABELS_TO_INT = {
    '01教学楼': 0,
    '02教学楼': 1,
    '03教学楼': 2,
    '三院一号楼（主楼）': 3,
    '三院2号楼（老楼）': 4,
    '东跨线桥': 5,
    '俱乐部': 6,
    '体育馆': 7,
    '博士宿舍': 8,
    '图书馆': 9,
    '校主楼': 10,
    '游泳馆': 11,
    '海天楼': 12,
    '老图书馆': 13,
    '西跨线桥': 14,
    '银河大楼': 15,
    '航院主楼': 16,
    '天河大楼': 17,
    '四院主楼': 18,
    '北斗': 19
}

def map_labels_to_int(labels):
    """将文本标签映射为整数"""
    return [MAP_LABELS_TO_INT[label] for label in labels]

# 加载数据
train_loader, test_loader = get_data_loaders(IMAGE_DIR, LABEL_DIR, BATCH_SIZE, image_transform)


# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BuildingGNN(num_features=2348, num_classes=NUM_CLASSES).to(device)  # num_features根据实际情况填写
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


word2vec_model = load_word2vec_model('/data4/lvxin/butianci/butianci/model/sgns.baidubaike.bigram-char')

# 训练循环
for epoch in range(NUM_EPOCHS):
    for images, labels in train_loader:
        if images is None or labels is None:
            continue
        # 提取特征
        image_embeddings = get_visual_embeddings(images).to(device)
        text_embeddings = get_text_embeddings(labels, word2vec_model).to(device)
        
        # 构建图
        combined_embeddings = combine_embeddings(image_embeddings, text_embeddings)
        adjacency_matrix = build_superpixel_graph(image_embeddings)
        
        # 创建图数据
        graph_data = Data(x=combined_embeddings, edge_index=adjacency_matrix.nonzero().t().contiguous())
        graph_data = graph_data.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(graph_data)
        
        map_label = map_labels_to_int(labels)
        map_label = torch.tensor(map_label).to(device)
        # 计算损失
        loss = criterion(output, map_label)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}')
    
    # 在测试集上评估模型
    test_model(model, test_loader)