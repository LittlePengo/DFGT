import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset,AmazonCoBuyPhotoDataset,FlickrDataset
import torch
import torch.nn.functional as F
import random
import os

# 设置 DGL 下载路径
os.environ['DGL_DOWNLOAD_DIR'] = 'C:\DGLdatasets'
#随机注入异常
def inject_anomalies_random(dataset_name, num_structure_anomaly_nodes_percentage,
                     num_attribute_anomaly_nodes_percentage, num_overlap_percentage):
    if dataset_name == 'Cora' or dataset_name == 'cora-multi':
        dataset = CoraGraphDataset()
    elif dataset_name == 'Citeseer' or dataset_name == 'citeseer-multi':
        dataset = CiteseerGraphDataset()
    else:
        dataset = PubmedGraphDataset()

    graph = dataset[0]
    # 获取节点标签
    labels = graph.ndata['label']
    indices = [i for i in range(labels.shape[0])]



    # 注入多少个异常
    node_num = graph.number_of_nodes()
    num_structure_anomaly_nodes_all = (int)(num_structure_anomaly_nodes_percentage * node_num)
    num_attribute_anomaly_nodes_all = (int)(num_attribute_anomaly_nodes_percentage * node_num)
    anomlay_number = num_structure_anomaly_nodes_all + num_attribute_anomaly_nodes_all
    print("Number of anomalies", anomlay_number)
    num_overlap_all =(int)( num_overlap_percentage * node_num)
    anomaly_nodes_structure = random.sample(indices, num_structure_anomaly_nodes_all)

    # 建立全连接关系
    anomaly_edges = [(i, j) for i in anomaly_nodes_structure for j in anomaly_nodes_structure if i != j]
    src, dst = zip(*anomaly_edges)
    graph.add_edges(src, dst)


    left_indices =[]
    for i in indices:
            left_indices.append(i)

    common = (int) (num_overlap_all/2)
    num_remaining_nodes = num_attribute_anomaly_nodes_all - common

    #采样num_overlap_all一半 从结构异常
    anomaly_nodes_attr = random.sample(anomaly_nodes_structure, common)

    for i in random.sample(left_indices, num_remaining_nodes):
        anomaly_nodes_attr.append(i)



    # 对每个节点注入属性异常
    for node in anomaly_nodes_attr:
        # 随机选择 50 个节点
        random_nodes = random.sample(indices, 50)

        # 计算与某点的欧几里得距离
        distances = F.pairwise_distance(graph.ndata['feat'][node], graph.ndata['feat'][random_nodes], p=2)

        # 找到距离最大的节点
        max_distance_node = random_nodes[torch.argmax(distances)]

        # 将最大节点的属性赋值给当前节点
        graph.ndata['feat'][node] = graph.ndata['feat'][max_distance_node]


    graph.ndata['anomaly'] = torch.zeros(graph.number_of_nodes(), dtype=torch.long)
    graph.ndata['anomaly'][anomaly_nodes_structure] = 1
    graph.ndata['anomaly'][anomaly_nodes_attr] = 1

    return graph

def inject_anomalies_multi(dataset_name, num_structure_anomaly_nodes_percentage,
                           num_attribute_anomaly_nodes_percentage, num_overlap_percentage):
    graph = inject_anomalies_random(dataset_name=dataset_name, num_structure_anomaly_nodes_percentage=num_structure_anomaly_nodes_percentage,
                     num_attribute_anomaly_nodes_percentage=num_attribute_anomaly_nodes_percentage, num_overlap_percentage=num_overlap_percentage)
    labels = graph.ndata['label']

    # 获取类别数
    num_classes = len(torch.unique(labels))

    # 初始化一个列表，用于存储每个类别的子图
    class_subgraphs = []
    #dims = []

    # 构建每个类别的子图
    for i in range(num_classes):
        class_nodes = torch.nonzero(labels == i).squeeze()
        # 获取子图中节点的特征和标签
        subgraph_feats = graph.ndata['feat'][class_nodes]
        # subgraph_anomaly = graph.ndata['anomaly'][class_nodes]

        # 构建子图时传递节点的特征和标签
        class_subgraph = dgl.node_subgraph(graph, class_nodes)

        # 将子图中的特征和标签更新为新的值
        class_subgraph.ndata['feat'] = subgraph_feats

        class_subgraphs.append(class_subgraph)
        #dims.append(subgraph_feats.shape[-1])

    # 打印每个类别的子图信息
    for i, subgraph in enumerate(class_subgraphs):
        print(f"Class {i} subgraph:")
        print(subgraph)
        print(f"Number of nodes: {subgraph.number_of_nodes()}")
        print(f"Number of edges: {subgraph.number_of_edges()}")
        print("=" * 50)
    return graph, class_subgraphs