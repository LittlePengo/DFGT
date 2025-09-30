import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl
from sklearn.metrics import precision_score,recall_score
import h5py
from scipy.linalg import fractional_matrix_power
import os
def _load_mat_any(mat_path, variable_names=None):
    """
    兼容 v5/v7.3 的 .mat 读取：
    - 先尝试 scipy.io.loadmat（v5）
    - 若失败再用 h5py 读取（v7.3/HDF5）
    - 若文件损坏/太小，直接报错提示重新下载
    """
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    if os.path.getsize(mat_path) < 128:
        raise OSError(f"MAT file looks empty or corrupted: {mat_path}")

    # 第一种：普通 v5
    try:
        return sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except OSError:
        pass  # 尝试 v7.3

    # 第二种：v7.3 / HDF5
    try:
        import h5py
        with h5py.File(mat_path, 'r') as f:
            keys = list(f.keys()) if variable_names is None else [k for k in variable_names if k in f]
            out = {}
            for k in keys:
                d = f[k][()]
                # h5py 默认按列主序，很多公开数据集变量需要转置成与 v5 风格一致
                if hasattr(d, "ndim") and d.ndim >= 2:
                    out[k] = np.array(d).T
                else:
                    out[k] = np.array(d)
            return out
    except Exception as e:
        raise OSError(f"Failed to read MAT file as v5 or v7.3 (HDF5). "
                      f"The file may be corrupted or partially downloaded. Path={mat_path}. "
                      f"Underlying error: {e}")
def normalize_adjj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def load_anomaly_detection_dataset(dataset, datadir='datasets'):
    mat_path = os.path.join(datadir, f'{dataset}.mat')
    print(f"Loading dataset from: {mat_path}")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"File not found: {mat_path}")
    data_mat = _load_mat_any(mat_path)
    try:
        adj = data_mat['Network']
        feat = data_mat['Attributes']
        truth = data_mat['Label']
    except Exception:
        adj = data_mat['A']
        feat = data_mat['X']
        truth = data_mat['gnd']
    print(f"Raw adjacency shape: {adj.shape}")
    print(f"Number of non-zero entries in original adjacency matrix (raw edges): {adj.nnz}")
    print(f"Estimated number of undirected edges: {adj.nnz // 2}")  # 如果无向图
    truth = truth.flatten()
        # 查看 truth 中 1 的个数
    unique_labels, counts = np.unique(truth, return_counts=True)
    print(f"Unique labels: {unique_labels}")
    print(f"Counts per label: {dict(zip(unique_labels, counts))}")
    num_ones = np.sum(truth == 1)
    print(f'Number of anomalies (1s) in truth: {num_ones}')
    
    adj_norm = normalize_adjj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    # adj = adj + sp.eye(adj.shape[0])
    adj = adj.toarray()
    if sp.issparse(feat):
        feat = feat.toarray()
    elif not isinstance(feat, np.ndarray):
        feat = np.array(feat)
    num_nodes = adj.shape[0]
    print(f'Number of nodes: {num_nodes}')
    return adj_norm, feat, truth, adj

def load_anomaly_detection_dgl(dataset, datadir='datasets'):
    # 加载数据
    adj_norm, feat, truth, adj = load_anomaly_detection_dataset(dataset, datadir)
    
    # 转换为稀疏矩阵 (DGL 需要)
    adj_csr = sp.csr_matrix(adj)
    
    # 转换为 DGL 图
    graph = dgl.from_scipy(adj_csr)
    
    # 添加节点属性
    graph.ndata['feat'] = torch.tensor(feat, dtype=torch.float32)
    
    # 添加节点标签
    graph.ndata['anomaly'] = torch.tensor(truth, dtype=torch.int64)
    
    return graph
def recall_at_k(true_labels, predicted_scores, k):
    sorted_indices = sorted(range(len(predicted_scores)), key=lambda i: predicted_scores[i], reverse=True)
    top_k_predictions = [true_labels[i] for i in sorted_indices[:k]]

    total_relevant_items = sum(true_labels)
    relevant_items_in_top_k = sum(top_k_predictions)

    recall_at_k = relevant_items_in_top_k / total_relevant_items if total_relevant_items > 0 else 0
    return recall_at_k

def precision_at_k(true_labels, predicted_scores, k):
    sorted_indices = sorted(range(len(predicted_scores)), key=lambda i: predicted_scores[i], reverse=True)
    top_k_predictions = [true_labels[i] for i in sorted_indices[:k]]

    precision_at_k = sum(top_k_predictions) / k if k > 0 else 0
    return precision_at_k
def f1_at_k(true_labels, predicted_scores, k):
    precision = precision_at_k(true_labels, predicted_scores, k)
    recall = recall_at_k(true_labels, predicted_scores, k)

    f1_at_k = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_at_k


def normalize_adj(adj):
    """对称归一化邻接矩阵"""
    if isinstance(adj, torch.Tensor):
        adj = adj.detach().cpu().numpy()
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return torch.from_numpy(normalized_adj.todense()).float()

