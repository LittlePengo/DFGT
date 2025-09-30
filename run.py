import torch
import time
import dgl
from datasets.makeAnomaly import inject_anomalies_random
from utils import  normalize_adj,load_anomaly_detection_dgl
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
import argparse
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, average_precision_score
from model import GADGT_model

def get_best_f1(labels, probs, positive_index=1):
    """
    labels: 1D array-like, shape [N], {0,1}
    probs:  1D array-like [N]  (正类概率)
            或 2D array-like [N, C]（全类别概率矩阵）
    positive_index: 当 probs 为 2D 时，正类列的索引（默认 1）
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)

    # 取正类概率
    if probs.ndim == 2:
        pos = probs[:, positive_index]
    elif probs.ndim == 1:
        pos = probs
    else:
        raise ValueError(f"Unsupported probs.ndim={probs.ndim}")

    best_f1, best_thre = 0.0, 0.5
    # 从 0.05 到 0.95 搜阈值
    for thres in np.linspace(0.05, 0.95, 19):
        preds = (pos > thres).astype(int)
        mf1 = f1_score(labels, preds, average='macro', zero_division=0)
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def training(args):
    output_file = f"{args.dataset}_metrics2222.txt"
    with open(output_file, 'w') as f:
        f.write(f"Training on dataset: {args.dataset}\n\n")
        f.write("Parser Arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")
    if args.dataset=='Cora' or args.dataset=='Citeseer' or args.dataset=='PubMed':
        graph = inject_anomalies_random(args.dataset, args.num_structure_anomaly_nodes,args.num_attribute_anomaly_nodes, args.num_overlap)
    else:
        graph = load_anomaly_detection_dgl(args.dataset)
    attrs = graph.ndata['feat']
    adj_label = graph.adjacency_matrix()
    dense_adj_matrix = adj_label.to_dense()
    adj = normalize_adj(dense_adj_matrix + torch.eye(dense_adj_matrix.shape[0]))
    adj = torch.FloatTensor(adj)
    labels = graph.ndata['anomaly']
    device = torch.device(args.device)
    graph = dgl.add_self_loop(graph)
    graph = graph.to(device)
    attrs = torch.as_tensor(attrs, dtype=torch.float32, device=device)
    adj = torch.as_tensor(adj, dtype=torch.float32, device=device)
    labels = labels.to(device)
    labels_np = labels.detach().cpu().numpy()
    index = list(range(len(labels)))
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels_np[index], stratify=labels_np[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)], device=device).bool()
    val_mask = torch.zeros([len(labels)], device=device).bool()
    test_mask = torch.zeros([len(labels)], device=device).bool()
    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    msg = f'train/dev/test samples: {train_mask.sum().item()} {val_mask.sum().item()} {test_mask.sum().item()}'
    print(msg)
    with open(output_file, 'a') as f:
        f.write(msg + '\n')
    print("feature size:", attrs.size(1))
    model = GADGT_model(args, feat_size= attrs.size(1)).to(device) 
    optimiser = torch.optim.Adam(model.parameters(), lr = args.lr)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    msg = f'cross entropy weight: {weight}'
    print(msg)
    with open(output_file, 'a') as f:
        f.write(msg + '\n')
    time_start = time.time()
    use_cuda_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)
    for epoch in range(args.epoch):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        if use_cuda_amp:
            with torch.cuda.amp.autocast():
                logits, C_loss = model(attrs, adj)
                class_weight = torch.tensor([1.0, weight], device=logits.device, dtype=logits.dtype)
                loss_ce = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weight)
                print("C_loss",C_loss)
                print("loss_ce",loss_ce)
                loss = loss_ce +  C_loss * args.alpha
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
        else:
            logits, C_loss = model(attrs, adj)
            class_weight = torch.tensor([1.0, weight], device=logits.device, dtype=logits.dtype)
            loss_ce = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weight)
            print("C_loss",C_loss)
            print("loss_ce",loss_ce)
            loss = loss_ce +  C_loss * args.alpha
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(attrs, adj)
            probs_eval = torch.softmax(logits_eval, dim=1)
            val_mask_np = val_mask.detach().cpu().numpy().astype(bool)
            test_mask_np = test_mask.detach().cpu().numpy().astype(bool)
            labels_np = labels.detach().cpu().numpy()

            val_labels_np = labels_np[val_mask_np]
            test_labels_np = labels_np[test_mask_np]

            val_probs_np = probs_eval[:, 1].detach().cpu().numpy()[val_mask_np]
            all_probs_np = probs_eval[:, 1].detach().cpu().numpy()
            all_scores_np = logits_eval[:, 1].detach().cpu().numpy()  # 用于AUC（更稳定）

            # 验证集找最优阈值（你的函数）
            f1, thres = get_best_f1(val_labels_np, val_probs_np)

            # 基于阈值得到测试集预测
            preds_np = (all_probs_np > thres).astype(int)

            # 各项指标（测试集）
            trec = recall_score(test_labels_np, preds_np[test_mask_np])
            tpre = precision_score(test_labels_np, preds_np[test_mask_np],zero_division=0)
            tmf1 = f1_score(test_labels_np, preds_np[test_mask_np], average='macro')
            tauc = roc_auc_score(test_labels_np, all_scores_np[test_mask_np])
            tauprc = average_precision_score(test_labels_np, all_probs_np[test_mask_np])
            if best_f1 < f1:
                best_f1 = f1
                final_trec = trec
                final_tpre = tpre
                final_tmf1 = tmf1
                final_tauc = tauc
                final_tauprc = tauprc
            msg = f'Epoch {epoch}, loss: {loss:.4f}, val mf1: {f1:.4f}, auc: {tauc:.4f}, auprc: {tauprc:.4f},(best {best_f1:.4f})'
            print(msg)
            with open(output_file, 'a') as f:
                f.write(msg + '\n')
    time_end = time.time()
    msg = f'time cost: {time_end - time_start} s'
    print(msg)
    with open(output_file, 'a') as f:
        f.write(msg + '\n')
    
    msg = f'Test: REC {final_trec*100:.2f} PRE {final_tpre*100:.2f} MF1 {final_tmf1*100:.2f} AUC {final_tauc*100:.2f} AUPRC {final_tauprc*100:.2f}'
    print(msg)
    with open(output_file, 'a') as f:
        f.write(msg + '\n')
    return final_tmf1, final_tauc, final_tauprc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='acmv9_both', help='dataset name: Cora/Citeseer') #Amazon-all/YelpChi-all/acmv9_both/citationv1_both.mat
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--num_structure_anomaly_nodes', type=float, default=0.1, help='1/2')
    parser.add_argument('--num_attribute_anomaly_nodes', type=float, default=0.1, help='1/2')
    parser.add_argument('--num_overlap', type=float, default=0.0, help='1/2')
    parser.add_argument('--epoch', type=int, default=500, help='Training epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', default=0,  help='weight_decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=1,help='trade-off hybrid representation in loss function') #控制结构误差和属性误差占比的超参数
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument('--dropout_att', type=float, default=0.5, help='Att Dropout rate')
    parser.add_argument('--nb_classes', type=int, default=2, help='Binary classification')
    parser.add_argument('--Trans_layer_num', type=int, default=2, help='the number of layers.')
    args = parser.parse_args()
    mf1, auc ,auprc= training(args)
    final_mf1s, final_aucs,final_auprcs = [], [] , []
    final_mf1s.append(mf1)
    final_aucs.append(auc)
    final_auprcs.append(auprc)
    final_mf1s = np.array(final_mf1s)
    final_aucs = np.array(final_aucs)
    final_auprcs = np.array(final_auprcs)
    msg = f'MF1-mean: {100 * np.mean(final_mf1s):.2f}, MF1-std: {100 * np.std(final_mf1s):.2f}, ' \
          f'AUC-mean: {100 * np.mean(final_aucs):.2f}, AUC-std: {100 * np.std(final_aucs):.2f}, ' \
          f'AUPRC-mean: {100 * np.mean(final_auprcs):.2f}, AUPRC-std: {100 * np.std(final_auprcs):.2f}'
    print(msg)
    with open(f"{args.dataset}_metrics2222.txt", 'a') as f:
        f.write(msg + '\n')