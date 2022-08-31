from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

import numpy as np

import torch

def get_accuracy_from_logits(logits, labels):
    probs = logits.unsqueeze(-1)
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc


def get_f1_from_logits(logits, labels):
    preds = (logits > 0.5).astype(int)
    _, _, f, _ = precision_recall_fscore_support(labels, preds, pos_label=1, average="binary")
    return f


def get_roc_auc_from_logits(logits, labels):
    preds = (logits > 0.5).astype(int)
    return roc_auc_score(preds,labels)


def evaluate(net, criterion, dataloader, device):
    net.eval()
    mean_acc, mean_loss = 0, 0
    count = 0
    all_log = np.array([])
    all_labels = np.array([])
    with torch.no_grad():
        for seq, mask, seg, dev_stat, labels, idx in dataloader:
#             reg_loss = 0
#             for param in net.parameters():
#                 reg_loss += torch.sum(torch.abs(param))
            labels = labels.to(device)
            # stats = np.array(dev_stat)
            # stats = torch.tensor(stats[idx]).float()
            #Obtaining the logits from the model
            logits = net(seq.to(device), mask.to(device), seg.to(device), dev_stat.to(device)) 
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()  
#             + 0.0001 * reg_loss
            # mean_acc += get_accuracy_from_logits(logits, labels)

            all_log = np.hstack((all_log, logits.squeeze().cpu().numpy()))
            all_labels = np.hstack((all_labels, labels.cpu().numpy()))
            count += 1

        f = get_f1_from_logits(all_log, all_labels)
        roc_auc = get_roc_auc_from_logits(all_log, all_labels)
    return f, roc_auc, mean_loss / count
