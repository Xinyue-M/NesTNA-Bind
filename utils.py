import os, sys, shutil
import pickle
import random

import torch
import numpy as np 
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
from torch_scatter import scatter_max
from dataloader import IterPatchDataset, PatchDataset


AA_CODES = {
     'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
     'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
     'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
     'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
     'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W', 
     'UNK':'X', 'HID':'H'}

DATASET = {
    'RNA': ["RNA-663_Train", "RNA-157_Test"],
    'DNA': ["DNA-719_Train", "DNA-179_Test"],
    'RNA_our': ["RNA_our_Train", "RNA_our_Test"],
    'PPI': ["training_ppi", "testing_ppi"]
}

def printf(args, *content):
    file = sys.stdout
    f_handler = open(os.path.join(args.checkpoints_dir, 'log.txt'), 'a+')
    sys.stdout = f_handler
    print(' '.join(content))
    f_handler.close()
    sys.stdout = file
    print(' '.join(content))

def eval_metrics(probs, targets,cal_AUC=True):

    threshold_list = []
    for i in range(1, 50):
        threshold_list.append(i / 50.0)

    if cal_AUC:
        if isinstance(probs, torch.Tensor) and isinstance(targets, torch.Tensor):
            fpr, tpr, thresholds = roc_curve(y_true=targets.detach().cpu().numpy(),
                                             y_score=probs.detach().cpu().numpy())
            precision, recall, thresholds = precision_recall_curve(y_true=targets.detach().cpu().numpy(),
                                                                   probas_pred=probs.detach().cpu().numpy())            
        elif isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray):
            fpr, tpr, thresholds = roc_curve(y_true=targets,y_score=probs)
            precision, recall, thresholds = precision_recall_curve(y_true=targets,
                                                                   probas_pred=probs)       
        else:
            print('ERROR: probs or targets type is error.')
            raise TypeError
        auc_ = auc(x=fpr, y=tpr)
        auprc = auc(recall, precision)
    else:
        auc_ = 0
        auprc = 0

    threshold_best, rec_best, pre_best, acc_best, f1_best, mcc_best, pred_bi_best = 0, 0, 0, 0, 0, -1, None
    for threshold in threshold_list:
        rec, pre, acc, f1, mcc, pred_labels = th_eval_metrics(threshold, probs, targets,cal_AUC=False)
        if f1 > f1_best:
            threshold_best, rec_best, pre_best, acc_best, f1_best, mcc_best, pred_bi_best = threshold, rec, pre, acc, f1, mcc, pred_labels

    return threshold_best, rec_best, pre_best, acc_best, f1_best, mcc_best, auc_, auprc, pred_bi_best

def th_eval_metrics(threshold, probs, targets, cal_AUC=True):
    '''
    probs: 1-D array, targets: 1-D array
    '''
    pred_labels = np.abs(np.ceil(probs - threshold))
    rec = metrics.recall_score(targets, pred_labels)
    pre = metrics.precision_score(targets, pred_labels)
    acc = metrics.accuracy_score(targets, pred_labels)
    f1 = metrics.f1_score(targets, pred_labels)
    mcc = metrics.matthews_corrcoef(targets, pred_labels)

    if cal_AUC:
        # auc_ = roc_auc_score(targets, probs)
        fpr, tpr, thresholds = roc_curve(y_true=targets,y_score=probs)
        auroc = auc(x=fpr, y=tpr)
        precision, recall, thresholds = precision_recall_curve(y_true=targets,probas_pred=probs)    
        auprc = auc(recall, precision)
        return rec, pre, acc, f1, mcc, pred_labels, auroc, auprc
    else: 
        return rec, pre, acc, f1, mcc, pred_labels

def pred_on_site(threshold, probs, point2site, patch_idx, targets):

    proba_test_1d = torch.Tensor(probs).repeat_interleave(128, )
    patch_idx_1d = patch_idx.view(-1, )

    point_proba, _ = scatter_max(proba_test_1d, patch_idx_1d)
    site_proba, _ = scatter_max(point_proba, point2site)
    # print(site_proba)

    points_pred = torch.where(point_proba > threshold)[0]
    site_pred_idx = torch.unique(point2site.squeeze()[points_pred])
    # site_pred_id = res_idx2id[site_pred_idx]
    site_pred_label = torch.zeros_like(targets)
    site_pred_label[site_pred_idx] = 1
    return site_pred_label

def eval_on_site(targets, site_pred_label):
    rec = metrics.recall_score(targets, site_pred_label)
    pre = metrics.precision_score(targets, site_pred_label)
    acc = metrics.accuracy_score(targets, site_pred_label)
    f1 = metrics.f1_score(targets, site_pred_label)
    mcc = metrics.matthews_corrcoef(targets, site_pred_label)
    auc = metrics.roc_auc_score(targets, site_pred_label)
    return rec, pre, acc, f1, mcc, auc


def save_scripts(src_path, checkpoints_dir):

    dst_path = os.path.join(checkpoints_dir, "scripts")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    for script in os.listdir(src_path):
        if script.endswith(".py"):
            shutil.copy(os.path.join(src_path, f"{script}"), dst_path)

def compute_loss(criterion, output, sitelabel, pointlabel, patch_idx, point2site=None, patch2site=None):
    if point2site == None: #  tasktype == 'vertices'  
        # sitelabel -> patchlabel  
        loss = criterion(output, sitelabel)
        proba = torch.softmax(output, dim=1)[:, 1] # patch_proba
        return loss, proba
    else: #  tasktype == 'site'

        _, pos = scatter_max(torch.softmax(output, dim=1), patch2site, dim=0)
        pos = pos[:, 1]
        a = torch.where(pos < output.shape[0])

        # preds_sites = torch.softmax(output, dim=1)[pos[a]]
        # loss = criterion(torch.log(preds_sites), sitelabel[a])
        iface_softmax = torch.softmax(output, dim=1)[:, 1]
        site_proba = scatter_max(iface_softmax, patch2site, dim=0)[0][a]
        old_site_label = sitelabel[a]
        # print(site_proba.shape, old_site_label.shape)

        b = torch.unique(point2site[torch.where(pointlabel == 1)[0]])
        site_label = torch.zeros_like(sitelabel)
        site_label[b] = 1

        proba_test = torch.softmax(output, dim=1)[:, 1]
        iface_softmax, pos = scatter_max(proba_test, point2site[patch_idx[:, 0]].squeeze())
        a = torch.where(pos < output.shape[0])
        preds_sites = torch.softmax(output, dim=1)[pos[a]]
        loss = criterion(torch.log(preds_sites), site_label[a])
        iface_softmax = iface_softmax[a]
        site_label = site_label[a]

        # print(iface_softmax.shape, site_label.shape)
        

        # if site_label.shape[0] != iface_softmax.shape[0]:
        #     print(site_label.shape, iface_softmax.shape)
        #     new_t = torch.zeros_like(site_label)
        #     new_t[:iface_softmax.shape[0]] = iface_softmax
        #     iface_softmax = new_t
        #     print(site_label.shape, iface_softmax.shape)
        return loss, site_proba, old_site_label, iface_softmax, site_label

def read_esminfo(esmfile):
    if os.path.exists(esmfile):
        with open(esmfile, 'rb') as f:
            esm_info = pickle.load(f)
        f.close()
    return esm_info

def load_data(args):

    # if args.taskname == 'RNA' or args.taskname == 'DNA':
    train_ds, test_ds = DATASET[args.taskname]
    train_dir = f"./pt_{args.num_sampling}_{args.num_neighbors}_12_1/{train_ds}"
    test_dir = f"./pt_{args.num_sampling}_{args.num_neighbors}_12_1/{test_ds}"
    train_esmfile = f'./esm/{train_ds}'
    test_esmfile = f'./esm/{test_ds}'
    print(train_esmfile, test_esmfile)

    train_esminfo = read_esminfo(train_esmfile)
    test_esminfo = read_esminfo(test_esmfile)
    train_valid_file = os.listdir(train_dir)
    sample_idx = [i for i in range(len(train_valid_file))]
    random.shuffle(sample_idx)  
    train_filelist, valid_filelist = [], [] 
    for i in range(int(len(train_valid_file) * args.split_ratio)):
        train_filelist.append(train_valid_file[sample_idx[i]])
    for i in range(int(len(train_valid_file) * args.split_ratio), len(train_valid_file)):
        valid_filelist.append(train_valid_file[sample_idx[i]])    
    dataset_train = IterPatchDataset(train_dir, 'train', train_esminfo, train_filelist)
    dataset_val = IterPatchDataset(train_dir, 'valid', train_esminfo, valid_filelist)
    dataset_test = IterPatchDataset(test_dir, 'test', test_esminfo)
    # dataset_train = PatchDataset(train_dir, train_esminfo, train_filelist)
    # dataset_val = PatchDataset(train_dir, train_esminfo, valid_filelist)
    # dataset_test = PatchDataset(test_dir, test_esminfo)
    setattr(args, 'dataset', 'train')   
    printf(args, '#training meshes = %d' % len(dataset_train))
    setattr(args, 'dataset', 'val')
    printf(args, '#val meshes = %d' % len(dataset_val))
    setattr(args, 'dataset', 'test')
    printf(args, '#test meshes = %d' % len(dataset_test))
    return dataset_train, dataset_val, dataset_test

class EarlyStopping:
    def __init__(self, opt, save_dir, patience_stop=10, patience_lr=5, verbose=False, delta=0.0001, path='check1.pth'):
        self.opt = opt
        self.stop_patience = patience_stop
        self.lr_patience = patience_lr
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        # self.path = path
        self.save_dir = save_dir
        self.cnt = 0
        self.best_epoch = -1

    def __call__(self, metric, model, optimizer):
        if self.best_score is None:
            self.best_score = metric
            self.save_checkpoint(model)
            printf(self.opt, 'saving best model...')
        elif metric <= self.best_score + self.delta:
            self.counter += 1
            if self.counter == self.lr_patience:
                self.adjust_lr(model, optimizer)
                # self.counter = 0

            if self.counter >= self.stop_patience:
                self.early_stop = True
        else:
            self.best_score = metric
            self.best_epoch = self.cnt
            self.save_checkpoint(model)
            self.counter = 0
            printf(self.opt, 'saving best model...')
        self.cnt += 1

    def adjust_lr(self, model, optimizer):
        lr = optimizer.param_groups[0]['lr']
        lr = lr/10
        # lr = lr / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        path = os.path.join(self.save_dir, f"Bestmodel_Epoch{self.best_epoch}.pth")
        model.load_state_dict(torch.load(path))
        printf(self.opt, 'loading best model, changing learning rate to %.7f' % lr)

    def save_checkpoint(self, model):
        path = os.path.join(self.save_dir, f"Bestmodel_Epoch{self.cnt}.pth") # 0-based
        torch.save(model.state_dict(), path)

class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha fixed
    """
    def __init__(self, gamma=0, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, _input, target):
        
        # print(_input.shape, target.shape)
        score = _input[:, 1].squeeze()
        pt = torch.sigmoid(score)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss

