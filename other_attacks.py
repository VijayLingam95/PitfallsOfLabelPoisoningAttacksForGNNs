import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import random

import scipy as sp
from sklearn.utils.extmath import randomized_svd



def similarity_matrix(features, gamma):
    tmp = features @ features.T
    n_data = features.shape[0]
    diag = np.diag(tmp)
    S = gamma * (2 * tmp - diag.reshape(1, n_data) - diag.reshape(n_data, 1))
    return np.exp(S)

def diagonal(similarity_matrix):
    D = np.diag(np.sum(similarity_matrix, axis=1))
    return D

def func_gradient(adj, features, labels, idx_train, y_l, args):
    """
    Borrowed from: https://github.com/lgl6633/MG-attack
    """

    # calculate the objective function value
    y_0 = torch.zeros(size=(labels.shape[0], labels.max().item() + 1))
    for i in idx_train:
        y_0[i][labels[i]] = 1.0
    y_lo = y_0.float()

    idx_all = np.arange(labels.shape[0])
    idx_other = np.delete(idx_all, idx_train)

    #idx_other_t = torch.from_numpy(idx_other)

    # ground truth
    y0 = torch.zeros(size=(labels.shape[0], labels.max().item() + 1))
    for j in idx_other:
        y0[j][labels[j]] = 1.0
    y_uo = y0.float()
    #y_u_u = torch.index_select(y_uo, 0, idx_other_t)
    y_u_u = y_uo[idx_other]


    #Similarity Metrix
    features_np = features.numpy()
    S = similarity_matrix(features_np, gamma=0.1)
    # S=S.numpy()
    D = diagonal(S)
    y_tr = labels[idx_train]
    n_tr = len(y_tr)
    Suu = S[n_tr:, n_tr:]
    Duu = D[n_tr:, n_tr:]
    Sul = S[n_tr:, :n_tr]
    SM_A = np.linalg.inv(Duu - Suu) @ Sul
    SM_A = torch.from_numpy(SM_A).float()
    y_pre_sm = SM_A @ y_lo[idx_train] #original predicted y_u
    y_sm_u = SM_A @ y_l[idx_train] #A@adv_y

    # y_sm = torch.zeros(size=(labels.shape[0], labels.max().item() + 1))
    # y_sm[idx_other] = y_sm_u


    #APPNP
    for _ in range(args.K):
        y_appo = torch.matmul(adj, y_lo)
        y_appo = (1 - args.alpha) * y_appo + args.alpha * y_lo
    #y_pre_app = torch.index_select(y_appo, 0, idx_other_t) #original predicted y_u
    y_pre_app = y_appo[idx_other]

    for _ in range(args.K):
        advy_app = torch.matmul(adj, y_l)
        advy_app = (1 - args.alpha) * advy_app + args.alpha * y_l
    #y_app_u = torch.index_select(advy_app, 0, idx_other_t) #A@adv_y
    y_app_u = advy_app[idx_other]


    #S^K
    SK_A = adj.pow(10)
    y_sk_o = SK_A @ y_lo
    #y_pre_sk = torch.index_select(y_sk_o, 0, idx_other_t) #original predicted y_u
    y_pre_sk = y_sk_o[idx_other]

    y_sk = SK_A @ y_l
    #y_sk_u = torch.index_select(y_sk, 0, idx_other_t) #A@adv_y
    y_sk_u = y_sk[idx_other]


    # SM - SP
    #tmp = y_sm_u - y_pre_app
    #SK - SP
    #tmp = y_sk_u - y_pre_app


    #SM - SM
    #tmp = y_sm_u - y_pre_sm
    #SK - SM
    #tmp = y_sk_u - y_pre_sm
    # SP - SM
    #tmp = y_app_u - y_pre_sm

    #SM-SK
    tmp = y_sm_u - y_pre_sk
    #SK-SK
    # tmp = y_sk_u - y_pre_sk
    #SP-SK
    #tmp = y_app_u - y_pre_sk


    #SM - yu
    #tmp = y_sm_u - y_u_u
    # SK - yu
    #tmp = y_sk_u - y_u_u
    # SP - yu
    #tmp = y_app_u - y_u_u

    f = -0.5 * torch.sum(tmp * tmp)
    return f

def gradient_max(adj, features, idx_train, labels, noise_num, args):
    """
    Borrowed from: https://github.com/lgl6633/MG-attack
    """

    if noise_num == 0:  # fix a corner case
        return labels
    nclass = max(labels) + 1
    labels_new = labels.copy()

    for _ in range(2):
        y0 = torch.zeros(size=(labels_new.shape[0], labels_new.max().item() + 1))
        for i in idx_train:
            y0[i][labels_new[i]] = 1.0
        y_l = y0.float()
        adv_y = Variable(y_l, requires_grad=True)

        f = func_gradient(adj, features, labels, idx_train, adv_y, args)
        grad = torch.autograd.grad(f, adv_y, retain_graph=False, create_graph=False)[0]
        grad_train = []
        for i in idx_train:
            grad_train.append(grad[i][labels_new[i]].tolist())  # list
        grad_sorted = sorted(grad_train)[-noise_num:]  #[-noise_num * nclass:]

        index_g = []
        for i in range (len(grad_train)):
            if grad_train[i] in grad_sorted:
                index_g.append(i)
                if len(index_g) == noise_num: #noise_num * nclass:
                    break
        labels_new = labels.copy()
        labels_new[idx_train[index_g]] = max(labels)

    return labels_new


def flip_label(lbl, total_classes):
    """
        Flip given label to a random false label
    """
    lbl_ = lbl.copy()
    
    lbl_class = np.argmax(lbl)
    
    possible_classes = [i for i in range(total_classes) if i != lbl_class]
    
    lbl_[lbl_class] = 0.
    lbl_[np.random.choice(possible_classes)] = 1.
    
    return lbl_
    

def poison_labels_random(labels, train_idx, BUDGET):
    
    labels_ = labels.copy()
    total_classes = labels_.shape[1]
    
    random_ids = np.random.choice(train_idx, size=BUDGET, replace=False)
    
    for idx in random_ids:
        labels_[idx] = flip_label(labels_[idx], total_classes)
      
    # santiy check to verify total poisoned labels
    #print(np.where(labels.argmax(1) != labels_.argmax(1))[0].shape[0], budget)

    return labels_


def poison_labels_degree(labels, adj, train_idx, BUDGET):

    labels_ = labels.copy()
    total_classes = labels_.shape[1]

    # get train node ids correspondind to max degree 
    degree_ids =  train_idx[np.argsort(np.array(-adj[train_idx].sum(1)).squeeze())[:BUDGET]]

    for idx in degree_ids:
        labels_[idx] = flip_label(labels_[idx], total_classes)

    return labels_


def accuracy(y_pred, y_truth):
    correct = np.sum(y_pred == y_truth)
    return correct / len(y_truth)


class LabelProp(object):

    """
    Adapted from: https://github.com/xuanqing94/AdvSSL
    """

    def __init__(self, data):
        self.data_name = data.name

        self.features, self.labels = data.dataset_data['X'], data.dataset_data['labels'].argmax(1)

        self.train_ids = data.train_ids 
        self.val_ids = data.val_ids 
        self.test_ids = data.test_ids 

        self.unlabeled_ids = np.delete(np.arange(self.features.shape[0]), np.append(self.train_ids, self.val_ids))

    def set_hparam(self, gamma):
        self.gamma = gamma
        return self

    def split_data(self):
        # split data
        train_features = self.features[self.train_ids]
        test_features = self.features[self.test_ids]
        train_labels = self.labels[self.train_ids]
        test_labels = self.labels[self.test_ids]

        self.X_tr, self.X_te, self.y_tr, self.y_te = \
                train_features, test_features, train_labels, test_labels

    def similarity_matrix(self, X, gamma):
        cache_file = f'./{self.data_name}.npy'
        if os.path.exists(cache_file):
            tmp = np.load(cache_file)
        else:
            if sp.sparse.issparse(X):
                X = X.tocoo()
                tmp = X @ X.T
                tmp = np.asarray(tmp.todense())
            else:
                tmp = X @ X.T
            np.save(cache_file, tmp)
        n_data = X.shape[0]
        diag = np.diag(tmp)
        S = gamma * (2 * tmp - diag.reshape(1, n_data) - diag.reshape(n_data, 1))
        return np.exp(S)

    def diagnoal(cls, similarity_matrix):
        D = np.diag(np.sum(similarity_matrix, axis=1))
        return D

    def training(self, n_trial=1, perturb=None):
        mse_te = []
        for k_trial in range(n_trial):
            y_pred = self.prediction(perturb)
            # evaluation
            mse = self.metric(y_pred, self.y_te)
            mse_te.append(mse)
        return np.mean(mse_te), np.std(mse_te)

    def prediction(self, perturb=None):

        X = self.features
        y_tr = self.y_tr
        if perturb is not None:
            if self.task == 'regression':
                y_tr = self.y_tr + perturb
            elif self.task == 'classification':
                y_tr = self.y_tr * perturb
            else:
                raise ValueError(f'Invalid self.task: {self.task}')
        S = self.similarity_matrix(X, self.gamma)
        D = np.diag(np.sum(S, axis=1, keepdims=False))

        Suu = S[self.unlabeled_ids, :][:, self.unlabeled_ids]
        Duu = D[self.unlabeled_ids, :][:, self.unlabeled_ids]
        Sul = S[self.unlabeled_ids, :][:, self.train_ids]

        tmp = np.linalg.inv(Duu - Suu) @ Sul
        if self.task == 'regression':
            y_pred = np.dot(tmp, y_tr)
        elif self.task == 'classification':
            y_pred = np.sign(np.dot(tmp, y_tr))

        else:
            raise ValueError(f'Invalid self.task: {self.task}')
        return y_pred


    def perturb_y_classification(self, c_max, supervised=True):
        self.split_data()
        X = self.features
        y_tr = self.y_tr
        n_tr = len(y_tr)
        if supervised:
            # knowns the ground truth label
            y_te = self.y_te
        else:
            # does not know the ground truth label
            y_te = self.prediction()
        S = self.similarity_matrix(X, self.gamma)
        D = np.diag(np.sum(S, axis=1, keepdims=False))

        # Suu = S[self.unlabeled_ids, :][:, self.unlabeled_ids]
        # Duu = D[self.unlabeled_ids, :][:, self.unlabeled_ids]
        # Sul = S[self.unlabeled_ids, :][:, self.train_ids]   

        Suu = S[self.test_ids, :][:, self.test_ids]
        Duu = D[self.test_ids, :][:, self.test_ids]
        Sul = S[self.test_ids, :][:, self.train_ids]     

        K = np.linalg.inv(Duu - Suu) @ Sul
        # greedy / threshold / probablistic / exhaustive_search
        distortion = probablistic_method_soft(K, y_tr, y_te, c_max)
        return distortion


def probablistic_method_soft(K, y_l, y_u, c):
    """
    Adapted from: https://github.com/xuanqing94/AdvSSL
    """

    # move numpy array to torch tensor
    K = torch.from_numpy(K).float()
    n_l = len(y_l)
    y_l = torch.from_numpy(y_l).float()
    y_u = torch.from_numpy(y_u).float()
    d_y = torch.ones_like(y_l, requires_grad=True)
    # loss = -0.5 * || tanh(K (yl*d_y)) - yu || ^2 + beta * 0.125 * || 1-d_y || ^2
    alpha = 0.5 * torch.ones_like(y_l, requires_grad=True)
    lam = 0.01
    tau = 0.2
    T = 10
    lr = 1.0e-6
    for i in range(1500):
        U1 = torch.rand((n_l,))
        U2 = torch.rand((n_l,))
        epsilon = torch.log(torch.log(U1) / torch.log(U2))
        tmp = torch.exp((torch.log(alpha / (1.0 - alpha)) + 1e-8) / tau) #epsilon
        #print('tmp: ', tmp)
        z = 2.0 / (1.0 + tmp) - 1.0     # normalize z from [0, 1] to [-1, 1]
        v = y_l * z

        diff = torch.tanh(T * K @ v) - y_u
        loss = -0.5 * torch.sum(diff * diff) + 0.5 * lam * torch.sum(alpha * alpha)
        #print(loss.item())
        g = torch.autograd.grad(loss, [alpha])[0]
        alpha.detach().add_(-lr * g)
        alpha.detach().clamp_(1.0e-3, 1.0-1.0e-3)

    alpha = alpha.detach().numpy()
    # evaluate function value
    idx = np.argsort(alpha)[::-1]
    d_y = np.ones_like(y_l)
    count = 0
    for i in idx:
        if alpha[i] > 0.5 and count < c:
            d_y[i] = -1
            count += 1
            if count == c:
                break
    #val = func_val(K, y_l, d_y, y_u)
    #print('#flip: ', np.sum(d_y < 0), 'function value: ', val)
    return d_y

def lp_attack(g_data, BUDGET):

    """
    Adapted from: https://github.com/xuanqing94/AdvSSL
    """

    dname = g_data.name
 
    # create LP object
    lp = LabelProp(data=g_data)
    class_bincount = np.bincount(lp.labels)
    top_classes = np.argsort(-class_bincount)

    clean_labels = lp.labels.copy()

    #get top 2 classes ids in train_ids
    orig_train_ids = lp.train_ids.copy()
    top_two_class_ids = np.append(np.where(lp.labels[orig_train_ids]==top_classes[0])[0], \
                                  np.where(lp.labels[orig_train_ids]==top_classes[1])[0])

    #reset train ids to contain only nodes corresponding to top-2 classes
    lp.train_ids = orig_train_ids[top_two_class_ids]

    lp.set_hparam(0.1)

    # get label flips 
    d_y = lp.perturb_y_classification(BUDGET)

    # perturb labels
    perturb_ids = np.where(d_y == -1)[0]

    class_a, class_b = top_classes[0], top_classes[1]
    for pid in orig_train_ids[ top_two_class_ids[perturb_ids] ]:
        if lp.labels[pid] == class_a:
            lp.labels[pid] = class_b
        else:
            lp.labels[pid] = class_a

    # if there is still available budget for poisoning continue to next two top classes
    remaining = BUDGET - perturb_ids.shape[0]

    if remaining:
        if dname != 'pubmed':
            class_a, class_b = top_classes[2], top_classes[3]
            next_two_class_ids = np.append(np.where(lp.labels[orig_train_ids]==top_classes[2])[0], \
                                           np.where(lp.labels[orig_train_ids]==top_classes[3])[0])
 
        else:
            class_a, class_b = top_classes[1], top_classes[2]
            next_two_class_ids = np.append(np.where(lp.labels[orig_train_ids]==top_classes[1])[0], \
                                           np.where(lp.labels[orig_train_ids]==top_classes[2])[0]) 

        lp.train_ids = orig_train_ids[next_two_class_ids]

        d_y2 = lp.perturb_y_classification(remaining)
        new_perturb_ids = np.where(d_y2 == -1)[0]

        for pid in orig_train_ids[ next_two_class_ids[new_perturb_ids] ]:
            if lp.labels[pid] == class_a:
                lp.labels[pid] = class_b
            else:
                lp.labels[pid] = class_a
    
    print("Total train labels flipped using LP-attack:", np.where(clean_labels != lp.labels)[0].shape)

    changed = np.where(clean_labels != lp.labels)[0]

    #reset train_ids
    lp.train_ids = orig_train_ids
    lp.metric = accuracy
    lp.task = 'classification'

    #reset split data
    lp.split_data()

    return lp.labels