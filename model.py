import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import time
import torch.optim as optim
import os
import glob
import time
import pandas as pd
import random
import math
from net.ContextGNN import GNN_graphpred, GNN

from sklearn.metrics import confusion_matrix, roc_auc_score
from torch_geometric.nn import GATConv, TopKPooling, GINConv, GCNConv, GINEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_adj, dense_to_sparse, remove_self_loops
import torch_geometric.transforms as T

from torch.nn.functional import cosine_similarity

EPS = 1E-10
EDGE = 20
BEST = 0
WARM_UP = 50
DIM = 64
REVERSE = 0.8

gdc = T.GDC(self_loop_weight=None,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=20, dim=0),
            exact=True,
            )


class NeuroSparse(object):
    def __init__(self, in_channels, out_channels, num_site, ratio, alpha, n_epochs, lr, weight_decay, temperature,
                 site_adapt, augment, train_loader, val_loader, test_loader, threshold, l0, l1, l2, l3, pretrain_cl,
                 pretrain_encoder, beta, sampling, mix, aggr, gnn, edgepredictor):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuroSparse_model(in_channels,
                                    out_channels,
                                    num_site,
                                    ratio,
                                    alpha,
                                    temperature,
                                    site_adapt,
                                    sampling,
                                    aggr,
                                    gnn,
                                    edgepredictor)  # .to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.optimizer, self.scheduler = self._opt()
        self.augment = augment
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.l0 = l0  # class_loss
        if site_adapt:
            self.l1 = l1  # site_loss
        else:
            self.l1 = 0
        self.l2 = l2  # edge_loss
        self.l3 = l3  # MSE_loss
        self.ratio = ratio  # node drop prob
        self.site_adapt = site_adapt
        self.num_nodes = in_channels
        self.threshold = threshold
        self.pretrain_cl = pretrain_cl
        self.pretrain_encoder = pretrain_encoder
        self.beta = beta
        self.mix = mix
        # self.sampling = sampling

    def _opt(self):
        import torch.optim
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay
                                     )
        # optimizer = torch.optim.SGD(self.model.parameters(),
        #                              lr=self.lr,
        #                              weight_decay=self.weight_decay
        #                              )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=20,
                                                    gamma=0.5
                                                    )
        return optimizer, scheduler

    def pretrain_encoder(self):
        print('Pretraining encoder--')
        optimizer = torch.optim.Adam(self.model.ep.parameters(),
                                     lr=self.lr / 2
                                     )
        for _ in range(100):
            self.model.train()
            for data in self.train_loader:
                data.to(self.device)
                adj_logits = self.model.encoder(data.x, data.edge_index)
                adj_org_dense = to_dense_adj(data.edge_index, data.batch, data.edge_attr).squeeze()
                m = self.model.ep.m
                s = self.model.ep.s
                kl_loss = -0.5 * torch.mean(torch.sum(1 + 2 * s - m ** 2 - s.exp() ** 2, dim=1))
                loss = F.binary_cross_entropy_with_logits(adj_logits, adj_org_dense, pos_weight=None)
                loss -= kl_loss
                loss.backward()
                optimizer.step()
        print('--Done pretraining encoder')


    def pretrain_classifier(self):
        print('Pretraining nc net--')
        optimizer = torch.optim.Adam(self.model.classifier.parameters(),
                                     lr=self.lr,
                                     )
        for _ in range(10):
            self.model.train()
            for data in self.train_loader:
                data.to(self.device)
                output = self.model.classifier(data, data.edge_index, data.edge_attr)
                loss = F.nll_loss(output[0], data.y)
                if self.site_adapt:
                    loss += self.l1 * F.nll_loss(output[1], data.site)
                loss.backward()
                optimizer.step()
        print('--Done pretraining nc net')

    def calculate_metrics(self, labels, predicted):
        # Calculate AUC
        auc = roc_auc_score(labels, predicted)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(labels, predicted)
        TN, FP, FN, TP = conf_matrix.ravel()

        # Calculate specificity (SPE) and sensitivity (SEN)
        specificity = TN / (TN + FP)
        sensitivity = TP / (TP + FN)

        return auc, specificity, sensitivity
    def train(self, model, epoch):
        t = time.time()
        correct_train = 0
        loss_train = 0
        if self.mix:
            adj_lst = []
        model.train()
        y_true_train = []
        y_pred_train = []
        for data in self.train_loader:
            data.to(self.device)
            self.optimizer.zero_grad()
            if self.site_adapt:
                label, site, adj_org, adj_logits, z = model(data, data.edge_index, data.edge_attr)
                loss_st = F.nll_loss(site, data.site)
                loss_st_nonreduce = F.nll_loss(site, data.y, reduction='none') + EPS
                loss_st_nonreduce = torch.reshape(loss_st_nonreduce, (data.num_graphs, 1, 1))
            else:
                label, adj_org, adj_logits, z = model(data, data.edge_index, data.edge_attr)
                loss_st = 0
            m = self.model.ep.m
            s = self.model.ep.s
            kl_loss = -0.5 * torch.mean(torch.sum(1 + 2 * s - m ** 2 - s.exp() ** 2, dim=1))
            loss_ep = F.binary_cross_entropy_with_logits(adj_org, adj_logits.squeeze(), pos_weight=None)
            loss_ep -= kl_loss
            L2_loss = nn.MSELoss()
            loss_z = L2_loss(data.x, z)
            loss_cl = F.nll_loss(label, data.y)

            loss = self.l0 * loss_cl + self.l1 * loss_st + self.l2 * loss_z + self.l3 * loss_ep
            correct_train += self.count_correct(label, data.y)
            loss_train += loss_cl.item() * data.num_graphs
            loss.backward()
            self.optimizer.step()

            # Compute predictions
            output_probs = torch.exp(label)
            _, predicted = torch.max(output_probs, 1)
            # Append true labels and predicted labels for AUC calculation
            y_true_train += data.y.tolist()
            y_pred_train += predicted.tolist()

        self.scheduler.step()
        acc_train = correct_train / len(self.train_loader.dataset)
        loss_train /= len(self.train_loader.dataset)
        # Calculate metrics for training set
        auc_train, spe_train, sen_train = self.calculate_metrics(y_true_train, y_pred_train)
        if self.mix:
            adj_train = torch.mean(torch.stack(adj_lst), 0, keepdim=False)
            adj_train = torch.div(adj_train, torch.max(adj_train, 0)[0])
            adj_threshold = self.topk(adj_train, self.threshold)

            adj_sparse, _ = dense_to_sparse(adj_threshold)
            adj_sparse.to(self.device)
            adj_threshold.to(self.device)

        loss_val = 0
        correct_val = 0

        model.eval()
        y_true_val = []
        y_pred_val = []
        with torch.no_grad():
            for data in self.val_loader:
                data.to(self.device)
                if self.mix:
                    edge_dense = to_dense_adj(data.edge_index, data.batch, data.edge_attr).squeeze()
                    adj = (1 - self.beta) * adj_threshold + self.beta * edge_dense
                    adj = self.topk(adj, self.threshold)
                    adj, _ = dense_to_sparse(adj)
                    adj, _ = remove_self_loops(adj)
                    output = model(data, adj, data.edge_attr)
                else:
                    output = model(data, data.edge_index, data.edge_attr)
                loss_cl = F.nll_loss(output[0], data.y)
                if self.site_adapt:
                    loss_cl += self.l1 * F.nll_loss(output[1], data.site)
                loss_val += loss_cl.item() * data.num_graphs
                correct_val += self.count_correct(output[0], data.y)

                # Compute predictions
                output_probs = torch.exp(output[0])
                _, predicted = torch.max(output_probs, 1)

                # Append true labels and predicted labels for AUC calculation
                y_true_val += data.y.tolist()
                y_pred_val += predicted.tolist()

                del data, output
                torch.cuda.empty_cache()

            acc_val = correct_val / len(self.val_loader.dataset)
            loss_val /= len(self.val_loader.dataset)
            auc_val, spe_val, sen_val = self.calculate_metrics(y_true_val, y_pred_val)
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train),
              'acc_train: {:.4f}'.format(acc_train),
              'spe_train: {:.4f}'.format(spe_train),
              'sen_train: {:.4f}'.format(sen_train),
              'auc_train: {:.4f}'.format(auc_train),
              'loss_val: {:.4f}'.format(loss_val),
              'acc_val: {:.4f}'.format(acc_val),
              'spe_val: {:.4f}'.format(spe_val),
              'sen_val: {:.4f}'.format(sen_val),
              'auc_val: {:.4f}'.format(auc_val),
              'time: {:.4f}s'.format(time.time() - t)
              )
        # del adj_threshold
        # torch.cuda.empty_cache()
        if self.mix:
            return loss_val, adj_threshold  # adj_sparse
        return loss_val, acc_val, spe_val, sen_val, auc_val

    def fit(self, patience):
        t_total = time.time()
        if self.mix:
            adj_best = torch.zeros([200, 200], dtype=torch.float)
        # loss_values = []
        acc_values = []
        spe_values = []
        sen_values = []
        auc_values = []

        bad_count = 0
        best_acc = BEST
        best_spe = BEST
        best_sen = BEST
        best_auc = BEST
        best_epoch = 0
        model = self.model.to(self.device)
        if self.pretrain_encoder:
            self.pretrain_encoder()
        if self.pretrain_cl:
            self.pretrain_classifier()
        for epoch in range(self.n_epochs):
            if self.mix:
                loss_v, adj = self.train(model, epoch)
            else:
                loss_v, acc_v, spe_v, sen_v, auc_v = self.train(model, epoch)
            acc_values.append(acc_v)
            spe_values.append(spe_v)
            sen_values.append(sen_v)
            auc_values.append(auc_v)
            torch.save(model.state_dict(), '{}.pkl'.format(epoch))
            if epoch > WARM_UP:
                max_acc = max(acc_values)
                max_acc_idx = acc_values.index(max(acc_values))
                if max_acc > best_acc:
                    best_acc = acc_values[max_acc_idx]
                    best_spe = spe_values[max_acc_idx]
                    best_sen = sen_values[max_acc_idx]
                    best_auc = auc_values[max_acc_idx]
                    if self.mix:
                        adj_best = adj
                    best_epoch = epoch
                    bad_count = 0
                elif max_acc_idx == best_acc:
                    if spe_values[max_acc_idx] > best_spe or sen_values[max_acc_idx] > best_sen or auc_values[max_acc_idx] > best_auc:
                        best_sen = sen_values[max_acc_idx]
                        best_spe = spe_values[max_acc_idx]
                        best_auc = auc_values[max_acc_idx]
                else:
                    bad_count += 1
                    if self.mix:
                        del adj
                        torch.cuda.empty_cache()
                if bad_count == patience:
                    break

                files = glob.glob('*.pkl')
                for file in files:
                    epoch_nb = int(file.split('.')[0])
                    if epoch_nb < best_epoch:
                        os.remove(file)

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)

        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Restore the best model
        print('Loading {}th epoch'.format(best_epoch))
        model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
        print('Results of validation : acc_best: {:.4f}'.format(best_acc),
              'spe_spe: {:.4f}'.format(best_spe),
              'sen_sen: {:.4f}'.format(best_sen),
              'auc_auc: {:.4f}'.format(best_auc))

        # Testing
        if self.mix:
            self.test(model, adj_best)
        else:
            self.test(model)

    @torch.no_grad()
    def test(self, model, adj_threshold=None):
        correct_test, loss_test = 0, 0
        model.eval()
        if adj_threshold is not None:
            adj_threshold.to(self.device)
        y_true_test = []
        y_pred_test = []
        for data in self.test_loader:
            data.to(self.device)
            if self.mix:
                edge_dense = to_dense_adj(data.edge_index, data.batch, data.edge_attr).squeeze()
                adj = (1 - self.beta) * adj_threshold + self.beta * edge_dense
                adj = self.topk(adj, self.threshold)
                adj, _ = dense_to_sparse(adj)
                adj, _ = remove_self_loops(adj)
                output = model(data, adj, data.edge_attr)
            else:
                edge_index, _ = remove_self_loops(data.edge_index)
                output = model(data, data.edge_index, data.edge_attr)
            loss = F.nll_loss(output[0], data.y)
            loss_test += loss.item() * data.num_graphs
            correct_test += self.count_correct(output[0], data.y)

            # Compute predictions
            output_probs = torch.exp(output[0])
            _, predicted = torch.max(output_probs, 1)

            # Append true labels and predicted labels for AUC calculation
            y_true_test += data.y.tolist()
            y_pred_test += predicted.tolist()

        acc_test = correct_test / len(self.test_loader.dataset)
        loss_test = loss_test / len(self.test_loader.dataset)
        auc_score, specificity, sensitivity = self.calculate_metrics(y_true_test, y_pred_test)
        print("Test set result:",
              "loss= {:.4f}".format(loss_test),
              "accuracy= {:.4f}".format(acc_test),
              "AUC= {:.4f}".format(auc_score),
              "Specificity= {:.4f}".format(specificity),
              "Sensitivity = {:.4f}".format(sensitivity))

    def topk(self, A, K=None):
        if K is None:
            K = self.threshold
        num_nodes = A.shape[1]
        row_index = np.arange(num_nodes)
        k = int(num_nodes * K)
        A[torch.argsort(A, dim=0)[:num_nodes - k], row_index] = 0.0
        A = (A > 0.0).type_as(A)
        # A = A.triu(0) + torch.transpose(A.triu(1), 0, 1)
        return A

    @staticmethod
    def attn_loss(a, r):
        a = a.sort(dim=1).values
        loss = -torch.log(a[:, -int(a.size(1) * r):] + EPS).mean() - torch.log(
            1 - a[:, :int(a.size(1) * r)] + EPS).mean()
        return loss

    @staticmethod
    def count_correct(output, target):
        pred = output.max(1)[1].type_as(target)
        correct = pred.eq(target).sum().item()
        return correct


class NeuroSparse(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_site, ratio, alpha, temperature, site_adapt, sampling, aggr, gnn,
                 edgepredictor):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.encoder = VGAE(in_channels)
        self.classifier = GraphClassifier(in_channels, out_channels, num_site, ratio, site_adapt, aggr, gnn)
        self.site_adapt = site_adapt
        self.sampling = sampling
        self.edgepredictor = edgepredictor

    def bernouliSampling(self, adj_logits, adj_org):
        adj_logits = F.sigmoid(adj_logits)
        edge_probs = adj_logits / torch.max(adj_logits)

        edge_probs = self.alpha * edge_probs + (1 - self.alpha) * adj_org
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature,
                                                                         probs=edge_probs).rsample()
        adj_sampled = adj_sampled.triu(0) + torch.transpose(adj_sampled.triu(1), 1, 2)

        return adj_sampled, dense_to_sparse(adj_sampled)[0], dense_to_sparse(adj_sampled)[1]

    def add_remove_adj(self, adj_logits, adj_org, theta=0.2):
        batch_size, num_nodes = adj_logits.shape[0], adj_logits.shape[1]
        edge_probs = adj_logits
        edge_probs = edge_probs - torch.min(torch.min(edge_probs, dim=1).values, dim=-1).values.reshape(-1, 1, 1)
        edge_probs = edge_probs / torch.max(torch.max(edge_probs, dim=1).values, dim=-1).values.reshape(-1, 1, 1)

        n_edges = int(torch.nonzero(adj_org).shape[0] / batch_size)
        n_changes = int(EDGE * theta / 2)

        adj_inv = 1 - adj_org
        # element-wise product
        mask_rm = edge_probs * adj_org
        mask_rm_lst = []
        for i in range(batch_size):
            mask_rm_i = mask_rm[i]
            thres_rm = torch.topk(mask_rm_i, n_changes, dim=0, largest=True)[0][-1]
            mask_rm_i[mask_rm_i < thres_rm] = 0
            mask_rm_i = CeilNoGradient.apply(mask_rm_i)
            mask_rm_lst.append(mask_rm_i)
        mask_rm = torch.stack(mask_rm_lst)
        adj_new = adj_org - mask_rm

        mask_add = edge_probs * adj_inv
        mask_add_lst = []
        for i in range(batch_size):
            mask_add_i = mask_add[i]
            thres_add = torch.topk(mask_add_i, n_changes, dim=0, largest=True)[0][-1]
            mask_add_i[mask_add_i < thres_add] = 0
            mask_add_i = CeilNoGradient.apply(mask_add_i)
            mask_add_lst.append(mask_add_i)
        mask_add = torch.stack(mask_add_lst)
        adj_new = adj_new + mask_add

        return adj_new, dense_to_sparse(adj_new)[0], dense_to_sparse(adj_new)[1]

    def forward(self, data, edge_index, edge_attr):
        x, batch = data.x, data.batch
        # original FC
        tensor_reshaped = x.view(100, x.shape[1], x.shape[1])
        z, adj_logits = self.encoder(x, edge_index, edge_attr)  # dense

        edge_dense_org = self.to_dense_(edge_index, batch, None)
        if self.edgepredictor:
            if self.sampling == 'bernouli':
                _, edge_sparse_sampled, edge_sparse_attr = self.bernouliSampling(adj_logits, edge_dense_org)
            else:
                _, edge_sparse_sampled, edge_sparse_attr = self.add_remove_adj(adj_logits, edge_dense_org)
        else:
            edge_sparse_sampled, edge_sparse_attr = edge_index, edge_attr
        if not self.site_adapt:
            xy = self.classifier(data, adj_logits)
            return xy, edge_dense_org, adj_logits, z
        else:
            xy, xs = self.classifier(data, adj_logits)
            return xy, xs, edge_dense_org, adj_logits, z

    def renormalization(self):
        pass

    @staticmethod
    def to_dense_(edge_index, batch, edge_attr):
        edge_dense = to_dense_adj(edge_index, batch, edge_attr).squeeze()
        edge_dense = (edge_dense > torch.zeros_like(edge_dense)).type_as(edge_dense)
        return edge_dense


class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_site, ratio, site_adapt, aggr='cat', gnn='gcn'):
        super().__init__()
        dim = DIM
        if gnn == 'gcn':
            self.conv1 = GCNConv(in_channels, dim, cached=False, normalize=False)
            self.pool1 = TopKPooling(dim, ratio=ratio)
            self.conv2 = GCNConv(dim, dim, cached=False, normalize=False)
            self.pool2 = TopKPooling(dim, ratio=ratio)
        else:  # GIN
            self.conv1 = GINConv(nn.Sequential(nn.Linear(in_channels, dim), nn.ReLU(), nn.Linear(dim, dim)))
            self.pool1 = TopKPooling(dim, ratio=ratio)
            self.conv2 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
            self.pool2 = TopKPooling(dim, ratio=ratio)
        # self.conv3 = GCNConv(dim, dim, cached=False, normalize=False)
        # self.pool3 = TopKPooling(dim, ratio=ratio)
        if aggr == 'cat' or aggr == 'catmeanmanx':
            dim = 2 * DIM
        self.fc1 = nn.Linear(2 * dim, 4 * dim)
        self.bn1 = nn.BatchNorm1d(4 * dim)
        self.fc2 = nn.Linear(4 * dim, 8 * dim)
        self.bn2 = nn.BatchNorm1d(8 * dim)
        self.fc3 = nn.Linear(8 * dim, out_channels)

        if site_adapt:
            self.siteclassifier = site_classifier(40000, num_site)
            # self.siteclassifier = site_classifier(2 * dim, num_site)
        else:
            self.siteclassifier = None
        self.aggr = aggr

    def forward(self, data, adj_logits):
        x, batch = data.x, data.batch
        abs_adj_logits = torch.abs(adj_logits)  # edge attributes
        num_elements = int(0.3 * adj_logits.shape[1] * adj_logits.shape[2])
        adj_logits_res = torch.zeros_like(adj_logits)
        for i in range(adj_logits.shape[0]):
            _, topk_indices = torch.topk(abs_adj_logits[i].view(-1), num_elements, largest=True)
            adj_logits_res[i].view(-1)[topk_indices] = abs_adj_logits[i].view(-1)[topk_indices]
        # predict edge features
        adj_logits_res[adj_logits_res != 0]=1
        # new fc
        new_fc = torch.mul(x.view(100, 200, 200), adj_logits_res)
        x = torch.mul(x.view(100,200,200), adj_logits_res)
        x = x.view(100, -1)
        xy = self.bn1(F.relu(self.fc1(x)))
        xy = F.dropout(xy, p=0.6, training=self.training)
        xy = self.bn2(F.relu(self.fc2(xy)))
        xy = F.log_softmax(self.fc3(xy), dim=-1)

        if self.siteclassifier is not None:
            xs = self.siteclassifier(x)
            return xy, xs
        return xy


class EdgePredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        embedded_channels = in_channels
        self.conv_base = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, embedded_channels)
        self.conv_logstd = GCNConv(hidden_channels, embedded_channels)

    def forward(self, x, edge_index):
        h = self.conv_base(x, edge_index)
        m = self.conv_mu(h, edge_index)  # .relu()
        s = self.conv_logstd(h, edge_index)  # .relu()
        return m, s


class VGAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.nodes = in_channels
        embedded_channels = in_channels
        # self.conv_base = GCNConv(in_channels, hidden_channels)
        # self.conv_base = GATConv(in_channels, in_channels)
        # self.conv_base = GINEConv(nn.Sequential(nn.Linear(in_channels, 200)), edge_dim = 4)
        self.conv_base = GNN(2, in_channels, JK='last', drop_ratio=0.2, gnn_type='gin')
        # self.conv_base = GINConv(nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)))
        self.conv_mu = GCNConv(embedded_channels, embedded_channels)
        # self.conv_mu = GATConv(embedded_channels, embedded_channels)
        # self.conv_mu = GINConv(nn.Sequential(nn.Linear(in_channels, 200)))
        self.conv_logstd = GCNConv(embedded_channels, embedded_channels)
        # self.conv_logstd = GATConv(embedded_channels, embedded_channels)
        # self.conv_logstd = GINConv(nn.Sequential(nn.Linear(in_channels, 200)))

    def forward(self, x, adj, edge_attr):
        h = self.conv_base(x, adj, edge_attr)
        self.m = self.conv_mu(h, adj).relu()
        self.s = self.conv_logstd(h, adj).relu()
        z = self.m + torch.randn_like(self.s) * torch.exp(self.s)
        z1 = z.view(-1, self.nodes, self.nodes)
        adj_logits = torch.matmul(z1, torch.transpose(z1, 1, 2))
        return z, adj_logits


def get_topk_matrix(self, A, K=0.1):
    num_nodes = A.shape[1]
    row_index = np.arange(num_nodes)
    k = int(num_nodes * K)
    A[torch.argsort(A, dim=0)[:num_nodes - k], row_index] = 0.0
    A = (A > 0.0).type_as(A)
    # A = A.triu(0) + torch.transpose(A.triu(1), 0, 1)
    return A

def cos_similar(p, q):
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    sim_matrix = torch.where(torch.isnan(sim_matrix), torch.full_like(sim_matrix, 0), sim_matrix)
    return sim_matrix



def sameLoss(x, x_aug):
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    loss = 2 - cosine_similarity(x, x_aug, dim=-1).mean() - cosine_similarity(x_abs, x_aug_abs, dim=-1).mean()
    return loss


class CeilNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.ceil()

    @staticmethod
    def backward(ctx, g):
        return g


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return REVERSE * grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


class site_classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(site_classifier, self).__init__()
        # dim = int(0.5 * in_features)
        dim = 256
        self.fc1 = nn.Linear(in_features, dim)  # 2dim, 4dim
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, 2 * dim)
        self.bn2 = nn.BatchNorm1d(2 * dim)
        self.fc3 = nn.Linear(2 * dim, out_features)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn2(F.relu(self.fc2(x)))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x