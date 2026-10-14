import torch
import torch as th
import torch.nn as nn

import config
from lib import kernel
import torch.nn.functional as F
from lib import contrastive_loss

EPSILON = 1E-9
DEBUG_MODE = False
SCLAE = 1E-3
def triu(X):
    # Sum of strictly upper triangular part
    return th.sum(th.triu(X, diagonal=1))


def _atleast_epsilon(X, eps=EPSILON):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return th.where(X < eps, X.new_tensor(eps), X)


def d_cs(A, K, n_clusters):
    """
    Cauchy-Schwarz divergence.

    :param A: Cluster assignment matrix
    :type A:  th.Tensor
    :param K: Kernel matrix
    :type K: th.Tensor
    :param n_clusters: Number of clusters
    :type n_clusters: int
    :return: CS-divergence
    :rtype: th.Tensor
    """
    nom = th.t(A) @ K @ A
    dnom_squared = th.unsqueeze(th.diagonal(nom), -1) @ th.unsqueeze(th.diagonal(nom), 0)

    nom = _atleast_epsilon(nom)
    dnom_squared = _atleast_epsilon(dnom_squared, eps=EPSILON**2)

    d = 2 / (n_clusters * (n_clusters - 1)) * triu(nom / th.sqrt(dnom_squared))
    return d


# ======================================================================================================================
# Loss terms
# ======================================================================================================================

class LossTerm:
    # Names of tensors required for the loss computation
    required_tensors = []

    def __init__(self, *args, **kwargs):
        """
        Base class for a term in the loss function.

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        """
        pass

    def __call__(self, net, cfg, extra):
        raise NotImplementedError()


class DDC1(LossTerm):
    """
    L_1 loss from DDC
    """
    required_tensors = ["hidden_kernel"]

    def __call__(self, net, cfg, extra):
        return d_cs(net.output, extra["hidden_kernel"], cfg.n_clusters)


class DDC2(LossTerm):
    """
    L_2 loss from DDC
    """
    def __call__(self, net, cfg, extra):
        n = net.output.size(0)
        return 2 / (n * (n - 1)) * triu(net.output @ th.t(net.output))


class DDC2Flipped(LossTerm):
    def __call__(self, net, cfg, extra):
        return 2 / (cfg.n_clusters * (cfg.n_clusters - 1)) * triu(th.t(net.output) @ net.output)


class DDC3(LossTerm):
    """
    L_3 loss from DDC
    """
    required_tensors = ["hidden_kernel"]

    def __init__(self, cfg):
        super().__init__()
        self.eye = th.eye(cfg.n_clusters, device=config.DEVICE)

    def __call__(self, net, cfg, extra):
        m = th.exp(-kernel.cdist(net.output, self.eye))
        return d_cs(m, extra["hidden_kernel"], cfg.n_clusters)



class Contrastive(LossTerm):
    large_num = 1e9

    def __init__(self, cfg):
        """
        Contrastive loss function

        :param cfg: Loss function config
        :type cfg: config.defaults.Loss
        """
        super().__init__()
        # Select which implementation to use
        if cfg.negative_samples_ratio == -1:
            self._loss_func = self._loss_without_negative_sampling
        else:
            self.eye = th.eye(cfg.n_clusters, device=config.DEVICE)
            self._loss_func = self._loss_with_negative_sampling

        # Set similarity function
        if cfg.contrastive_similarity == "cos":
            self.similarity_func = self._cosine_similarity
        elif cfg.contrastive_similarity == "gauss":
            self.similarity_func = kernel.vector_kernel
        else:
            raise RuntimeError(f"Invalid contrastive similarity: {cfg.contrastive_similarity}")

    @staticmethod
    def _norm(mat):
        return th.nn.functional.normalize(mat, p=2, dim=1)

    @staticmethod
    def get_weight(net):
        w = th.min(th.nn.functional.softmax(net.fusion.weights.detach(), dim=0))
        return w

    @classmethod
    def _normalized_projections(cls, net):
        n = net.projections.size(0) // 2
        h1, h2 = net.projections[:n], net.projections[n:]
        h2 = cls._norm(h2)
        h1 = cls._norm(h1)
        return n, h1, h2

    @classmethod
    def _cosine_similarity(cls, projections):
        h = cls._norm(projections)
        return h @ h.t()

    def _draw_negative_samples(self, net, cfg, v, pos_indices):
        """
        Construct set of negative samples.

        :param net: Model
        :type net: Union[models.simple_mvc.SiMVC, models.contrastive_mvc.CoMVC]
        :param cfg: Loss config
        :type cfg: config.defaults.Loss
        :param v: Number of views
        :type v: int
        :param pos_indices: Row indices of the positive samples in the concatenated similarity matrix
        :type pos_indices: th.Tensor
        :return: Indices of negative samples
        :rtype: th.Tensor
        """
        import  numpy as np
        cat = net.output.detach().argmax(dim=1)
        cat = th.cat(v * [cat], dim=0)
        prob,_ = net.output.detach().max(dim=1)
        n = len(prob)
        top_values, top_indices = th.topk(prob, int(n * 0.1))
        pl_matrix = 1 - self.eye[cat]
        length = len(pl_matrix)
        indices = th.arange(length, device=config.DEVICE) % n
        mask = ~th.isin(indices, top_indices)
        pl_matrix[mask, cat[mask]] = 1
        weights = pl_matrix[:, cat[[pos_indices]]].T
        for offset in range(0, length, n):
            diagonal_indices = th.arange(length, device=weights.device) + offset
            valid_mask = (diagonal_indices >= 0) & (diagonal_indices < length)
            weights[th.arange(length, device=weights.device)[valid_mask], diagonal_indices[valid_mask]] = 0
        for offset in range(-n, -length, -n):
            diagonal_indices = th.arange(length, device=weights.device) + offset
            valid_mask = (diagonal_indices >= 0) & (diagonal_indices < length)
            weights[th.arange(length, device=weights.device)[valid_mask], diagonal_indices[valid_mask]] = 0

        n_negative_samples = int(cfg.negative_samples_ratio * cat.size(0))
        negative_sample_indices = th.multinomial(weights, n_negative_samples, replacement=False)
        if DEBUG_MODE:
            self._check_negative_samples_valid(cat, pos_indices, negative_sample_indices)
        return negative_sample_indices

    @staticmethod
    def _check_negative_samples_valid(cat, pos_indices, neg_indices):
        pos_cats = cat[pos_indices].view(-1, 1)
        neg_cats = cat[neg_indices]
        assert (pos_cats != neg_cats).detach().cpu().numpy().all()

    @staticmethod
    def _get_positive_samples(logits, v, n):
        diagonals = []
        inds = []
        for i in range(1, v):
            diagonal_offset = i * n
            diag_length = (v - i) * n
            _upper = th.diagonal(logits, offset=diagonal_offset)
            _lower = th.diagonal(logits, offset=-1 * diagonal_offset)
            _upper_inds = th.arange(0, diag_length)
            _lower_inds = th.arange(i * n, v * n)
            if DEBUG_MODE:
                assert _upper.size() == _lower.size() == _upper_inds.size() == _lower_inds.size() == (diag_length,)
            diagonals += [_upper, _lower]
            inds += [_upper_inds, _lower_inds]

        pos = th.cat(diagonals, dim=0)
        pos_inds = th.cat(inds, dim=0)
        return pos, pos_inds

    def _loss_with_negative_sampling(self, net, cfg, extra):

        n = net.output.size(0)

        v = len(net.backbone_outputs)
        logits = self.similarity_func(net.projections) / cfg.tau

        pos, pos_inds = self._get_positive_samples(logits, v, n)
        neg_inds = self._draw_negative_samples(net, cfg, v, pos_inds)
        aaa = pos_inds.view(-1, 1)
        neg_inds = neg_inds.to(pos_inds.device)
        neg = logits[pos_inds.view(-1, 1), neg_inds]

        inputs = th.cat((pos.view(-1, 1), neg), dim=1)
        labels = th.zeros(v * (v - 1) * n, device=config.DEVICE, dtype=th.long)
        loss1 = th.nn.functional.cross_entropy(inputs, labels)

        loss = cfg.c1 * 10 * loss1

        return cfg.delta * loss


    def __call__(self, net, cfg, extra):
        return self._loss_func(net, cfg, extra)


class GraphCL(LossTerm):

    def __init__(self, cfg):
        super().__init__()
        self.similarity_func = self._cosine_similarity
    def _norm(mat):
        return th.nn.functional.normalize(mat, p=2, dim=1)
    @classmethod
    def _cosine_similarity(cls, projections):
        h = cls._norm(projections)
        return h @ h.t()

    def __call__(self, net, cfg, extra):
        numviews = len(net.views)

        device = net.backbone_outputs[0].device
        pos_nbrs_inx = []
        for v in range(numviews):
            X_nb = net.backbone_outputs[v].to(device)

            pos_nbrs_inx_v = []
            for i in range(X_nb.size(0)):
                distances = th.norm(X_nb - X_nb[i], dim=1)
                _, idx = th.topk(distances, k=5, largest=False)
                pos_nbrs_inx_v.append(idx[1:])
            pos_nbrs_inx.append(th.stack(pos_nbrs_inx_v).to(device))
        logits = []
        for v in range(numviews):
            logits_v = self.similarity_func(net.backbone_outputs[v].to(device)) / cfg.tau
            logits.append(logits_v)

        loss2 = 0
        for v in range(numviews):
            logits_v = logits[v].to(device)
            exp_logits_v = th.exp(logits_v)
            k0 = exp_logits_v.sum(dim=1) - th.diagonal(exp_logits_v)
            batch_indices = th.arange(logits_v.size(0), device=device).view(-1, 1)
            neighbor_indices = pos_nbrs_inx[v]
            nbr_logits = logits_v[batch_indices, neighbor_indices]
            mask = neighbor_indices != batch_indices
            loss_nbr = -th.log(th.exp(nbr_logits) / k0.unsqueeze(1))
            loss2 += loss_nbr[mask].sum()
        loss = cfg.c2 * SCLAE * (loss2)
        return cfg.graph_delta * loss

#==============================================================================

class Contrastive_label_level(LossTerm):
    large_num = 1e9

    def __init__(self, cfg):

        super().__init__()
        # Select which implementation to use
        self.similarity_func = self._cosine_similarity
        self._loss_func = self._loss_with_negative_sampling
        self.eye = th.eye(cfg.n_clusters, device=config.DEVICE)
    def _normalized_projections(cls, net):
        n = net.projections.size(0) // 2
        h1, h2 = net.projections[:n], net.projections[n:]
        h2 = cls._norm(h2)
        h1 = cls._norm(h1)
        return n, h1, h2
    @staticmethod
    def _norm(mat):
        return th.nn.functional.normalize(mat, p=2, dim=1)

    @staticmethod
    def get_weight(net):
        w = th.min(th.nn.functional.softmax(net.fusion.weights.detach(), dim=0))
        return w

    @classmethod
    def _cosine_similarity(cls, projections):
        h = cls._norm(projections)
        return h @ h.t()

    def _draw_negative_samples(self, net, cfg, v, pos_indices):
        cat = net.output.detach().argmax(dim=1)
        cat = th.cat(v * [cat], dim=0)
        prob,_ = net.output.detach().max(dim=1)
        n = len(prob)
        top_values, top_indices = th.topk(prob, int(n * 0.1))
        pl_matrix = 1 - self.eye[cat]
        length = len(pl_matrix)
        indices = th.arange(length, device=config.DEVICE) % n
        mask = ~th.isin(indices, top_indices)
        pl_matrix[mask, cat[mask]] = 1
        weights = pl_matrix[:, cat[[pos_indices]]].T
        for offset in range(0, length, n):
            diagonal_indices = th.arange(length, device=weights.device) + offset
            valid_mask = (diagonal_indices >= 0) & (diagonal_indices < length)
            weights[th.arange(length, device=weights.device)[valid_mask], diagonal_indices[valid_mask]] = 0
        for offset in range(-n, -length, -n):
            diagonal_indices = th.arange(length, device=weights.device) + offset
            valid_mask = (diagonal_indices >= 0) & (diagonal_indices < length)
            weights[th.arange(length, device=weights.device)[valid_mask], diagonal_indices[valid_mask]] = 0

        n_negative_samples = int(cfg.negative_samples_ratio * cat.size(0))
        negative_sample_indices = th.multinomial(weights, n_negative_samples, replacement=False)
        if DEBUG_MODE:
            self._check_negative_samples_valid(cat, pos_indices, negative_sample_indices)
        return negative_sample_indices

    @staticmethod
    def _check_negative_samples_valid(cat, pos_indices, neg_indices):
        pos_cats = cat[pos_indices].view(-1, 1)
        neg_cats = cat[neg_indices]
        assert (pos_cats != neg_cats).detach().cpu().numpy().all()

    @staticmethod
    def _get_positive_samples(logits, v, n):
        """
        Get positive samples

        :param logits: Input similarities
        :type logits: th.Tensor
        :param v: Number of views
        :type v: int
        :param n: Number of samples per view (batch size)
        :type n: int
        :return: Similarities of positive pairs, and their indices
        :rtype: Tuple[th.Tensor, th.Tensor]
        """
        diagonals = []
        inds = []
        for i in range(1, v):
            diagonal_offset = i * n
            diag_length = (v - i) * n
            _upper = th.diagonal(logits, offset=diagonal_offset)
            _lower = th.diagonal(logits, offset=-1 * diagonal_offset)
            _upper_inds = th.arange(0, diag_length)
            _lower_inds = th.arange(i * n, v * n)
            if DEBUG_MODE:
                assert _upper.size() == _lower.size() == _upper_inds.size() == _lower_inds.size() == (diag_length,)
            diagonals += [_upper, _lower]
            inds += [_upper_inds, _lower_inds]

        pos = th.cat(diagonals, dim=0)
        pos_inds = th.cat(inds, dim=0)
        return pos, pos_inds

    def _loss_with_negative_sampling(self, net, cfg, extra):

        n = net.output.size(0)
        v = 2
        loss1 = []
        for i in range (len(net.backbone_outputs)):
            logits = self.similarity_func(net.label_projections[i]) / cfg.tau

            pos, pos_inds = self._get_positive_samples(logits, v, n)
            neg_inds = self._draw_negative_samples(net, cfg, v, pos_inds)

            neg_inds = neg_inds.to(pos_inds.device)
            neg = logits[pos_inds.view(-1, 1), neg_inds]

            inputs = th.cat((pos.view(-1, 1), neg), dim=1)
            labels = th.zeros(v * (v - 1) * n, device=config.DEVICE, dtype=th.long)
            loss1.append(th.nn.functional.cross_entropy(inputs, labels))
        loss = sum(loss1)
        loss = cfg.c1 * 0.0001*loss
        return cfg.delta * loss

    def __call__(self, net, cfg, extra):
        return self._loss_func(net, cfg, extra)
#======================================================================================================================




class CC(LossTerm):

    def __init__(self, cfg):
        super().__init__()

    def get_weight(net):
        w = th.min(th.nn.functional.softmax(net.fusion.weights.detach(), dim=0))
        return w

    def __call__(self, net, cfg, extra):
        loss_device = th.device("cuda")
        criterion_cluster = contrastive_loss.ClusterLoss(7, 1.0, loss_device).to(
            loss_device)
        loss_cluster = []
        for i in range (len(net.backbone_ddc)):
            loss_cluster.append(criterion_cluster(net.backbone_ddc[i], net.output))
        loss = sum(loss_cluster)
        loss = (1-cfg.c1-cfg.c2) * loss
        return loss

# ======================================================================================================================
# Extra functions
# ======================================================================================================================

def hidden_kernel(net, cfg):
    return kernel.vector_kernel(net.hidden, cfg.rel_sigma)


# ======================================================================================================================
# Loss class
# ======================================================================================================================

class Loss(nn.Module):
    # Possible terms to include in the loss
    TERM_CLASSES = {
        "ddc_1": DDC1,
        "ddc_2": DDC2,
        "ddc_2_flipped": DDC2Flipped,
        "ddc_3": DDC3,
        "contrast": Contrastive,
        "instance_cluster": CC,
        "GraphCL":GraphCL,
        "Contrastive_label_level":Contrastive_label_level

    }
    # Functions to compute the required tensors for the terms.
    EXTRA_FUNCS = {
        "hidden_kernel": hidden_kernel,
    }

    def __init__(self, cfg):
        """
        Implementation of a general loss function

        :param cfg: Loss function config
        :type cfg: config.defaults.Loss
        """
        super().__init__()
        self.cfg = cfg

        self.names = cfg.funcs.split("|")
        self.weights = cfg.weights if cfg.weights is not None else len(self.names) * [1]

        self.terms = []
        for term_name in self.names:
            self.terms.append(self.TERM_CLASSES[term_name](cfg))

        self.required_extras_names = list(set(sum([t.required_tensors for t in self.terms], [])))

    def forward(self, net, ignore_in_total=tuple()):
        extra = {name: self.EXTRA_FUNCS[name](net, self.cfg) for name in self.required_extras_names}
        loss_values = {}
        for name, term, weight in zip(self.names, self.terms, self.weights):
            value = term(net, self.cfg, extra)
            # If we got a dict, add each term from the dict with "name/" as the scope.
            if isinstance(value, dict):
                for key, _value in value.items():
                    loss_values[f"{name}/{key}"] = weight * _value
            # Otherwise, just add the value to the dict directly
            else:
                loss_values[name] = weight * value

        loss_values["tot"] = sum([loss_values[k] for k in loss_values.keys() if k not in ignore_in_total])
        return loss_values

