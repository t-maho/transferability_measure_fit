import numpy as np
import torch


class RayS(object):
    def __init__(self, epsilon=8, max_queries=2000, order=np.inf, quantification=False):
        self.ord = order
        self.epsilon = epsilon
        self.eps = epsilon/255
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.queries = None
        self.quantification = quantification
        self.max_queries = max_queries

        self._history = {}
        self._best_dist_l2 = {}
        self._best_advs_l2 = {}

    def get_xadv(self, x, v, d, lb=0., ub=1.):
        if isinstance(d, int):
            d = torch.tensor(d).repeat(len(x)).cuda()
        out = x + d.view(len(x), 1, 1, 1) * v
        out = torch.clamp(out, lb, ub)
        return out
    
    def _quantify(self, x):
        return (x* 255).round() / 255

    def __call__(self, model, x, y, target=None, seed=None):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x, y): original image
        """
        self.labels = y
        self._originals = x
        shape = list(x.shape)
        dim = np.prod(shape[1:])
        if seed is not None:
            np.random.seed(seed)

        # init variables
        self._history = {i: [] for i in range(len(y))}
        self._best_dist_l2 = {i: np.inf for i in range(len(y))}
        self._best_advs_l2 = {i: None for i in range(len(y))}
        self.queries = torch.zeros_like(y).cuda()
        self.sgn_t = torch.sign(torch.ones(shape)).cuda()
        self.d_t = torch.ones_like(y).float().fill_(float("Inf")).cuda()
        working_ind = (self.d_t > self.eps).nonzero().flatten()

        stop_queries = self.queries.clone()
        dist = self.d_t.clone()
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)
 
        block_level = 0
        block_ind = 0
        for i in range(self.max_queries):
            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            valid_mask = (self.queries < self.max_queries) 
            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[valid_mask.nonzero().flatten(), start:end] *= -1.
            attempt = attempt.view(shape)

            self.binary_search(model, x, y, target, attempt, valid_mask)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.norm((self.x_final - x).view(shape[0], -1), self.ord, 1)
            stop_queries[working_ind] = self.queries[working_ind]
            working_ind = (dist > self.eps).nonzero().flatten()

            if torch.sum(self.queries >= self.max_queries) == shape[0]:
                print('out of queries')
                break 

        stop_queries = torch.clamp(stop_queries, 0, self.max_queries)

        if self.quantification:
            self.x_final = self._quantify(self.x_final)
        return self.x_final #, stop_queries, dist, (dist <= self.eps)

    # check whether solution is found
    def search_succ(self, model, x, y, target, mask):
        if self.quantification:
            x[mask] = self._quantify(x[mask])

        self.queries[mask] += 1
        if target:
            is_adv = model(x[mask]).argmax(1) == target[mask]
        else:
            is_adv = model(x[mask]).argmax(1) != y[mask]

        index_is_adv = 0
        for i, k in enumerate(mask.cpu().tolist()):
            if isinstance(k, bool):
                if k is True:
                    k = i
                else:
                    continue
              
            self._history[k].append((
                np.linalg.norm(self._originals[k].cpu() - x[k].cpu()),
                bool(is_adv[index_is_adv].cpu())
            ))

            if self._history[k][-1][1] and self._history[k][-1][0] < self._best_dist_l2[k]:
                self._best_dist_l2[k] = self._history[k][-1][0]
                self._best_advs_l2[k] = x[k]

            index_is_adv += 1
        return is_adv

    # binary search for decision boundary along sgn direction
    def binary_search(self, model, x, y, target, sgn, valid_mask, tol=1e-3):
        sgn_norm = torch.norm(sgn.view(len(x), -1), 2, 1)
        sgn_unit = sgn / sgn_norm.view(len(x), 1, 1, 1)

        d_start = torch.zeros_like(y).float().cuda()
        d_end = self.d_t.clone()

        initial_succ_mask = self.search_succ(model, self.get_xadv(x, sgn_unit, self.d_t), y, target, valid_mask)
        to_search_ind = valid_mask.nonzero().flatten()[initial_succ_mask]
        d_end[to_search_ind] = torch.min(self.d_t, sgn_norm)[to_search_ind]

        while len(to_search_ind) > 0:
            d_mid = (d_start + d_end) / 2.0
            search_succ_mask = self.search_succ(model, self.get_xadv(x, sgn_unit, d_mid), y, target, to_search_ind)
            d_end[to_search_ind[search_succ_mask]] = d_mid[to_search_ind[search_succ_mask]]
            d_start[to_search_ind[~search_succ_mask]] = d_mid[to_search_ind[~search_succ_mask]]
            to_search_ind = to_search_ind[((d_end - d_start)[to_search_ind] > tol)]

        to_update_ind = (d_end < self.d_t).nonzero().flatten()
        if len(to_update_ind) > 0:
            self.d_t[to_update_ind] = d_end[to_update_ind]
            self.x_final[to_update_ind] = self.get_xadv(x, sgn_unit, d_end)[to_update_ind]
            self.sgn_t[to_update_ind] = sgn[to_update_ind]
