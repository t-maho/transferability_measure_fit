from typing import Union, Any, Optional

import math
import torch
import os

import eagerpy as ep
import numpy as np

from foolbox.attacks import LinearSearchBlendedUniformNoiseAttack
from foolbox.models import Model
from foolbox.criteria import Criterion

from foolbox.attacks.base import MinimizationAttack, get_is_adversarial
from foolbox.attacks.base import get_criterion
from foolbox.attacks.base import T
from foolbox.distances import l2, linf, l1
from foolbox.devutils import atleast_kd
from fit.attacks.black-box_attacks.utils.generate_2d_dct_basis import generate_2d_dct_basis


class SubNoise(torch.nn.Module):
    """given subspace x and the number of noises, generate sub noises"""
    # x is the subspace basis
    def __init__(self, num_noises, x, shape, device=None):
        self.num_noises = num_noises
        self.x = torch.Tensor(x)
        self._device = device
        self.n_channel = shape[0]
        self.image_shape = shape[1:]
        super(SubNoise, self).__init__()

    def forward(self):
        r = torch.zeros([self.image_shape[0] * self.image_shape[1], self.n_channel * self.num_noises], dtype=torch.float32)
        noise = torch.randn([self.x.shape[1], self.n_channel * self.num_noises], dtype=torch.float32)
        if self._device is not None:
            noise = noise.cuda(self._device)
        sub_noise = torch.transpose(torch.mm(self.x, noise), 0, 1)
        r = sub_noise.view([ self.num_noises, self.n_channel, self.image_shape[0], self.image_shape[1]])

        if self._device is not None:
            r = r.cuda(self._device)
        return ep.astensor(r)

    
class BinarySearch:
    def __init__(
        self, 
        gamma: float = 1.0, 
        constraint="l2", 
        max_iteration=10000):
        self.constraint = constraint
        self.gamma = gamma
        self.max_iteration = max_iteration

    def get_threshold(self, originals, perturbed):
        d = np.prod(perturbed.shape[1:])
        if self.constraint == "linf":
            highs = linf(originals, perturbed)
            thresholds = highs * self.gamma / (d * d)
        else:
            highs = ep.ones(perturbed, len(perturbed))
            thresholds = self.gamma / (d * math.sqrt(d))
        lows = ep.zeros_like(highs)
        return highs, lows, thresholds


    def __call__(
        self,
        is_adversarial,
        originals: ep.Tensor,
        perturbed: ep.Tensor,
        return_steps=True
    ) -> ep.Tensor:
        # Choose upper thresholds in binary search based on constraint.
        highs, lows, thresholds = self.get_threshold(originals, perturbed)

        # use this variable to check when mids stays constant and the BS has converged
        old_mids = highs
        iteration = 0
        while ep.any(highs - lows > thresholds) and iteration < self.max_iteration:
            iteration += 1
            mids = (lows + highs) / 2
            mids_perturbed = self._project(originals, perturbed, mids)
            is_adversarial_ = is_adversarial(mids_perturbed)

            highs = ep.where(is_adversarial_, mids, highs)
            lows = ep.where(is_adversarial_, lows, mids)

            # check of there is no more progress due to numerical imprecision
            reached_numerical_precision = (old_mids == mids).all()
            old_mids = mids
            if reached_numerical_precision:
                # TODO: warn user
                break
        
        results = self._project(originals, perturbed, highs)
        if return_steps:
            return results, highs
        else:
            return results

    def _project(
            self, originals: ep.Tensor, perturbed: ep.Tensor, epsilons: ep.Tensor
        ) -> ep.Tensor:
        """Clips the perturbations to epsilon and returns the new perturbed

        Args:
            originals: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.
            epsilons: A batch of norm values to project to.
        Returns:
            A tensor like perturbed but with the perturbation clipped to epsilon.
        """
        epsilons = atleast_kd(epsilons, originals.ndim)
        if self.constraint == "linf":
            perturbation = perturbed - originals

            # ep.clip does not support tensors as min/max
            clipped_perturbed = ep.where(
                perturbation > epsilons, originals + epsilons, perturbed
            )
            clipped_perturbed = ep.where(
                perturbation < -epsilons, originals - epsilons, clipped_perturbed
            )
            return clipped_perturbed
        else:
            return (1.0 - epsilons) * originals + epsilons * perturbed



class GeoDA(MinimizationAttack):
    distance = l2
    def __init__(self, max_queries=10000, tol=0.0001, sigma=0.0002, mu=0.6, grad_estimator_batch_size=40, sub_dim=75, BS_params=None, sub_basis_path=None):
        self.sub_basis_path = sub_basis_path
        self.best_advs = None
        self.tol = tol
        self.sigma = sigma
        self.mu = mu
        self.binary_search = BinarySearch(**BS_params) if BS_params is not None else BinarySearch()
        self.grad_estimator_batch_size = grad_estimator_batch_size 
        self.sub_dim = sub_dim
        self.max_query = max_queries

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        starting_points: Optional[T] = None,
        init_attack_step=100,
        n_iteration=None,
        **kwargs: Any
    ):
        self._model = model
        originals, restore_type = ep.astensor_(inputs)
        del inputs, kwargs
        self._device = None if not originals.raw.is_cuda else originals.raw.device
        self.orig_labels = criterion
        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        # Initialize
        if starting_points is not None:
            assert is_adversarial(starting_points).all()
            self.best_advs = starting_points
        else:
            init_attack = LinearSearchBlendedUniformNoiseAttack(steps=init_attack_step)
            self.best_advs = init_attack.run(model, originals, criterion)

        self.best_advs = self.binary_search(is_adversarial, ep.astensor(originals), ep.astensor(self.best_advs), False)

        if True:
            if self.sub_basis_path is None:
                path = "/srv/tempdd/tmaho/geoda/2d_dct_basis_{}.npy".format(self.sub_dim)
            else:
                path = self.sub_basis_path

            if os.path.exists(path):
                print('Yes, we already have it ...')
                self.sub_basis_torch = np.load(path).astype(np.float32)
            else:
                print('Generating dct basis ......')
                self.sub_basis_torch = generate_2d_dct_basis(originals.shape[-2:], self.sub_dim, path).astype(np.float32)
                print('Done!\n')

            self.sub_basis_torch = torch.from_numpy(self.sub_basis_torch)
            if self._device is not None:
                self.sub_basis_torch = self.sub_basis_torch.cuda(self._device)

        iteration = round(self.max_query/500) if n_iteration is None else n_iteration
        q_opt_it = int(self.max_query  - (iteration)*25)
        q_opt_iter, iterate = opt_query_iteration(q_opt_it, iteration, self.mu)
        # print(q_opt_iter)
        advs_i = self.best_advs
        for i, q_opt_i in enumerate(q_opt_iter):
            # print("iteration", i)
            advs_i = self._run(is_adversarial, originals, advs_i, q_opt_i)
            # print("Iter", l2(originals, advs_i))

            distances_best = self.distance(originals, self.best_advs)
            distances_iter = self.distance(originals, advs_i)

            self.best_advs= self.best_advs.raw


            for i in range(len(advs_i)):
                if distances_best[i] > distances_iter[i]:
                    self.best_advs[i] = advs_i[i].raw

            self.best_advs = ep.astensor(self.best_advs)
            # print("best", l2(originals, self.best_advs))
            
        return self.best_advs

    def _run(self, is_adversarial, originals, x_b, q_opt):
        x_adv = []
        for i, x_b_i in enumerate(x_b):
            grad = 0
            # print(q_opt)
            # print(type(x_b_i))
            random_vec_o = ep.astensor(torch.randn(q_opt, *x_b_i.shape))
            
            grad_oi = self.black_grad_batch(is_adversarial, x_b_i, q_opt, random_vec_o, i)
            grad = grad_oi + grad
            x_adv_i = self.go_to_boundary(i, originals[i], grad, x_b_i)
            x_adv.append(x_adv_i.expand_dims(0))
        x_adv = ep.concatenate(x_adv, axis=0)
        x_adv = self.binary_search(is_adversarial, originals, x_adv, False)

        x_adv = x_adv.clip(0, 1)
        return x_adv
        
    def black_grad_batch(self, is_adversarial, x_boundary, q_max, random_noises, index_truth):
        
        batch_size = self.grad_estimator_batch_size

        num_batchs = math.ceil(q_max/batch_size)
        last_batch = q_max - (num_batchs-1)*batch_size
        EstNoise = SubNoise(batch_size, self.sub_basis_torch, x_boundary.shape, device=self._device)
        all_noises = []
        all_noises_boundary = []
        for j in range(num_batchs):
            if j == num_batchs-1:
                EstNoise_last = SubNoise(last_batch, self.sub_basis_torch, x_boundary.shape, device=self._device)
                current_batch = EstNoise_last()

            else:
                current_batch = EstNoise()

            for cb in current_batch:
                all_noises_boundary.append(x_boundary + self.sigma * cb)
                all_noises.append(cb) 


        grad_tmp = [] # estimated gradients in each estimate_batch
        z = []        # sign of grad_tmp
        for i, noise in enumerate(all_noises_boundary):
            is_adv = self._single_is_adversarial(noise, index_truth)
            if not is_adv:
                z.append(1)
                grad_tmp.append(all_noises[i])
            else:
                z.append(-1)
                grad_tmp.append(-all_noises[i])
        
        grad = -(1/q_max)*sum(grad_tmp)
        if self._device is not None:
            grad = ep.astensor(grad.raw.cuda(self._device))
        grad = grad.expand_dims(0)

        return grad

    def _single_is_adversarial(self, adv, truth_index):
        decision_adv = self._model(adv.expand_dims(0), indexes_truth=truth_index)[0].argmax(0)
        return decision_adv != self.orig_labels.labels[truth_index]

    def go_to_boundary(self, index_truth, x_0, grad, x_b):

        epsilon = 5

        num_calls = 1
        perturbed = x_0 

        if self.distance == l1 or self.distance == l2:
            grads = grad

        if self.distance == linf:
            grads = torch.sign(grad)/torch.norm(grad)
            
        while not self._single_is_adversarial(perturbed, index_truth):

            perturbed = x_0 + (num_calls*epsilon* grads[0])
            perturbed = perturbed.clip(0, 1)

            num_calls += 1
            
            if num_calls > 100:
                print('falied ... ')
                break
        return perturbed

def opt_query_iteration(Nq, T, eta): 
    coefs=[eta**(-2*i/3) for i in range(0,T)]
    coefs[0] = 1*coefs[0]
    
    sum_coefs = sum(coefs)
    opt_q=[round(Nq*coefs[i]/sum_coefs) for i in range(0,T)]
    
    if opt_q[0]>80:
        T = T + 1
        opt_q, T = opt_query_iteration(Nq, T, eta)
    elif opt_q[0]<50:
        T = T - 1

        opt_q, T = opt_query_iteration(Nq, T, eta)

    return opt_q, T

def uni_query(Nq, T, eta): 

    opt_q=[round(Nq/T) for i in range(0,T)]
    
        
    return opt_q