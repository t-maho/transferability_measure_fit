import os
import numpy as np
import torch

from torch.utils.data import DataLoader

from fit.fbi import FBI
from fit.transq import TransQ
from fit.utils.load_models import get_model
from fit.utils.dataset import InferenceDataset


class FiT:
    def __init__(
            self, 
            attacks=["di"], 
            batch_size=8,
            fbi_images=200, 
            transq_method="1", 
            images_path="./data/",
            bin_steps=10,
            device=0):
        self.modsim = FBI(fbi_images)
        self.transq = TransQ(transq_method)
        self.images_path = images_path
        attacks = attacks if isinstance(attacks, list) else [attacks]
        self.attacks = {a: self._get_instance_attack(a) for a in attacks}
        self.batch_size = batch_size
        self.device = device
        self.bin_steps = bin_steps

    def __call__(self, x, sources: list, target: str, predictions: dict=None, y=None):
        model_instances = {model: get_model(model) for model in sources + [target]}

        # Get Transferable adversarial examples
        print("Getting Transferable Adversarial Examples")
        directions = {}
        for attack in self.attacks:
            directions[attack] = {}
            for model_name in sources:
                model = model_instances[model_name].to(self.device)
                advs = self._run_attack(x, model, attack, y=y)
                directions[attack][model_name] = advs - x
                directions[attack][model_name] /= torch.norm(directions[attack][model_name], dim=(1, 2, 3), keepdim=True)

        # Get Distances in each transferable direction
        distances = {}
        for attack in self.attacks:
            distances[attack] = {}
            for source in sources:
                distances[attack][source] = {}

                for target_name in sources:
                    target = model_instances[target_name].to(self.device)
                    y_target = y if y is not None else target(x).argmax(dim=1)
                    distances[attack][source][target] = self._line_search(
                        model, x, directions[attack][source], y_target)

        
        # Get predicted labels
        if predictions is None:
            for model in sources + [target]:
                predictions[model] = self._get_model_fbi_predictions(model)
        else:
            assert all([model in predictions for model in sources + [target]])
        
        # Get ModSim score
        modsim_score = self.modsim(sources, target, predictions)

        # Get TransQ score
        transq_score = self.transq(sources, target, predictions)

        fit_score = modsim_score * transq_score

        best_transferable_adv = []
        for i in range(x.shape[0]):
            best_source_i = sources[fit_score[i].argmax()]
            best_transferable_adv.append(advs[best_source_i][i])

        return best_transferable_adv

    def _run_attack(self, x, model, attack, y=None):
        print("Running attack: ", attack)
        n_batch = int(np.ceil(x.shape[0] / self.batch_size))

        for batch_i in range(n_batch):
            batch_x = x[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
            batch_x = batch_x.to(self.device)
            if y is not None:
                batch_y = y[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
            else:
                batch_y = model(batch_x).argmax(dim=1)

            adv = self.attacks[attack](model, batch_x, batch_y)
            advs.append(adv.cpu())

        advs = torch.cat(advs, dim=0)
        return advs
    
    def _line_search(self, model, x, u, y_target):
        x = x.cuda()
        u = u.cuda()
        y_target = y_target.cuda()

        n_batch = int(np.ceil(x.shape[0] / self.batch_size))
        distances = []
        for batch_i in range(n_batch):
            batch_x = x[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
            batch_u = u[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
            batch_y_target = y_target[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]

            distance = self._line_search_batch(model, batch_x, batch_u, batch_y_target)
            distances.append(distance)

        distances = np.concatenate(distances, axis=0)
        return distances


    def _line_search_batch(self, model, x, u, label, start_norm=10):
        high = torch.ones(len(x), 1, 1, 1) * start_norm
        low = torch.zeros_like(high)
        high = torch.where((model(x).argmax(1) == label).view(len(x), 1, 1, 1), high, low)
        n_step = 0
        while (model((x + high * u).clip(0, 1)).argmax(1) == label).any():
            if n_step > 40:
                break
            with torch.no_grad():
                is_adv = model((x + high * u).clip(0, 1)).argmax(1) != label
                is_adv = is_adv.view(len(x), 1, 1, 1)

                low = torch.where(is_adv, low, high)
                high = torch.where(is_adv, high, 2*high)
                n_step += 1

        for _ in range(self.bin_steps):
            with torch.no_grad():
                mid = (low + high) / 2
                is_adv = model((x + mid * u).clip(0, 1)).argmax(1) != label
                is_adv = is_adv.view(len(x), 1, 1, 1)
                low = torch.where(is_adv, low, mid)
                high = torch.where(is_adv, mid, high)
        is_high_adv = model((x + high * u).clip(0, 1)).argmax(1) != label
        return high.flatten(), is_high_adv.flatten()

    def _get_model_fbi_predictions(self, model):
        print("Predicting labels for model: ", model)
        dataset = InferenceDataset(self.images_path)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        for x, _ in data_loader:
            x = x.cuda()
            with torch.no_grad():
                y = self.models[model](x).argmax(dim=1)
            predictions.append(y.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        return predictions

    def _get_instance_attack(self, attack):
        attack = attack.lower().strip()
        if attack.startswith("di"):
            params = attack.split("-")[1:]
            params = {p.split("=")[0]: p.split("=")[1] for p in params}

            from fit.attacks.di import DI
            return DI(
                epsilon=params.get("epsilon", 8), 
                num_iteration=params.get("num_iteration", 20),
                diversity_prob=params.get("diversity_prob", 0.5)
                )
        else:
            raise ValueError("Attack not supported: ", attack)

