import os
from re import X
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
            wb_attacks=None,
            bb_attacks=None,
            batch_size=8,
            fbi_images=200, 
            transq_method="1", 
            images_path="./data/",
            bin_steps=10,
            device=0):
        self.modsim = FBI(fbi_images)

        self.transq = TransQ(transq_method)
        self.transq_method = transq_method
        self.images_path = images_path
        attacks = attacks if isinstance(attacks, list) else [attacks]
        self.attacks = {a: self._get_instance_attack(a) for a in attacks}
        self.batch_size = batch_size
        self.device = device
        self.bin_steps = bin_steps

        if self.transq_method == "2" and (wb_attacks is None or bb_attacks is None):
            raise ValueError("You must specify wb_attacks and bb_attacks when using transq_method=2")

        self.wb_attacks = wb_attacks
        self.bb_attacks = bb_attacks

    def __call__(self, x, sources: list, target: str, predictions: dict=None, y=None):
        model_instances = {model: get_model(model) for model in sources + [target]}

        # Get Transferable adversarial examples
        print("Getting Transferable Adversarial Examples")
        directions = {}
        for attack in self.attacks:
            directions[attack] = {}
            print("Run attack: ", attack)
            for model_name in sources:
                print("\t- Source: ", model_name)
                model = model_instances[model_name].to(self.device)
                advs = self._run_attack(x, model, attack, y=y)
                directions[attack][model_name] = advs - x
                directions[attack][model_name] /= torch.norm(directions[attack][model_name], dim=(1, 2), keepdim=True).norm(dim=3, keepdim=True)
            
        if self.transq_method == "2":
            wb_norms_mat = np.zeros((len(sources), len(x)))
            bb_norms_mat = np.zeros((len(sources), len(x)))
            for model_i, model_name in enumerate(sources):
                bb_norms_model = np.zeros((len(self.bb_attacks), len(x)))
                wb_norms_model = np.zeros((len(self.wb_attacks), len(x)))
                model = model_instances[model_name].to(self.device)
                print("White-box and black-box attacks for: ", model_name)
                for attack_i, attack in self.wb_attacks:
                    print("\t- attack: ", attack)
                    advs = self._run_attack(x, model, attack, y=y)
                    wb_norms_model[attack_i] = torch.norm((advs - x).flatten(1), dim=1)
                for attack_i, attack in self.bb_attacks:
                    print("\t- attack: ", attack)
                    advs = self._run_attack(x, model, attack, y=y)
                    bb_norms_model[attack_i] = torch.norm((advs - x).flatten(1), dim=1)

                wb_norms_mat[model_i] = wb_norms_model.min(dim=0).cpu().numpy()
                bb_norms_mat[model_i] = bb_norms_model.max(dim=0).cpu().numpy()
        else:
            wb_norms_mat = None
            bb_norms_mat = None
                

        # Get Distances in each transferable direction
        distances = {}
        print("Get distances between sources.")
        for attack in self.attacks:
            distances[attack] = {}
            print("Run attack: ", attack)
            for source in sources:
                distances[attack][source] = {}
                print("\t - From source: ", source)

                for target_name in sources:
                    print("\t\t - To target: ", target_name)
                    model_target = model_instances[target_name].to(self.device)
                    y_target = y if y is not None else model_target(x.to(self.device)).argmax(dim=1)
                    distances[attack][source][target_name] = self._line_search(
                        model_target, x, directions[attack][source], y_target)

        
        # Get predicted labels
        if predictions is None:
            for model in sources + [target]:
                predictions[model] = self._get_model_fbi_predictions(model)
        else:
            assert all([model in predictions for model in sources + [target]])
        
        # Get ModSim score
        modsim_score = self.modsim(sources, target, predictions)

        # Get TransQ score
        transq_score = {}
        for attack in self.attacks:
            tf_norms_mat = np.zeros((len(sources), len(sources), len(x)))
            for source_i, source in enumerate(sources):
                for target_i, target in enumerate(sources):
                    tf_norms_mat[source_i, target_i] = distances[attack][source][target]
            transq_score[attack] = self.transq(tf_norms_mat, wb_norms_mat, bb_norms_mat)

        modsim_score = np.repeat(modsim_score.reshape((-1, 1)), len(x), 1)
        fit_score = {attack: modsim_score * transq_score[attack] for attack in self.attacks}

        best_directions = []
        best_attacks = []
        best_models = []
        for i in range(len(x)):
            best_attack = None,
            best_source_i = None
            best_score = -np.inf
            for attack in self.attacks:
                if fit_score[attack][:, i].max() > best_score:
                    best_attack = attack
                    best_source_i = sources[fit_score[attack][:, i].argmax()]
                    best_score = fit_score[attack][:, i].max()

            best_directions.append(directions[best_attack][best_source_i][i])
            best_attacks.append(best_attack)
            best_models.append(best_source_i)

            print("Image ", i, " best attack: ", best_attack, " best source: ", best_source_i)

        best_directions = torch.stack(best_directions).cpu().numpy()
        return best_attacks, best_models, best_directions

    def _run_attack(self, x, model, attack, y=None):
        n_batch = int(np.ceil(x.shape[0] / self.batch_size))

        advs = []
        for batch_i in range(n_batch):
            batch_x = x[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
            batch_x = batch_x.to(self.device)
            if y is not None:
                batch_y = y[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
            else:
                batch_y = model(batch_x).argmax(dim=1)
            batch_y = batch_y.to(self.device)
            adv = self.attacks[attack](model, batch_x, batch_y)
            advs.append(adv.cpu())

        advs = torch.cat(advs, dim=0)
        return advs
    
    def _line_search(self, model, x, u, y_target):
        x = x.to(self.device)
        u = u.to(self.device)
        y_target = y_target.to(self.device)

        n_batch = int(np.ceil(x.shape[0] / self.batch_size))
        distances = []
        for batch_i in range(n_batch):
            batch_x = x[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
            batch_u = u[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
            batch_y_target = y_target[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]

            distance = self._line_search_batch(model, batch_x, batch_u, batch_y_target)
            distances.append(distance)

        distances = torch.cat(distances, axis=0)
        return distances.cpu().numpy()


    def _line_search_batch(self, model, x, u, label, start_norm=10):
        high = torch.ones(len(x), 1, 1, 1).to(self.device) * start_norm
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

        high = high.flatten()
        high = torch.where(
            is_high_adv, 
            high,
            torch.ones_like(high) * 500)
        return high

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

            from fit.attacks.transferable_attacks.di import DI
            return DI(
                epsilon=params.get("epsilon", 8), 
                num_iteration=params.get("num_iteration", 20),
                diversity_prob=params.get("diversity_prob", 0.5)
                )
        else:
            raise ValueError("Attack not supported: ", attack)

