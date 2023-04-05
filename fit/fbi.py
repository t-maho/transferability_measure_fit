import math
import numpy as np
import torch

from collections import Counter

class FBI:
    def __init__(self, num_images: int = 100):
        self._num_images = num_images

    def __call__(self, sources: list, target: str, truths: dict):
        p_target = torch.Tensor(truths[target]).long().unsqueeze(0).repeat(len(sources), 0)
        p_sources = torch.Tensor([truths[source] for source in sources]).long()

        images_indexes = self._select_images(p_sources)

        p_sources = p_sources[:, images_indexes[:self._num_images]]

        mi = self._similarity_mutual_information(p_sources, p_target).numpy()
        fbi_score_entropy_index = {"###".join(sorted([sources[i], target])): mi[i] for i in range(len(sources))}
        return fbi_score_entropy_index

    def _select_images(self, prediction: np.ndarray):
        """Select images with high entropy.
        Args:
            prediction (np.ndarray): Prediction of the models. Shape: (num_images, n_models).
        Returns:
            indexes (list): Indexes of the selected images.
        """
            
        scores = []
        for v in prediction:
            count = Counter([str(e) for e in v])
            count = [e / len(v) for e in count.values()]
            scores.append([- e * math.log(e, 2) for e in count if e != 0])
        indexes = sorted(list(range(len(prediction))), key=lambda i: scores[i], reverse=True)
        return indexes

    def _similarity_mutual_information(o, v):
        assert o.shape == v.shape
        m = max(o.max(), v.max()) + 1
        m = int(m)
        mat = torch.zeros(*(m, m, m, m))
        for i in range(m):
            for j in range(m):    
                mat[i, j, i, j] = 1
                
        t = torch.cat([o.unsqueeze(2), v.unsqueeze(2)], dim=2)
        
        e = mat[tuple(t.reshape(-1, 2).transpose(1, 0))]    
        counts = e.reshape((t.shape[0], t.shape[1], m, m)).transpose(2, 3).sum(1)
        counts /= o.shape[1]
        p_o = counts.sum(1)
        p_v = counts.sum(2)


        h_o = - (p_o * torch.log2(p_o)).nan_to_num().sum(1)
        h_v = - (p_v * torch.log2(p_v)).nan_to_num().sum(1)
        h_o_v = - (counts * torch.log2(counts)).nan_to_num().sum([1, 2])

        mutual_information = h_o + h_v - h_o_v
        m, _ = torch.min(torch.cat((h_o.unsqueeze(1), h_v.unsqueeze(1)), dim=1), 1)
        mutual_information /= m
        return (1 - mutual_information.clip(0, 1)).nan_to_num(np.inf)
    


