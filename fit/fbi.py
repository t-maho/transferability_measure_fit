import math
from operator import truth
import numpy as np
import torch

from collections import Counter

class FBI:
    def __init__(self, num_images: int = 100):
        self._num_images = num_images

    def __call__(self, sources: list, target: str, predictions: dict):
        p_sources = torch.Tensor(np.array([predictions[source] for source in sources])).long()
        images_indexes = self._select_images(p_sources)
        p_sources = p_sources[:, images_indexes[:self._num_images]]

        p_target = torch.Tensor(predictions[target]).long().unsqueeze(0).repeat_interleave(len(sources), 0)
        p_target = p_target[:, images_indexes[:self._num_images]]

        truths = predictions["truths"][images_indexes[:self._num_images]]
        truths = torch.Tensor(truths).long().unsqueeze(0).repeat_interleave(len(sources), 0)

        p_target = (p_target != truths).long()
        p_sources = (p_sources != truths).long()

        mi = self._similarity_mutual_information(p_sources, p_target).numpy()
        return mi

    def _select_images(self, prediction: np.ndarray):
        """Select images with high entropy.
        Args:
            prediction (np.ndarray): Prediction of the models. Shape: (num_images, n_models).
        Returns:
            indexes (list): Indexes of the selected images.
        """
        
        scores = []
        prediction = prediction.transpose(0, 1)
        for v in prediction:
            count = Counter([str(e) for e in v])
            count = [e / len(v) for e in count.values()]
            scores.append([- e * math.log(e, 2) for e in count if e != 0])
        indexes = sorted(list(range(len(prediction))), key=lambda i: scores[i], reverse=True)
        return indexes

    def _similarity_mutual_information(self, o, v):
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
        # return (1 - mutual_information.clip(0, 1)).nan_to_num(np.inf)
        return mutual_information.clip(0, 1).nan_to_num(np.inf)
    


