import numpy as np
from regex import P

class TransQ:
    def __init__(self, method="1") -> None:
        self.method = str(method)

    def __call__(self, tf_norms: np.ndarray, wb_norms: np.ndarray=None, bb_norms: np.ndarray=None):

        if self.method == "2" and (wb_norms is None or bb_norms is None):
            raise ValueError("wb_norms and bb_norms must be provided when method is 2")

        n_models = tf_norms.shape[0]
        n_images = tf_norms.shape[2]
        transq_matrix = np.zeros((n_models, n_images))
        for m_i in range(n_models):
            for image_i in range(n_images):
                tf_norms_i = tf_norms[m_i, :, image_i]
                wb_norms_i = wb_norms[:, image_i] if wb_norms is not None else None
                bb_norms_i = bb_norms[:, image_i] if bb_norms is not None else None

                transq_matrix[m_i, image_i] = self._single_image_call(tf_norms_i, wb_norms_i, bb_norms_i) 

        return transq_matrix


    def _single_image_call(self, tf_norms, wb_norms=None, bb_norms=None):
        """
        Args:
            tf_norms (np.ndarray): Minimum perturbation norm in the transferable direction obtained on a source model. Shape n_models
            wb_norms (np.ndarray): Minimum perturbation norm with white-box attacks. Shape n_models
            bb_norms (np.ndarray): Minimum perturbation norm with black-box attacks. Shape n_models
        Returns:
            transq (float): TransQ score for a single image and a single model
        """

        if self.method == "2":
            assert tf_norms.shape == wb_norms.shape == bb_norms.shape

        if self.method.startswith("ASR"):
            threshold = float(self.method.split("_")[1])
            return (tf_norms > threshold).sum() / tf_norms.shape[0]
        elif self.method == "1":
            return np.mean(tf_norms)
        elif self.method == "2":
            pass
        else:
            raise NotImplementedError