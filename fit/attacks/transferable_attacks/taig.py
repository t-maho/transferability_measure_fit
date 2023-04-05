import math
import torch
import numpy as np


class TAIG:
    def __init__(self, niters=40, epsilon=8, s_num=20, r_flag=True, batch_size=8):
        self.niters = niters
        self.s_num = s_num
        self.epsilon = epsilon/255
        self.r_flag = r_flag
        self.batch_size = batch_size
        self.name = "TAIG-{}_iter-epsilon={}-{}_s_num".format(niters, epsilon, s_num)

    def compute_ig(self, inputs, label_inputs, model):

        if isinstance(model, dict) and "preprocessing" in model:
            prepro = model["preprocessing"]
            model = model["model"]
        else:
            prepro= None

        if prepro is not None:
            inputs = prepro(inputs)

        inputs = inputs.cpu().detach().numpy()

        baseline = np.zeros(inputs.shape)
        scaled_inputs = [baseline + (float(i) / self.s_num) * (inputs - baseline) for i in
                        range(0, self.s_num + 1)]
        scaled_inputs = np.asarray(scaled_inputs)
        if self.r_flag==True:
            scaled_inputs = scaled_inputs + np.random.uniform(-self.epsilon,self.epsilon,scaled_inputs.shape)
        inputs = torch.from_numpy(scaled_inputs)

        n_batch = math.ceil(inputs.shape[0] / self.batch_size)
        grads = []
        for batch_i in range(n_batch):
            s = batch_i * self.batch_size
            e = s + self.batch_size
            inputs_batch = inputs[s:e].to(0, dtype=torch.float)
            inputs_batch.requires_grad_(True)
            att_out = model(inputs_batch)
            score = att_out[:, label_inputs]
            loss = -torch.mean(score)
            model.zero_grad()
            loss.backward(retain_graph=True)
            grads.append(inputs_batch.grad.data.detach())

        grads = torch.cat(grads, dim=0)
        avg_grads = torch.mean(grads, dim=0).cpu()
        delta_X = inputs[-1] - inputs[0]
        integrated_grad = delta_X * avg_grads
        #del integrated_grad,delta_X,avg_grads,grads,loss,score,att_out
        return integrated_grad.detach().cuda(0)

    def get_attack_name(self):
        return self.name

    def __call__(self, model, inputs, labels):
        img = inputs.clone()
        for i in range(self.niters):
            igs = []
            for im_i in range(list(img.shape)[0]):
                integrated_grad = self.compute_ig(
                    img[im_i], 
                    labels[im_i], model)
                igs.append(integrated_grad.unsqueeze(0))
            igs = torch.cat(igs, dim=0)
            
            if isinstance(model, dict) and "preprocessing" in model:
                model["model"].zero_grad()
            else:
                model.zero_grad()
            img = (img + 1./255 * torch.sign(igs)).float()
            img = torch.where(img > inputs + self.epsilon, inputs + self.epsilon, img)
            img = torch.where(img < inputs - self.epsilon, inputs - self.epsilon, img)
            img = torch.clamp(img, min=0, max=1)

        return img
