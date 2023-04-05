import torch
from attack.surfree.utils import atleast_kdim



def get_starting_points(model, X, y, init=None):
    """
    if init is None, create an init. Otherwise, check if it's really adversarial
    """
    if init is None:
        init = (X + 0.5*torch.randn_like(X)).clip(0, 1)

    p = model(init).argmax(1)
    u = init - X
    u /= u.flatten(1).norm(dim=1).view(len(u), 1, 1, 1)
    step = 2
    step_i = 0
    noise_sigma = 0.1
    while any(p == y):
        if step_i <= 20:
            step_i += 1
            init = torch.where(
                atleast_kdim(p == y, len(X.shape)), 
                (X + step_i * step * u).clip(0, 1), 
                init)
        else:
            init = torch.where(
                atleast_kdim(p == y, len(X.shape)), 
                (X + noise_sigma*torch.randn_like(X)).clip(0, 1), 
                init)
            noise_sigma *= 1.5

        p = model(init).argmax(1)

    print("INIT", (X- init).flatten(1).norm(dim=1))
    return init