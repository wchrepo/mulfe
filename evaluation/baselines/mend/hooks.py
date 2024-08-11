from .utils import parent_module
import torch

def linear_backward_hook(mod, grad_in, grad_out):
    if not hasattr(mod, "weight"):
        print(f"{mod} has no weight!")
        return

    if hasattr(mod.weight, "__x__"):
        assert len(grad_out) == 1
        # mod.weight.__bgrad__ = grad_out[0].unsqueeze(-1) * mod.__x__[0].unsqueeze(-2)
        mod.weight.__delta__ = grad_out[0].reshape(-1,grad_out[0].size(-1)).clone().detach()
    else:
        print(f"{mod} has no __x__")


def linear_forward_hook(mod, activations, output):
    assert len(activations) == 1
    mod.weight.__x__ = activations[0].reshape(-1,activations[0].size(-1)).clone().detach()

def linear_backward_hook_concat(mod, grad_in, grad_out):
    if not hasattr(mod, "weight"):
        print(f"{mod} has no weight!")
        return

    if hasattr(mod.weight, "__x__"):
        assert len(grad_out) == 1
        # mod.weight.__bgrad__ = grad_out[0].unsqueeze(-1) * mod.__x__[0].unsqueeze(-2)
        if hasattr(mod.weight, "__delta__"):
            mod.weight.__delta__ = torch.cat([mod.weight.__delta__, grad_out[0].reshape(-1,grad_out[0].size(-1)).detach()], dim=0) 
        else:
            mod.weight.__delta__ = grad_out[0].reshape(-1,grad_out[0].size(-1)).detach()
    else:
        print(f"{mod} has no __x__")


def linear_forward_hook_concat(mod, activations, output):
    assert len(activations) == 1
    if hasattr(mod.weight, "__x__"):
        mod.weight.__x__ = torch.cat([mod.weight.__x__, activations[0].reshape(-1,activations[0].size(-1)).detach()], dim=0)
    else:
        mod.weight.__x__ = activations[0].reshape(-1,activations[0].size(-1)).detach()


def hook_model(model, pnames):
    handles = []
    for m in [parent_module(model, pname) for pname in pnames]:
        handles.append(m.register_full_backward_hook(linear_backward_hook))
        handles.append(m.register_forward_hook(linear_forward_hook))

    model.handles = handles

from contextlib import contextmanager
@contextmanager
def hook_model_context(model, pnames):
    handles = []
    for m in [parent_module(model, pname) for pname in pnames]:
        handles.append(m.register_full_backward_hook(linear_backward_hook_concat))
        handles.append(m.register_forward_hook(linear_forward_hook_concat))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
