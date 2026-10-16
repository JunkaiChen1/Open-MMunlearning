"""Original gradient-norm MIA calculation and score conversion."""

from evals.metrics import register_metric
from evals.utils.inference import tokenwise_logprobs

from .base import Attack, unlearning_metric
from .utils import score_inputs


def gradient_norms(model, batch_log_probs, *, p):
    """Compute mean parameter-gradient norms; clears model gradients per sample."""
    import torch

    batch_loss = [-torch.mean(lps) for lps in batch_log_probs]
    values = []
    for index, sample_loss in enumerate(batch_loss):
        parameter_norms = []
        model.zero_grad()
        # Samples share one forward graph; release it after the final backward.
        sample_loss.backward(retain_graph=index < len(batch_loss) - 1)
        for param in model.parameters():
            if param.grad is not None:
                parameter_norms.append(param.grad.detach().norm(p=p))
        values.append(torch.stack(parameter_norms).mean())
    return values


def gradnorm_score(gradient_norm):
    import torch

    if not isinstance(gradient_norm, torch.Tensor):
        gradient_norm = torch.as_tensor(gradient_norm)
    return gradient_norm.cpu().to(torch.float32).numpy()


@register_metric
@unlearning_metric("gradnorm", required_inputs=("grad_norm",))
def gradnorm(model, *, pre_compute, access_key="generation", **kwargs):
    return score_inputs(
        pre_compute,
        lambda item: float(gradnorm_score(item["grad_norm"])),
        access_key=access_key,
        required_inputs=("grad_norm",),
    )


class GradNormAttack(Attack):
    def setup(self, p, **kwargs):
        if p not in [1, 2, float("inf")]:
            raise ValueError(f"Invalid p-norm value: {p}")
        self.p = p

    def compute_batch_values(self, batch):
        """Compute gradients of examples w.r.t model parameters. More grad norm => more loss."""
        self.model.train()
        batch_log_probs = tokenwise_logprobs(self.model, batch, grad=True)
        batch_grad_norms = gradient_norms(self.model, batch_log_probs, p=self.p)
        self.model.eval()
        return batch_grad_norms

    def compute_score(self, sample_stats):
        """Return the gradient norm with the original positive orientation."""
        return gradnorm_score(sample_stats)
