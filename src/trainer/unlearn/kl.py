from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_kl_divergence


class KL(GradDiff):
    def __init__(self, *args, **kwargs):
        kwargs.pop("retain_loss_type", None)
        super().__init__(*args, retain_loss_type="NLL", **kwargs)
        self.ref_model = self._prepare_ref_model(self.model)

    def compute_retain_loss(self, model, retain_inputs):
        kl_loss, _ = compute_kl_divergence(model, self.ref_model, retain_inputs)
        return kl_loss
