import logging

logger = logging.getLogger(__name__)


def patch_zero3_none_grad_reduce():
    """Patch DeepSpeed ZeRO-3 to tolerate repeated checkpoint grad hooks.

    LoRA with frozen base parameters and reentrant gradient checkpointing can fire a
    ZeRO-3 gradient-reduction hook for a trainable parameter more than once in the
    same backward pass. Depending on timing, DeepSpeed can see either a missing
    gradient after the first reduction or a duplicate parameter in the active IPG
    bucket. The patch skips those repeat-hook calls before DeepSpeed buckets them.
    """
    try:
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
    except ImportError as exc:
        logger.warning("DeepSpeed ZeRO-3 patch requested but unavailable: %s", exc)
        return False

    patch_attr = "_open_unlearning_zero_reentrant_grad_patch_v2"
    if getattr(DeepSpeedZeroOptimizer_Stage3, patch_attr, False):
        return True

    reduce_ipg = (
        DeepSpeedZeroOptimizer_Stage3._DeepSpeedZeroOptimizer_Stage3__reduce_and_partition_ipg_grads
    )
    add_to_bucket = (
        DeepSpeedZeroOptimizer_Stage3._DeepSpeedZeroOptimizer_Stage3__add_grad_to_ipg_bucket
    )

    def has_param_in_ipg_bucket(self, param):
        param_ds_id = getattr(param, "ds_id", None)
        for bucket_param in self.params_in_ipg_bucket:
            bucket_ds_id = getattr(bucket_param, "ds_id", None)
            if param_ds_id is not None and bucket_ds_id is not None:
                if param_ds_id == bucket_ds_id:
                    return True
            elif bucket_param is param:
                return True
        return False

    def patched_reduce_independent_p_g_buckets_and_remove_grads(self, param):
        if param.grad is None:
            return

        if has_param_in_ipg_bucket(self, param):
            return

        if (
            self.elements_in_ipg_bucket + param.ds_numel > self.reduce_bucket_size
            and self.elements_in_ipg_bucket > 0
        ):
            self.report_ipg_memory_usage(
                "In ipg_remove_grads before reduce_ipg_grads", param.ds_numel
            )
            reduce_ipg(self)

        if param.grad is None:
            return

        if has_param_in_ipg_bucket(self, param):
            return

        add_to_bucket(self, param)

    DeepSpeedZeroOptimizer_Stage3.reduce_independent_p_g_buckets_and_remove_grads = (
        patched_reduce_independent_p_g_buckets_and_remove_grads
    )
    setattr(DeepSpeedZeroOptimizer_Stage3, patch_attr, True)
    logger.info("Patched DeepSpeed ZeRO-3 None-gradient reduction handling.")
    return True
