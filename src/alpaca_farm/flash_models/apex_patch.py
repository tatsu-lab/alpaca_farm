from .. import logging

logger = logging.get_logger(__name__)

try:
    import apex

    apex_is_installed = True
    logger.warning("`apex` is installed. Using fused operators.")
except ImportError as e:
    apex_is_installed = False
    logger.warning("`apex` is not installed. Reverting to non-fused operators.")


def apex_layernorm(ln_module, input_):
    if apex_is_installed:
        return apex.normalization.fused_layer_norm.FusedLayerNormAffineFunction.apply(
            input_, ln_module.weight, ln_module.bias, ln_module.normalized_shape, ln_module.eps
        )
    else:
        return ln_module(input_)


def apex_rmsnorm(ln_module, input_):
    if apex_is_installed:
        return apex.normalization.fused_layer_norm.FusedRMSNormAffineFunction.apply(
            input_, ln_module.weight, ln_module.weight.size(), ln_module.variance_epsilon
        )
    else:
        return ln_module(input_)
