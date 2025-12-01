import torch.nn.functional as F

def supervised_attention_kld(alpha_pred, alpha_tgt, eps=1e-8):
    return F.kl_div((alpha_pred+eps).log(), alpha_tgt, reduction="batchmean")

def focal_kl(alpha_pred, alpha_tgt, gamma=2.0, eps=1e-8):
    term = F.kl_div((alpha_pred+eps).log(), alpha_tgt, reduction="none")
    weight = (alpha_tgt.clamp(min=eps))**gamma
    return (weight * term).sum()
