import torch.nn.functional as F
import torch

def supervised_attention_kld(alpha_pred, alpha_tgt, eps=1e-8):
    return F.kl_div((alpha_pred+eps).log(), alpha_tgt, reduction="batchmean")

def focal_kl(alpha_pred, alpha_tgt, gamma=2.0, eps=1e-8):
    term = F.kl_div((alpha_pred+eps).log(), alpha_tgt, reduction="none")
    weight = (alpha_tgt.clamp(min=eps))**gamma
    return (weight * term).sum()


def bag_contrastive_ntxent(reps, labels, temperature=0.1, eps=1e-8):
    """Compute a simple NT-Xent-style contrastive loss between bag representations.

    reps: Tensor [B, D]
    labels: Tensor [B]
    """
    if reps is None or reps.size(0) < 2:
        return reps.new_tensor(0.0)
    reps = reps.view(reps.size(0), -1)
    device = reps.device
    labels = labels.view(-1)

    reps_norm = reps / (reps.norm(dim=1, keepdim=True) + eps)
    sim = torch.matmul(reps_norm, reps_norm.t())
    sim = sim / temperature


    logits_mask = (~torch.eye(sim.size(0), dtype=torch.bool, device=device)).float()


    loss = 0.0
    denom = (logits_mask * torch.exp(sim)).sum(dim=1)
    for i in range(sim.size(0)):
        pos_mask = (labels == labels[i]).float()
        pos_mask[i] = 0.0
        if pos_mask.sum() == 0:

            continue
        num = (pos_mask * torch.exp(sim[i])).sum()
        loss_i = -torch.log((num + eps) / (denom[i] + eps))
        loss = loss + loss_i
    loss = loss / reps.size(0)
    return loss
