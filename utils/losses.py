import torch
import torch.nn as nn
from torch.nn import functional as F


class KLD(nn.Module):
    def forward(self, targets, inputs):
        batch_size = targets.shape[0]
        targets = F.softmax(targets, dim=1)
        inputs = F.log_softmax(inputs, dim=1)

        return torch.sum(F.kl_div(inputs, targets, reduction='none'))/batch_size


def uncertainity_loss(student_logits, teacher_logits):

    with torch.no_grad():
        student_probs = F.softmax(student_logits, dim=-1)
        student_entropy = - \
            torch.sum(student_probs * torch.log(student_probs + 1e-6), dim=1)
        instance_weight = student_entropy / \
            torch.log(torch.ones_like(student_entropy)
                      * student_logits.size(1))

    input = F.log_softmax(student_logits, dim=-1)
    target = F.softmax(teacher_logits, dim=-1)
    batch_loss = F.kl_div(
        input, target, reduction="none").sum(-1)
    updated_kld = torch.mean(batch_loss * instance_weight)

    return updated_kld
