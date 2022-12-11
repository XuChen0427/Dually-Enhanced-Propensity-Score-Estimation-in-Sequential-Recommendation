# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.model.loss
#######################
Common Loss in recommender system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F




class IPSDualBCELoss(nn.Module):
    def __init__(self,clip, max_clip,total_steps,warmup_rate,alpha=0.5):
        super(IPSDualBCELoss, self).__init__()
        self.alpha = alpha
        self.min_clip = clip
        self.max_clip = max_clip

        self.warmup_rate = warmup_rate

        self.clip = max_clip
        self.total_steps = total_steps
        self.current_step = 0

        #self.Loss = nn.BCELoss(reduce=True)
        #return float(current_step) / float(max(1.0, num_warmup_steps))
    def forward(self,outputs, labels,pi_score,pu_score):
        pi_scores = torch.clamp(pi_score.detach(), self.clip, 1)
        pu_scores = torch.clamp(pu_score.detach(), self.clip, 1)
        IPS_weight = (1/pi_scores)*self.alpha+(1/pu_scores)*(1-self.alpha)

        loss = F.binary_cross_entropy(outputs, labels, weight=IPS_weight.squeeze(-1), reduction='mean')
        return loss

    def update(self):
        if self.current_step < self.total_steps * self.warmup_rate:
            self.clip = 1
        else:
            self.clip = self.min_clip

        self.current_step = self.current_step + 1

class BPRLoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class RegLoss(nn.Module):
    """ RegLoss, L2 regularization on model parameters

    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class EmbMarginLoss(nn.Module):
    """ EmbMarginLoss, regularization on embeddings
    """

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding ** self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss
