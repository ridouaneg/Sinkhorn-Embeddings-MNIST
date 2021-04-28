import torch
from torch.nn import Module
import torch.nn.functional as F

def euclidean_distance(output1, output2):
    return distances = (output2 - output1).pow(2).sum(1)

def sinkhorn_divergence(output1, output2, M=1, k=2, n_iter=20, lambd=0.05, p=2):
    bs = output1.size(0)
    x, y = output1, output2
    x, y = x.view((bs, M, k)), y.view((bs, M, k))

    x_col = x.unsqueeze(2)
    y_lin = y.unsqueeze(1)
    Dp = torch.sum((torch.abs(x_col - y_lin)) ** p, 3)

    K = torch.exp(-Dp / lambd)
    c, u, v  = 1/M * torch.ones((bs, M)).cuda(), 1/M * torch.ones((bs, M)).cuda(), 1/M * torch.ones((bs, M)).cuda()
    for _ in range(n_iter):
        r = u / torch.matmul(K, c.unsqueeze(2)).squeeze(2)
        c = v / torch.matmul(torch.transpose(K, 1, 2), r.unsqueeze(2)).squeeze(2)
    diag_r, diag_c = torch.diag_embed(r), torch.diag_embed(c)
    transport = torch.matmul(torch.matmul(diag_r, K), diag_c)
    distances = torch.diagonal(torch.matmul(torch.transpose(Dp, 1, 2), transport), dim1=-2, dim2=-1).sum(-1)
    return distances

class ContrastiveLoss(Module):
    def __init__(self, margin, M, k, lambd=0.05, distance='euclidean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

        self.M, self.k = M, k
        self.lambd = lambd
        
        self.distance = distance

    def forward(self, output1, output2, target):
    
        # Contrastive loss with Euclidean distance
        if self.distance == 'euclidean':
            distances = euclidean_distance(output1, output2)
            losses = 0.5 * (target.float() * distances + (1 - target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        
        # Contrastive loss with Sinkhorn divergence
        elif self.distance == 'sinkhorn':
            distances = sinkhorn_divergence(output1, output2, M=self.M, k=self.k, lambd=self.lambd)
            losses = 0.5 * (target.float() * distances ** 2 + (1 - target.float()) * F.relu(self.margin - distances) ** 2)
        
        return losses.mean(), distances