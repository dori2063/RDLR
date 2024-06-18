import torch
import torch.nn.functional as F
'''
## Usage 
x_batch (batch, feature dimension) and y_batch (batch, classes) are given.
e.g., (32,1582) and (32,4).
##

y_pr_batch = model(x_batch)
loss = 0
criterion = RDloss.RDloss_(n_classes).cuda()
loss += criterion(y_pr_batch, y_tr_batch) 
loss += other_losses()
loss.backward()
optimizer.step()

'''
class RDloss_(torch.nn.Module):
    def __init__(self, nb_classes, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        self.nb_classes = nb_classes # 4
        self.mrg = mrg # 0
        self.alpha = alpha # 1
    def forward(self, X, Y):
        nonX = torch.softmax(X,-1)
        cos_tmp = F.linear(l2_norm(nonX), l2_norm(Y))[0]
        conf_exp = torch.exp(-cos_tmp) 
        sim_sum = (conf_exp).sum()/X.size(0)
        return torch.log(sim_sum)

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

# Proxy-Anchor and other metric losses are in the below link.
# https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/blob/3df643b36d62a46e7d90d68bd476521fba65e9d5/code/losses.py