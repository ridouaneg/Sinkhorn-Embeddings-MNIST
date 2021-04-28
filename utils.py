import torch
from .loss import sinkhorn_divergence

def train_epoch(train_loader, model, criterion, optimizer, cuda):
    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Put data on gpu
        if cuda:
            data = tuple(d.cuda() for d in data)
            target = target.cuda()
        
        # Clear the gradients
        optimizer.zero_grad()

        # Compute model's output
        outputs = model(*data)

        # Compute loss
        loss_inputs = outputs + (target,)
        loss, _ = criterion(*loss_inputs)
        loss.cuda()

        # Update weights
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        total_loss += loss.item()

    total_loss /= (batch_idx + 1)
    return total_loss

def test_epoch(val_loader, model, criterion, cuda):
    with torch.no_grad():
        model.eval()
        val_loss, dist_pos, dist_neg = 0.0, 0.0, 0.0
        total_pos, total_neg = 0, 0
        for batch_idx, (data, target) in enumerate(val_loader):
            if cuda:
                data = tuple(d.cuda() for d in data)
                target = target.cuda()

            outputs = model(*data)

            loss_inputs = outputs + (target,)
            loss, distances = criterion(*loss_inputs)
            
            dist_pos += (target * distances).sum()
            dist_neg += ((1 - target) * distances).sum()
            total_pos += target.sum()
            total_neg += len(target) - target.sum()
            val_loss += loss
            
        val_loss /= len(val_loader)
        dist_pos /= total_pos
        dist_neg /= total_neg

    return val_loss.item(), dist_pos.item(), dist_neg.item()