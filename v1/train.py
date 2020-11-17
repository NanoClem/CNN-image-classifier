import torch
from torch.autograd import Variable


# checking for cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def training(epoch: int, model, optimizer, criterion, 
            trainLoader, validLoader, train_losses: list, valid_losses: list) -> None:
    """
    """
    train_loss, valid_loss = 0, 0

    # training-the-model
    model.train()
    for data, target in trainLoader:

        # move-tensors-to-GPU 
        data   = data.to(device)
        target = target.to(device)
        model.to(device)
        
        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()                  

        output = model(data)                    # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        loss   = criterion(output, target)      # calculate-the-batch-loss

        loss.backward()                         # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        optimizer.step()                        # perform-a-ingle-optimization-step (parameter-update)
        
        # update-training-loss
        train_loss += loss.item() * data.size(0)        
        
    # validate-the-model
    model.eval()
    for data, target in validLoader:
        
        data   = data.to(device)
        target = target.to(device)
        model.to(device)
        
        output = model(data)
        loss   = criterion(output, target)
        
        # update-average-validation-loss 
        valid_loss += loss.item() * data.size(0)
    
    # calculate-average-losses
    train_loss = train_loss/len(trainLoader.sampler)
    valid_loss = valid_loss/len(validLoader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
        
    # print-training/validation-statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))