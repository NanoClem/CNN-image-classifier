import tqdm

import torch

from config import device



def training(epoch: int, model, optimizer, criterion, 
            trainLoader, validLoader, train_losses: list, valid_losses: list) -> None:
    """
    """
    train_loss, valid_loss = 0, 0

    # training-the-model
    model.train()
    for data, target in trainLoader:

        # CPU to GPU
        data   = data.to(device)
        target = target.to(device)
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()                  
        
        output = model(data)                    # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        loss   = criterion(output, target)      # calculate-the-batch-loss

        loss.backward()                         # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        optimizer.step()                        # perform-a-ingle-optimization-step (parameter-update)
        
        # updating training loss
        train_loss += loss.item() * data.size(0)        
        
    # validate the model
    model.eval()
    for data, target in validLoader:
        data   = data.to(device)
        target = target.to(device)

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
    print('Epoch: {} Training Loss: {:.6f} Validation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))


def evaluate(model, evalLoader):
    """
    """
    model.eval()  # disabling dropout

    with torch.no_grad():   # clear all gradients
        correct = 0
        total = 0
        for images, labels in evalLoader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() #accuracy_score(labels, predicted, normalize=False)    # nb of correct classified samples
        
        return 100 * correct / total