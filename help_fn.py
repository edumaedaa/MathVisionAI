import torch


"""
  Function created for calculate accuracy of model
"""
def accuracy_fn(y_true, y_preds):
    correct = torch.eq(y_true, y_preds).sum().item()
    acc = (correct/len(y_true))*100
    return acc