
import torch

def calc_val_data(preds, masks, num_classes):
    preds = torch.argmax(preds, dim=1)

    # TODO: calc intersection for each class
    intersection = [torch.logical_and(preds == class_, masks == class_).sum(dim=[1, 2])[None] for class_ in range(num_classes)]
    intersection = torch.cat(intersection, dim=0).t()

    # TODO: calc union for each class
    union = [torch.logical_or(preds == class_, masks == class_).sum(dim=[1, 2])[None] for class_ in range(num_classes)]
    union = torch.cat(union, dim=0).t()

    # TODO: calc number of pixels in groundtruth mask per class
    target = [(masks == class_).sum(dim=[1, 2])[None] for class_ in range(num_classes)]
    target = torch.cat(target, dim=0).t()
    # Output shapes: B x num_classes

    assert isinstance(intersection, torch.Tensor), 'Output should be a tensor'
    assert isinstance(union, torch.Tensor), 'Output should be a tensor'
    assert isinstance(target, torch.Tensor), 'Output should be a tensor'

    assert intersection.shape == union.shape == target.shape, 'Wrong output shape'
    assert union.shape[0] == masks.shape[0] and union.shape[1] == num_classes, 'Wrong output shape'

    return intersection, union, target

def calc_val_loss(intersection, union, target, eps = 1e-7):
    # TODO: calc mean class iou
    mean_iou = (intersection.sum(dim=0) / (union.sum(dim=0) + eps)).mean()
    # TODO: calc mean class recall
    mean_class_rec = (intersection.sum(dim=0) / (target.sum(dim=0) + eps)).mean()
    # TODO: calc mean accuracy
    mean_acc = intersection.sum() / (target.sum() + eps) 
    return mean_iou, mean_class_rec, mean_acc