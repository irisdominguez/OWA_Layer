from fastai.vision import *
from fastai.metrics import error_rate
import sys
import copy

###############################################
# Order metrics

def total_variation_image_batch(images):
    # Input: 4D tensor {batch, channel, rows, cols}
    # Output: 2D tensor {batch, channel} of total variations
    loss = torch.mean(torch.abs(images[:, :, :, :-1] - images[:, :, :, 1:]), dim=[2, 3]) + \
        torch.mean(torch.abs(images[:, :, :-1, :] - images[:, :, 1:, :]), dim=[2, 3])
    return loss

def entropy_image_batch(images):
    image_view = images.view(images.shape[0], images.shape[1], -1)
    b = F.softmax(image_view, dim=2) * F.log_softmax(image_view, dim=2)
    b = -1.0 * b.sum(dim=2)
    return b

def entropy_shannon_1diff_image_batch(images):
    horizontal = images[:, :, 1:, :] - images[:, :, :-1, :]
    vertical = images[:, :, :, 1:] - images[:, :, :, :-1]
    horizontal = horizontal.view(images.shape[0], images.shape[1], -1)
    vertical = vertical.view(images.shape[0], images.shape[1], -1)
    diffs = torch.cat((horizontal, vertical), 2)
    b = F.softmax(diffs, dim=2) * F.log_softmax(diffs, dim=2)
    b = -1.0 * b.sum(dim=2)
    return b

def activation_sum_image_batch(images):
    loss = torch.sum(torch.abs(images[:, :, :, :]), dim=[2, 3])
    return loss

def max_activation_image_batch(images):
    loss = torch.max(torch.max(torch.abs(images), 3)[0], 2)[0]
    return loss

def median_activation_image_batch(images):
    loss = torch.median(torch.abs(images).view(images.shape[0], images.shape[1], -1), 2)[0]
    return loss

def random_image_batch(images):
    r = torch.rand(images.shape[:2])
    r = r.to(images.device)
    return r

def no_sorting_image_batch(images):
    r = torch.zeros(images.shape[:2])
    r[:, :] = torch.arange(images.shape[1])
    r = r.to(images.device)
    return r

# Order metrics
###############################################


###############################################
# Sorting functions

def reorder_images_by_rank(images, rank):
    # Input: 4D tensor {batch, channel, rows, cols}
    # Rank: 2D tensor {batch, channel}
    # Output: sorted 4D images tensor
    x = rank.argsort(dim=1)
    y = torch.arange(0, x.shape[0] * x.shape[1], x.shape[1], dtype=x.dtype, device=images.device).view(x.shape[0], -1).expand(-1, x.shape[1])
    im1 = images.view(images.shape[0] * images.shape[1], images.shape[2], images.shape[3])
    im2 = im1.index_select(0, (x + y).view(-1))
    return im2.view(images.shape)

def replace_images_with_rank(images, rank):
    # Input: 4D tensor {batch, channel, rows, cols}
    # Rank: 2D tensor {batch, channel}
    # Output: sorted 4D images tensor
    images_sort = rank[:, :, None, None].expand(images.shape)
    return images_sort

# Sorting functions
###############################################