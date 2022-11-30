import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def threshold_criticality(softmax, threshold):
    """
    Sets automatically the criticality values to 1 if the treshold is reached,
    otherwise take the argmax.

    :param softmax: Tensor of shape (1, num_classes, width, height).
    :param threshold: Threshold that has to be reached to assign criticality 1.

    :return: Tensor of shape (width, height) containing the class labels.
    """
    mask = (softmax[0][0] > threshold).float()
    softmax[0][0] = mask + softmax[0][0]
    softmax[0][0] = torch.clamp(softmax[0][0], 0.0, 1.0)

    # Set all other points in other criticalities where threhold
    # was not reached to 0, so softmax will sum up to 1.0 again.
    for i in range(1, softmax.shape[1]):
        softmax[0][i] *= (1-mask)

    return softmax[0].argmax(dim=0)

def threshold_percentile(softmax, percentile, sort_softmax=False, ascending=True):
    """
    Calculates the cummulative sum of softmax from left to right and
    assigns a class as soon as the percentile value is reached.

    :param softmax: Tensor of shape (1, num_classes, width, height).
    :param percentile: Percentile that has to be reached to assign a class.
    :param: sort_softmax: Sort the softmax values descending before summing up.
    :param ascending: Sum up ascending by crititcality if True, otherwise descending.

    :return: Tensor of shape (width, height) containing the class labels.
    """
    unsorted_softmax = softmax[0]
    num_crits = unsorted_softmax.shape[0]
    proj_argmax = torch.zeros(unsorted_softmax.shape[1:], dtype=torch.long, device=device)
    
    """ Cell wise approach. Slow!
    for row in range(proj_argmax.shape[0]):
    for width in range(proj_argmax.shape[1]):
        cell = proj_output[0, :, row, width]

        sorted_cell = torch.sort(cell, dim=0, descending=True)
        values = sorted_cell.values
        indices = sorted_cell.indices
        
        sum=0.0
        for i in range(num_crits):
        if values[i].item() + sum >= self.percentile_threshold:
            proj_argmax[row, width] = indices[i]
            break
        else:
            sum += values[i]
        pass
    """

    """ Batch approach
    """
    if sort_softmax:
        sorted_softmax_result = torch.sort(unsorted_softmax, dim=0, descending=True)
        values = sorted_softmax_result.values
        indices = sorted_softmax_result.indices
    else:
        values = unsorted_softmax
        indices = torch.zeros(values.shape, dtype=torch.long, device=device)

        for i in range(1,num_crits):
            indices[i,:] += i   
    todo = torch.ones(proj_argmax.shape, dtype=torch.bool, device=device)
    sum  = torch.zeros(proj_argmax.shape, device=device)

    order = range(num_crits)
    if not ascending:
        order = reversed(order)

    for i in order:
        sum += values[i,:]
        mask = (sum >= percentile)
        mask = torch.logical_and(mask, todo) # Only set criticality values once
        proj_argmax[mask] = indices[i][mask]

        todo = torch.logical_xor(todo, mask) # Set todo to false if a values was set

    # Sanity check: All todo's should be False by now:
    if not torch.all(torch.logical_not(todo)):
        raise ValueError("Not all criticalities where assigned!")

    return proj_argmax