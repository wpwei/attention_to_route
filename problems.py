import torch


def TSP(graphs, permutations):
    """
    Author: wouterkool

    Copied from https://github.com/wouterkool/attention-learn-to-route
    """
    # Check that tours are valid, i.e. contain 0 to n -1
    assert (torch.arange(
        permutations.size(1), out=permutations.data.new()).view(
            1, -1).expand_as(permutations) == permutations.data.sort(1)[0]
            ).all(), "Invalid tour"

    # Gather dataset in order of tour
    d = graphs.gather(1, permutations.unsqueeze(-1).expand_as(graphs))

    # Length is distance (L2-norm of difference) from each next location from
    # its prev and of last from first
    return ((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) +
            (d[:, 0] - d[:, -1]).norm(p=2, dim=1)).view(-1, 1)
