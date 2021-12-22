import torch


def notnone_collate_fn(batch):
    len_batch = len(batch)  # original batch length
    batch = list(filter(lambda x: x is not None, batch))  # filter out all the Nones
    if len_batch > len(
        batch
    ):  # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
        diff = len_batch - len(batch)
        for _ in range(diff):
            batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)
