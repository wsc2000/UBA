import torch
import numpy as np
from tqdm import tqdm

def flatten_grads(gradients):
    param_order = gradients[0].keys()
    flat_epochs = []
    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        for param in param_order:
            if param.split('.')[-1] != 'num_batches_tracked':
                try:
                    user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
                except:
                    user_arr.extend(
                        [grads[param].cpu().numpy().flatten().tolist()])
        flat_epochs.append(user_arr)
    flat_epochs = np.array(flat_epochs)

    return flat_epochs

# def flatten_grads(gradients):
#     flat_ep = []
#     for n_user in range(len(gradients)):
#         user_arr = []
#         grad = gradients[n_user]
#         for key in grad.keys():
#             user_arr.extend(grad[key].cpu().numpy().flatten().tolist())
#         flat_ep.append(user_arr)
#     return np.asarray(flat_ep)

def multi_krum(gradients, n_attackers, multi_k=True):

    grads = flatten_grads(gradients)

    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        scores = None
        for update in tqdm(remaining_updates):
            distance = []
            for update_ in remaining_updates:
                # distance.append(torch.norm((update - update_)) ** 2)
                distance.append(pow(np.linalg.norm(update - update_, ord=2),2))
            distance = torch.Tensor(distance).float()
            print(torch.sum(distance))
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=-1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=-1)
        print(scores)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break
    print(candidate_indices)
    return np.asarray(candidate_indices)

