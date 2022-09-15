import torch.nn.functional as F
import dgl
import torch





def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss



def assign_neighborhoods(graph) -> None:
    # assign neighborhoods for nodes in batched graphs, return {'0':[n1, n2], ...}
    batch = torch.repeat_interleave(graph.batch_num_nodes())

    neighbors = {}
    for node in graph.nodes():
        out_edges = graph.out_edges(node)[1]
        if out_edges.shape[0] > 0:
            neighbors[int(node)] = out_edges

    n_neighborhoods = len(neighbors)
    
    # mask for neighbors, not all nodes have 4 neighbors
    # 4 works for qm9, may cause bug for drug dataset
    neighbor_masks = torch.zeros([n_neighborhoods, 4])

    # # maps node index to hidden index as given by self.neighbors
    # x_to_h_map = torch.zeros(x.size(0))

    # map neighborhood to batch molecule
    neighborhood_to_mol_map = torch.zeros(n_neighborhoods, dtype=torch.int64)

    for i, (a, n) in enumerate(neighbors.items()):
        # x_to_h_map[a] = i
        neighbor_masks[i, 0:len(n)] = 1
        neighborhood_to_mol_map[i] = batch[a]

    return neighbors, neighbor_masks


def ground_truth_local_stats(pos, neighbors, neighbor_mask):
    n_neighborhoods = len(neighbors)
    local_coords = torch.zeros(n_neighborhoods, 4, 3)
    for i, (a, n) in enumerate(neighbors.items()):
        local_coords[i, 0:len(n)] = pos[n] - pos[a]
    
    bond_lengths, bond_angles = batch_local_stats_from_coords(local_coords, neighbor_mask)
    return bond_lengths, bond_angles


def batch_local_stats_from_coords(local_coords, neighbor_mask):
    one_hop_ds = torch.linalg.norm(torch.zeros_like(local_coords[0]).unsqueeze(0) - local_coords, dim=-1)
    angles = batch_angles_from_coords(local_coords, neighbor_mask)
    return one_hop_ds, angles


def batch_angles_from_coords(coords, mask):
    angle_mask_ref = torch.tensor([[0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0],
                                   [1, 1, 1, 0, 0, 0],
                                   [1, 1, 1, 1, 1, 1]])

    angle_combos = torch.tensor([[0, 1],
                                 [0, 2],
                                 [1, 2],
                                 [0, 3],
                                 [1, 3],
                                 [2, 3]])
    all_possible_combos = coords[:, angle_combos]
    v_a, v_b = all_possible_combos.split(1, dim=2)
    angle_mask = angle_mask_ref[mask.sum(dim=1).long()]
    angles = batch_angle_between_vectors(v_a.squeeze(2), v_b.squeeze(2)) * angle_mask
    return angles


def batch_angle_between_vectors(a, b):
    """
    Compute angle between two batches of input vectors
    """
    inner_product = (a * b).sum(dim=-1)

    # norms
    a_norm = torch.linalg.norm(a, dim=-1)
    b_norm = torch.linalg.norm(b, dim=-1)

    # protect denominator during division
    den = a_norm * b_norm + 1e-10
    cos = inner_product / den

    return cos


def von_Mises_loss(a, b, a_sin=None, b_sin=None):
    """
    :param a: cos of first angle
    :param b: cos of second angle
    :return: difference of cosines
    """
    if torch.is_tensor(a_sin):
        out = a * b + a_sin * b_sin
    else:
        out = a * b + torch.sqrt(1-a**2 + 1e-5) * torch.sqrt(1-b**2 + 1e-5)
    return out