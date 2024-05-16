import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch import Tensor
from torch_geometric.nn import global_add_pool
from typing import Tuple, Dict, List, Literal
from utils import compute_invariants_3d
from topomodelx.base.message_passing import MessagePassing
from topomodelx.base.conv import Conv
from topomodelx.base.aggregation import Aggregation


class enhancedEMPSN(nn.Module):
    """
    E(n) Equivariant Message Passing Simplicial Networks (EMPSN)
    """
    def __init__(self, num_input: int, num_hidden: int, num_out: int, num_layers: int, max_com: str, adjacencies: list) -> None: #adjacenies is a list of strings are still to do
        super().__init__()

        # layers
        self.feature_embedding = nn.Linear(num_input, num_hidden)

        self.layers = nn.ModuleList(
            [EMPSNLayer(adjacencies, self.max_dim, num_hidden) for _ in range(num_layers)]
        )

        self.pre_pool = nn.ModuleDict()
        for dim in range(self.max_dim+1):
            self.pre_pool[str(dim)] = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_hidden))
        self.post_pool = nn.Sequential(
            nn.Sequential(nn.Linear((max_dim + 1) * num_hidden, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_out))
        )

    def forward(self, New_graph: Data) -> Tensor: #different input from the original so it's shorter
        
        # message passing #up until here is fine
        for layer in self.layers:
            x = layer(x, adj, inv)

        # read out
        x = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}
        x = {dim: global_add_pool(x[dim], x_batch[dim]) for dim, feature in x.items()}
        state = torch.cat(tuple([feature for dim, feature in x.items()]), dim=1)
        out = self.post_pool(state) #classifier across 19 variables 
        out = torch.squeeze(out)

        return out



class EMPSNLayer(torch.nn.Module):
    """Simplicial Complex Convolutional Network (SCCN) layer by [1]_.

    This implementation applies to simplicial complexes of any rank.

    This layer corresponds to the leftmost tensor diagram labeled Yang22c in
    Figure 11 of [3]_.

    Parameters
    ----------
    channels : int
        Dimension of features on each simplicial cell.
    max_rank : int
        Maximum rank of the cells in the simplicial complex.
    aggr_func : {"mean", "sum"}, default="sum"
        The function to be used for aggregation.
    update_func : {"relu", "sigmoid", "tanh", None}, default="sigmoid"
        The activation function.

    See Also
    --------
    topomodelx.nn.simplicial.scn2_layer.SCN2Layer
        SCN layer proposed in [1]_ for simplicial complexes of rank 2.
        The difference between SCCN and SCN is that:
        - SCN passes messages between cells of the same rank,
        - SCCN passes messages between cells of the same ranks, one rank above
        and one rank below.

    References
    ----------
    .. [1] Yang, Sala, Bogdan.
        Efficient representation learning for higher-order data with
        simplicial complexes (2022).
        https://proceedings.mlr.press/v198/yang22a.html
    .. [2] Papillon, Sanborn, Hajij, Miolane.
        Equations of topological neural networks (2023).
        https://github.com/awesome-tnns/awesome-tnns/
    .. [3] Papillon, Sanborn, Hajij, Miolane.
        Architectures of topological deep learning: a survey on topological neural networks (2023).
        https://arxiv.org/abs/2304.10031.
    """

    def __init__(
        self,
        channels,
        max_rank,
        aggr_func: Literal["mean", "sum"] = "sum",
        update_func: Literal["relu", "sigmoid", "tanh"] | None = "sigmoid",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.max_rank = max_rank

        # # convolutions within the same rank
        # self.convs_same_rank = torch.nn.ModuleDict(
        #     {
        #         f"rank_{rank}": Conv(
        #             in_channels=channels,
        #             out_channels=channels,
        #             update_func=None,
        #         )
        #         for rank in range(max_rank + 1)
        #     }
        # )

        # convolutions from lower to higher rank
        self.convs_low_to_high = torch.nn.ModuleDict(
            {
                f"rank_{rank}": Conv(
                    in_channels=channels,
                    out_channels=channels,
                    update_func=None,
                )
                for rank in range(1, max_rank + 1)
            }
        )

        # # convolutions from higher to lower rank
        # self.convs_high_to_low = torch.nn.ModuleDict(
        #     {
        #         f"rank_{rank}": Conv(
        #             in_channels=channels,
        #             out_channels=channels,
        #             update_func=None,
        #         )
        #         for rank in range(max_rank)
        #     }
        # )

        # aggregation functions
        self.aggregations = torch.nn.ModuleDict(
            {
                f"rank_{rank}": Aggregation(
                    aggr_func=aggr_func, update_func=update_func
                )
                for rank in range(max_rank + 1)
            }
        )

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        # for rank in self.convs_same_rank:
        #     self.convs_same_rank[rank].reset_parameters()
        for rank in self.convs_low_to_high:
            self.convs_low_to_high[rank].reset_parameters()
        # for rank in self.convs_high_to_low:
        #     self.convs_high_to_low[rank].reset_parameters()

    def forward(self, features, incidences, adjacencies):
        r"""Forward pass.

        Parameters
        ----------
        features : dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_cells, channels)
            Input features on the cells of the simplicial complex.
        incidences : dict[int, torch.sparse], length=max_rank, shape = (n_rank_r_minus_1_cells, n_rank_r_cells)
            Incidence matrices :math:`B_r` mapping r-cells to (r-1)-cells.
        adjacencies : dict[int, torch.sparse], length=max_rank, shape = (n_rank_r_cells, n_rank_r_cells)
            Adjacency matrices :math:`H_r` mapping cells to cells via lower and upper cells.

        Returns
        -------
        dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_cells, channels)
            Output features on the cells of the simplicial complex.
        """
        out_features = {}
        for rank in range(self.max_rank + 1):
            list_to_be_aggregated = []
        #         self.convs_same_rank[f"rank_{rank}"](
        #             features[f"rank_{rank}"],
        #             adjacencies[f"rank_{rank}"],
        #         )
        #     ]
        #     if rank < self.max_rank:
        #         list_to_be_aggregated.append(
        #             self.convs_high_to_low[f"rank_{rank}"](
        #                 features[f"rank_{rank+1}"],
        #                 incidences[f"rank_{rank+1}"],
        #             )
        #         )
            if rank > 0:
                list_to_be_aggregated.append(
                    self.convs_low_to_high[f"rank_{rank}"](
                        features[f"rank_{rank-1}"],
                        incidences[f"rank_{rank}"].transpose(1, 0),
                    )
                )

            out_features[f"rank_{rank}"] = self.aggregations[f"rank_{rank}"](
                list_to_be_aggregated
            )

        return out_features