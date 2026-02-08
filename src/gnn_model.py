"""
Heterogeneous Graph Neural Network for UPMSP State Representation

Implements the GNN architecture described in the paper with:
- Job nodes
- Machine nodes
- Setup nodes
- Multiple edge types (job-machine, machine-setup, job-setup, setup-machine)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.data import HeteroData
import numpy as np
from typing import Dict, Tuple


class HeterogeneousGNN(nn.Module):
    """
    Heterogeneous GNN for UPMSP state representation.
    Uses GATv2 message passing with multiple layers.
    """

    def __init__(
        self,
        job_feature_dim: int = 5,
        machine_feature_dim: int = 4,
        setup_feature_dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input projections for different node types
        self.job_input_proj = nn.Linear(job_feature_dim, hidden_dim)
        self.machine_input_proj = nn.Linear(machine_feature_dim, hidden_dim)
        self.setup_input_proj = nn.Linear(setup_feature_dim, hidden_dim)

        # Heterogeneous graph convolution layers
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv = HeteroConv({
                ('job', 'can_process', 'machine'): GATv2Conv(
                    hidden_dim, hidden_dim // n_heads, heads=n_heads, edge_dim=3, dropout=dropout
                ),
                ('machine', 'rev_can_process', 'job'): GATv2Conv(
                    hidden_dim, hidden_dim // n_heads, heads=n_heads, edge_dim=3, dropout=dropout
                ),
                ('machine', 'has_setup', 'setup'): GATv2Conv(
                    hidden_dim, hidden_dim // n_heads, heads=n_heads, edge_dim=1, dropout=dropout
                ),
                ('setup', 'rev_has_setup', 'machine'): GATv2Conv(
                    hidden_dim, hidden_dim // n_heads, heads=n_heads, edge_dim=1, dropout=dropout
                ),
                ('job', 'requires_setup', 'setup'): GATv2Conv(
                    hidden_dim, hidden_dim // n_heads, heads=n_heads, edge_dim=1, dropout=dropout
                ),
                ('setup', 'rev_requires_setup', 'job'): GATv2Conv(
                    hidden_dim, hidden_dim // n_heads, heads=n_heads, edge_dim=1, dropout=dropout
                ),
            }, aggr='mean')
            self.convs.append(conv)

        # Normalization layers
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                'job': nn.LayerNorm(hidden_dim),
                'machine': nn.LayerNorm(hidden_dim),
                'setup': nn.LayerNorm(hidden_dim),
            }) for _ in range(n_layers)
        ])

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the heterogeneous GNN.

        Args:
            data: HeteroData object containing node features and edge indices

        Returns:
            Tuple of (job_embeddings, machine_embeddings, setup_embeddings)
        """
        # Initial projections
        x_dict = {
            'job': F.relu(self.job_input_proj(data['job'].x)),
            'machine': F.relu(self.machine_input_proj(data['machine'].x)),
            'setup': F.relu(self.setup_input_proj(data['setup'].x)),
        }

        # Message passing
        for i, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, data.edge_index_dict, data.edge_attr_dict)

            # Add residual connections and normalization
            for node_type in x_dict.keys():
                if node_type in x_dict_new:
                    x_dict[node_type] = self.norms[i][node_type](
                        x_dict[node_type] + F.relu(x_dict_new[node_type])
                    )

        return x_dict['job'], x_dict['machine'], x_dict['setup']


class StateEncoder:
    """
    Converts UPMSP environment state to heterogeneous graph representation.
    """

    def __init__(self, instance: Dict):
        self.instance = instance
        self.n_jobs = instance['n_jobs']
        self.n_machines = instance['n_machines']
        # Simplified: use fixed number of setup types (could be based on job families)
        self.n_setup_types = min(10, self.n_jobs)

    def encode(self, observation: Dict, current_time: float) -> HeteroData:
        """
        Encode environment observation as heterogeneous graph.

        Args:
            observation: observation from environment
            current_time: current simulation time

        Returns:
            HeteroData graph
        """
        data = HeteroData()

        remaining_jobs = observation['remaining_jobs']
        n_remaining = len(remaining_jobs)

        if n_remaining == 0:
            # Create dummy graph when no jobs remain
            return self._create_dummy_graph()

        # Job node features
        job_features = []
        for job_id in remaining_jobs:
            features = [
                self.instance['weights'][job_id],
                self.instance['release_dates'][job_id],
                self.instance['due_dates'][job_id],
                current_time,
                self.instance['processing_times'][job_id].mean(),  # avg processing time
            ]
            job_features.append(features)
        data['job'].x = torch.tensor(job_features, dtype=torch.float)

        # Machine node features
        machine_features = []
        for machine_id in range(self.n_machines):
            features = [
                observation['machine_available_time'][machine_id],
                float(observation['machine_last_job'][machine_id]),
                current_time,
                observation['machine_available_time'][machine_id] - current_time,  # time until available
            ]
            machine_features.append(features)
        data['machine'].x = torch.tensor(machine_features, dtype=torch.float)

        # Setup node features (simplified)
        setup_features = []
        for setup_id in range(self.n_setup_types):
            features = [
                float(setup_id),
                0.0,  # placeholder
            ]
            setup_features.append(features)
        data['setup'].x = torch.tensor(setup_features, dtype=torch.float)

        # Build edges
        self._build_edges(data, remaining_jobs, observation)

        return data

    def _build_edges(self, data: HeteroData, remaining_jobs: list, observation: Dict):
        """Build heterogeneous edges."""

        # Job-Machine edges (eligibility)
        job_machine_edges = []
        job_machine_attrs = []

        for i, job_id in enumerate(remaining_jobs):
            for machine_id in range(self.n_machines):
                if self.instance['eligibility'][job_id, machine_id]:
                    job_machine_edges.append([i, machine_id])

                    # Edge features: processing time, setup time, eligibility
                    last_job = observation['machine_last_job'][machine_id]
                    setup_time = self.instance['setup_times'][last_job + 1, job_id, machine_id]
                    attrs = [
                        self.instance['processing_times'][job_id, machine_id],
                        setup_time if not np.isinf(setup_time) else 0.0,
                        1.0,  # eligible
                    ]
                    job_machine_attrs.append(attrs)

        if len(job_machine_edges) > 0:
            data['job', 'can_process', 'machine'].edge_index = torch.tensor(
                job_machine_edges, dtype=torch.long
            ).t().contiguous()
            data['job', 'can_process', 'machine'].edge_attr = torch.tensor(
                job_machine_attrs, dtype=torch.float
            )

            # Reverse edges
            data['machine', 'rev_can_process', 'job'].edge_index = data[
                'job', 'can_process', 'machine'
            ].edge_index.flip(0)
            data['machine', 'rev_can_process', 'job'].edge_attr = data[
                'job', 'can_process', 'machine'
            ].edge_attr

        # Machine-Setup edges (current setup state)
        machine_setup_edges = []
        machine_setup_attrs = []

        for machine_id in range(self.n_machines):
            last_job = observation['machine_last_job'][machine_id]
            setup_id = (last_job % self.n_setup_types) if last_job >= 0 else 0
            machine_setup_edges.append([machine_id, setup_id])
            machine_setup_attrs.append([current_time])

        data['machine', 'has_setup', 'setup'].edge_index = torch.tensor(
            machine_setup_edges, dtype=torch.long
        ).t().contiguous()
        data['machine', 'has_setup', 'setup'].edge_attr = torch.tensor(
            machine_setup_attrs, dtype=torch.float
        )

        # Reverse edges
        data['setup', 'rev_has_setup', 'machine'].edge_index = data[
            'machine', 'has_setup', 'setup'
        ].edge_index.flip(0)
        data['setup', 'rev_has_setup', 'machine'].edge_attr = data[
            'machine', 'has_setup', 'setup'
        ].edge_attr

        # Job-Setup edges (required setup)
        job_setup_edges = []
        job_setup_attrs = []

        for i, job_id in enumerate(remaining_jobs):
            setup_id = job_id % self.n_setup_types
            job_setup_edges.append([i, setup_id])
            job_setup_attrs.append([self.instance['weights'][job_id]])

        data['job', 'requires_setup', 'setup'].edge_index = torch.tensor(
            job_setup_edges, dtype=torch.long
        ).t().contiguous()
        data['job', 'requires_setup', 'setup'].edge_attr = torch.tensor(
            job_setup_attrs, dtype=torch.float
        )

        # Reverse edges
        data['setup', 'rev_requires_setup', 'job'].edge_index = data[
            'job', 'requires_setup', 'setup'
        ].edge_index.flip(0)
        data['setup', 'rev_requires_setup', 'job'].edge_attr = data[
            'job', 'requires_setup', 'setup'
        ].edge_attr

    def _create_dummy_graph(self) -> HeteroData:
        """Create dummy graph when no jobs remain."""
        data = HeteroData()
        data['job'].x = torch.zeros((1, 5), dtype=torch.float)
        data['machine'].x = torch.zeros((self.n_machines, 4), dtype=torch.float)
        data['setup'].x = torch.zeros((self.n_setup_types, 2), dtype=torch.float)
        return data
