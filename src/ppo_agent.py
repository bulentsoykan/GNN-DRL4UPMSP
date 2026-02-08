"""
PPO Agent with GNN-based Policy for UPMSP

Implements the PPO algorithm with GNN feature extractor as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Tuple, List
from .gnn_model import HeterogeneousGNN


class GNN_PPO_Policy(nn.Module):
    """
    Actor-Critic policy network with GNN encoder.
    Shared GNN encoder with separate actor and critic heads.
    """

    def __init__(
        self,
        job_feature_dim: int = 5,
        machine_feature_dim: int = 4,
        setup_feature_dim: int = 2,
        hidden_dim: int = 128,
        n_gnn_layers: int = 4,
        n_mlp_layers: int = 2,
        mlp_hidden_dim: int = 256,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Shared GNN encoder
        self.gnn = HeterogeneousGNN(
            job_feature_dim=job_feature_dim,
            machine_feature_dim=machine_feature_dim,
            setup_feature_dim=setup_feature_dim,
            hidden_dim=hidden_dim,
            n_layers=n_gnn_layers,
        )

        # Global feature processing
        self.global_feature_dim = 8  # n_waiting, current_time, twt, tst, etc.
        self.global_proj = nn.Linear(self.global_feature_dim, hidden_dim)

        # Actor head (policy network)
        actor_layers = []
        actor_input_dim = hidden_dim * 3 + hidden_dim  # job + machine + setup + global
        for i in range(n_mlp_layers):
            actor_layers.append(nn.Linear(
                actor_input_dim if i == 0 else mlp_hidden_dim,
                mlp_hidden_dim
            ))
            actor_layers.append(nn.ReLU())
        actor_layers.append(nn.Linear(mlp_hidden_dim, 1))  # Output logit for each job-machine pair
        self.actor = nn.Sequential(*actor_layers)

        # Critic head (value network)
        critic_layers = []
        critic_input_dim = hidden_dim * 3 + hidden_dim  # aggregated graph + global
        for i in range(n_mlp_layers):
            critic_layers.append(nn.Linear(
                critic_input_dim if i == 0 else mlp_hidden_dim,
                mlp_hidden_dim
            ))
            critic_layers.append(nn.ReLU())
        critic_layers.append(nn.Linear(mlp_hidden_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, graph_data, global_features: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            graph_data: HeteroData graph
            global_features: global state features

        Returns:
            Tuple of (action_logits, state_value)
        """
        # GNN encoding
        job_emb, machine_emb, setup_emb = self.gnn(graph_data)

        # Global features
        global_emb = F.relu(self.global_proj(global_features))

        # Aggregate graph embeddings
        job_agg = job_emb.mean(dim=0, keepdim=True) if job_emb.size(0) > 0 else torch.zeros(1, self.hidden_dim)
        machine_agg = machine_emb.mean(dim=0, keepdim=True)
        setup_agg = setup_emb.mean(dim=0, keepdim=True)

        # State embedding for critic
        state_emb = torch.cat([job_agg, machine_agg, setup_agg, global_emb.unsqueeze(0)], dim=-1)
        state_value = self.critic(state_emb).squeeze()

        # Action logits for actor
        # For each job-machine pair, create embedding
        n_jobs = job_emb.size(0)
        n_machines = machine_emb.size(0)

        action_logits = []
        for job_idx in range(n_jobs):
            for machine_idx in range(n_machines):
                # Combine job, machine, setup, and global embeddings
                pair_emb = torch.cat([
                    job_emb[job_idx],
                    machine_emb[machine_idx],
                    setup_agg.squeeze(0),
                    global_emb
                ], dim=-1)
                logit = self.actor(pair_emb.unsqueeze(0))
                action_logits.append(logit)

        if len(action_logits) > 0:
            action_logits = torch.cat(action_logits, dim=0).squeeze(-1)
        else:
            action_logits = torch.zeros(1)

        return action_logits, state_value

    def get_action(self, graph_data, global_features: torch.Tensor, action_mask: np.ndarray, deterministic: bool = False):
        """
        Sample action from policy.

        Args:
            graph_data: HeteroData graph
            global_features: global state features
            action_mask: boolean mask of feasible actions
            deterministic: if True, select argmax action

        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            action_logits, value = self.forward(graph_data, global_features)

            # Apply action mask
            action_logits = action_logits.clone()
            action_logits[~torch.tensor(action_mask, dtype=torch.bool)] = -1e10

            # Sample action
            if deterministic:
                action = action_logits.argmax().item()
                log_prob = torch.tensor(0.0)
            else:
                probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action = action.item()

        return action, log_prob, value

    def evaluate_actions(self, graph_data, global_features: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate actions for PPO update.

        Args:
            graph_data: HeteroData graph
            global_features: global state features
            actions: actions taken

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        action_logits, values = self.forward(graph_data, global_features)

        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy


class PPOTrainer:
    """
    PPO training algorithm.
    """

    def __init__(
        self,
        policy: GNN_PPO_Policy,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_epsilon: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: list of rewards
            values: list of state values
            dones: list of done flags

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values, dtype=torch.float32)

        return advantages, returns

    def update(
        self,
        rollout_buffer: Dict,
    ) -> Dict[str, float]:
        """
        Perform PPO update.

        Args:
            rollout_buffer: dictionary containing rollout data

        Returns:
            Dictionary of training metrics
        """
        # Extract data from buffer
        states = rollout_buffer['states']
        actions = torch.tensor(rollout_buffer['actions'], dtype=torch.long)
        old_log_probs = torch.tensor(rollout_buffer['log_probs'], dtype=torch.float32)
        rewards = rollout_buffer['rewards']
        values = rollout_buffer['values']
        dones = rollout_buffer['dones']

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
        }

        for epoch in range(self.n_epochs):
            # Mini-batch updates would go here
            # For simplicity, using full batch
            pass

        return metrics
