"""
UPMSP Environment/Simulator
Implements the discrete-event simulation environment for the Unrelated Parallel Machine Scheduling Problem.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import copy


class UPMSPEnvironment(gym.Env):
    """
    Unrelated Parallel Machine Scheduling Problem Environment with:
    - Release dates
    - Sequence-dependent and machine-dependent setup times
    - Machine eligibility constraints
    - Multi-objective: minimize TWT and TST
    """

    def __init__(self, instance: Dict):
        """
        Args:
            instance: Problem instance dictionary containing:
                - n_jobs: number of jobs
                - n_machines: number of machines
                - processing_times: (n_jobs, n_machines) array
                - release_dates: (n_jobs,) array
                - due_dates: (n_jobs,) array
                - weights: (n_jobs,) array
                - setup_times: (n_jobs+1, n_jobs, n_machines) array
                - eligibility: (n_jobs, n_machines) boolean array
        """
        super().__init__()

        self.instance = instance
        self.n_jobs = instance['n_jobs']
        self.n_machines = instance['n_machines']
        self.processing_times = instance['processing_times']
        self.release_dates = instance['release_dates']
        self.due_dates = instance['due_dates']
        self.weights = instance['weights']
        self.setup_times = instance['setup_times']
        self.eligibility = instance['eligibility']

        # State variables
        self.current_time = 0.0
        self.completed_jobs = []
        self.remaining_jobs = list(range(self.n_jobs))
        self.machine_available_time = np.zeros(self.n_machines)
        self.machine_last_job = np.full(self.n_machines, -1, dtype=int)  # -1 means idle
        self.job_completion_times = np.full(self.n_jobs, -1.0)
        self.total_weighted_tardiness = 0.0
        self.total_setup_time = 0.0

        # Action space: select job-machine pair
        self.action_space = spaces.Discrete(self.n_jobs * self.n_machines)

        # Observation space (will be graph-based)
        self.observation_space = spaces.Dict({
            'current_time': spaces.Box(low=0, high=np.inf, shape=(1,)),
            'n_waiting_jobs': spaces.Box(low=0, high=self.n_jobs, shape=(1,)),
        })

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.current_time = 0.0
        self.completed_jobs = []
        self.remaining_jobs = list(range(self.n_jobs))
        self.machine_available_time = np.zeros(self.n_machines)
        self.machine_last_job = np.full(self.n_machines, -1, dtype=int)
        self.job_completion_times = np.full(self.n_jobs, -1.0)
        self.total_weighted_tardiness = 0.0
        self.total_setup_time = 0.0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int):
        """
        Execute action: assign job to machine.

        Args:
            action: index representing (job, machine) pair

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Decode action
        job_id = action // self.n_machines
        machine_id = action % self.n_machines

        # Check feasibility
        if not self._is_feasible(job_id, machine_id):
            # Invalid action - give negative reward
            reward = -1000.0
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, False, False, info

        # Calculate start time
        start_time = max(
            self.machine_available_time[machine_id],
            self.release_dates[job_id]
        )

        # Calculate setup time
        last_job = self.machine_last_job[machine_id]
        setup_time = self.setup_times[last_job + 1, job_id, machine_id]  # +1 because 0 is idle state

        # Update time
        start_time += setup_time
        completion_time = start_time + self.processing_times[job_id, machine_id]

        # Update state
        self.machine_available_time[machine_id] = completion_time
        self.machine_last_job[machine_id] = job_id
        self.job_completion_times[job_id] = completion_time
        self.remaining_jobs.remove(job_id)
        self.completed_jobs.append(job_id)

        # Update objectives
        tardiness = max(0, completion_time - self.due_dates[job_id])
        weighted_tardiness = self.weights[job_id] * tardiness
        self.total_weighted_tardiness += weighted_tardiness
        self.total_setup_time += setup_time

        # Calculate reward (negative of weighted sum of objectives)
        # Multi-objective reward: balance TWT and TST
        alpha = 0.5  # Weight for TWT
        beta = 0.5   # Weight for TST
        reward = -(alpha * weighted_tardiness + beta * setup_time)

        # Check if episode is done
        terminated = len(self.remaining_jobs) == 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _is_feasible(self, job_id: int, machine_id: int) -> bool:
        """Check if assigning job to machine is feasible."""
        if job_id not in self.remaining_jobs:
            return False
        if not self.eligibility[job_id, machine_id]:
            return False
        return True

    def get_action_mask(self) -> np.ndarray:
        """Return boolean mask of feasible actions."""
        mask = np.zeros(self.n_jobs * self.n_machines, dtype=bool)
        for job_id in self.remaining_jobs:
            for machine_id in range(self.n_machines):
                if self.eligibility[job_id, machine_id]:
                    action_id = job_id * self.n_machines + machine_id
                    mask[action_id] = True
        return mask

    def _get_observation(self) -> Dict:
        """
        Get current observation.
        For GNN-based agent, this will include graph structure.
        """
        observation = {
            'current_time': np.array([self.current_time]),
            'n_waiting_jobs': np.array([len(self.remaining_jobs)]),
            'machine_available_time': self.machine_available_time.copy(),
            'machine_last_job': self.machine_last_job.copy(),
            'remaining_jobs': self.remaining_jobs.copy(),
            'completed_jobs': self.completed_jobs.copy(),
            'twt_so_far': np.array([self.total_weighted_tardiness]),
            'tst_so_far': np.array([self.total_setup_time]),
        }
        return observation

    def _get_info(self) -> Dict:
        """Get additional info."""
        info = {
            'total_weighted_tardiness': self.total_weighted_tardiness,
            'total_setup_time': self.total_setup_time,
            'n_completed_jobs': len(self.completed_jobs),
            'n_remaining_jobs': len(self.remaining_jobs),
        }
        return info

    def get_schedule(self) -> Dict:
        """Get the complete schedule."""
        schedule = {
            'job_completion_times': self.job_completion_times,
            'machine_schedules': self._build_machine_schedules(),
            'total_weighted_tardiness': self.total_weighted_tardiness,
            'total_setup_time': self.total_setup_time,
        }
        return schedule

    def _build_machine_schedules(self) -> List[List[int]]:
        """Build schedule for each machine."""
        schedules = [[] for _ in range(self.n_machines)]
        # This is a simplified version - would need more details for full schedule
        return schedules
