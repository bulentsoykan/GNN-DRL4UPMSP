"""
UPMSP Instance Generator
Generates problem instances following the methodology from the paper.
"""

import numpy as np
from typing import Dict, List


class InstanceGenerator:
    """
    Generate UPMSP instances with parameters:
    - n_jobs: number of jobs
    - n_machines: number of machines
    - tau: due date tightness
    - R: due date range
    - beta: setup time ratio
    - delta: machine eligibility density
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        n_jobs: int = 20,
        n_machines: int = 5,
        tau: float = 0.4,
        R: float = 0.6,
        beta: float = 0.1,
        delta: float = 0.75,
        lambda_arrival: float = 0.5
    ) -> Dict:
        """
        Generate a UPMSP instance.

        Args:
            n_jobs: number of jobs
            n_machines: number of machines
            tau: due date tightness (0.2, 0.4, 0.6)
            R: due date range (0.2, 0.6, 1.0)
            beta: setup time ratio (0.1, 0.25)
            delta: eligibility density (0.75, 1.0)
            lambda_arrival: arrival intensity factor (0.5)

        Returns:
            Dictionary containing problem instance
        """

        # Processing times: DU(1, 100) - unrelated machines
        processing_times = self.rng.randint(1, 101, size=(n_jobs, n_machines)).astype(float)

        # Average processing time across all job-machine pairs
        p_bar = processing_times.mean()

        # Release dates: DU(0, lambda * n * p_bar / m)
        max_release = lambda_arrival * n_jobs * p_bar / n_machines
        release_dates = self.rng.uniform(0, max_release, size=n_jobs)

        # Job weights: DU(1, 10)
        weights = self.rng.randint(1, 11, size=n_jobs).astype(float)

        # Machine eligibility
        eligibility = np.zeros((n_jobs, n_machines), dtype=bool)
        n_eligible = max(1, int(np.ceil(delta * n_machines)))
        for j in range(n_jobs):
            eligible_machines = self.rng.choice(n_machines, size=n_eligible, replace=False)
            eligibility[j, eligible_machines] = True

        # Average processing time per job (across eligible machines)
        p_j_bar = np.zeros(n_jobs)
        for j in range(n_jobs):
            eligible_machines = np.where(eligibility[j])[0]
            p_j_bar[j] = processing_times[j, eligible_machines].mean()

        # Due dates: DU(r_j + p_j_bar(1-tau-R/2), r_j + p_j_bar(1-tau+R/2))
        due_dates = np.zeros(n_jobs)
        for j in range(n_jobs):
            lower = release_dates[j] + p_j_bar[j] * (1 - tau - R / 2)
            upper = release_dates[j] + p_j_bar[j] * (1 - tau + R / 2)
            due_dates[j] = self.rng.uniform(lower, upper)

        # Setup times: DU(0, beta * p_bar)
        # Shape: (n_jobs+1, n_jobs, n_machines)
        # First index is previous job (0 for idle state)
        max_setup = beta * p_bar
        setup_times = self.rng.uniform(0, max_setup, size=(n_jobs + 1, n_jobs, n_machines))

        # Set setup times to infinity for ineligible machines
        for j in range(n_jobs):
            for k in range(n_machines):
                if not eligibility[j, k]:
                    setup_times[:, j, k] = np.inf

        instance = {
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'processing_times': processing_times,
            'release_dates': release_dates,
            'due_dates': due_dates,
            'weights': weights,
            'setup_times': setup_times,
            'eligibility': eligibility,
            'parameters': {
                'tau': tau,
                'R': R,
                'beta': beta,
                'delta': delta,
                'lambda': lambda_arrival,
            }
        }

        return instance

    def generate_test_set(
        self,
        n_jobs: int = 20,
        n_machines: int = 5,
        n_instances: int = 50,
        tau_values: List[float] = [0.2, 0.4, 0.6],
        R_values: List[float] = [0.2, 0.6, 1.0],
        beta_values: List[float] = [0.1, 0.25],
        delta_values: List[float] = [0.75, 1.0]
    ) -> List[Dict]:
        """
        Generate a test set of instances with varying parameters.

        Args:
            n_jobs: number of jobs
            n_machines: number of machines
            n_instances: number of instances per parameter combination
            tau_values: list of due date tightness values
            R_values: list of due date range values
            beta_values: list of setup time ratios
            delta_values: list of eligibility densities

        Returns:
            List of problem instances
        """
        instances = []

        for tau in tau_values:
            for R in R_values:
                for beta in beta_values:
                    for delta in delta_values:
                        for _ in range(n_instances):
                            instance = self.generate(
                                n_jobs=n_jobs,
                                n_machines=n_machines,
                                tau=tau,
                                R=R,
                                beta=beta,
                                delta=delta
                            )
                            instances.append(instance)

        return instances

    def save_instance(self, instance: Dict, filepath: str):
        """Save instance to file."""
        np.savez(filepath, **instance)

    def load_instance(self, filepath: str) -> Dict:
        """Load instance from file."""
        data = np.load(filepath, allow_pickle=True)
        instance = {key: data[key] for key in data.files}
        # Convert arrays back to proper types
        if 'eligibility' in instance:
            instance['eligibility'] = instance['eligibility'].astype(bool)
        if 'parameters' in instance:
            instance['parameters'] = instance['parameters'].item()
        return instance
