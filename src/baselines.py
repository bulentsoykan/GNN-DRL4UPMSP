"""
Baseline Methods for UPMSP

Implements:
1. ATCSR_Rm - Apparent Tardiness Cost with Setup and Ready times for unrelated machines
2. Genetic Algorithm - Multi-objective GA
"""

import numpy as np
from typing import Dict, List, Tuple
import copy


class ATCSR_Rm_Solver:
    """
    ATCSR_Rm dispatching rule for UPMSP.

    Apparent Tardiness Cost with Setup and Ready times for unrelated machines.
    Composite rule that considers tardiness, setup times, and release dates.
    """

    def __init__(self, k1: float = 2.0, k2: float = 2.0):
        """
        Args:
            k1: scaling parameter for setup times
            k2: scaling parameter for slack times
        """
        self.k1 = k1
        self.k2 = k2

    def solve(self, instance: Dict) -> Dict:
        """
        Solve UPMSP instance using ATCSR_Rm dispatching rule.

        Args:
            instance: problem instance

        Returns:
            Dictionary containing schedule and objectives
        """
        n_jobs = instance['n_jobs']
        n_machines = instance['n_machines']
        processing_times = instance['processing_times']
        release_dates = instance['release_dates']
        due_dates = instance['due_dates']
        weights = instance['weights']
        setup_times = instance['setup_times']
        eligibility = instance['eligibility']

        # State
        current_time = 0.0
        remaining_jobs = set(range(n_jobs))
        machine_available_time = np.zeros(n_machines)
        machine_last_job = np.full(n_machines, -1, dtype=int)
        job_completion_times = np.full(n_jobs, -1.0)
        schedule = {m: [] for m in range(n_machines)}

        # Average processing time for normalization
        p_bar = processing_times[eligibility].mean()

        total_setup_time = 0.0

        while remaining_jobs:
            # Find available machines
            min_available_time = machine_available_time.min()
            current_time = min_available_time

            # Calculate priority for each job-machine pair
            best_priority = -np.inf
            best_job = None
            best_machine = None

            for job_id in remaining_jobs:
                # Check if job is released
                if release_dates[job_id] > current_time + 1e-6:
                    continue

                for machine_id in range(n_machines):
                    if not eligibility[job_id, machine_id]:
                        continue

                    if machine_available_time[machine_id] > current_time + 1e-6:
                        continue

                    # Calculate ATCSR priority
                    priority = self._calculate_priority(
                        job_id, machine_id, current_time,
                        processing_times, due_dates, weights, setup_times,
                        machine_last_job, p_bar
                    )

                    if priority > best_priority:
                        best_priority = priority
                        best_job = job_id
                        best_machine = machine_id

            if best_job is None:
                # No available job, advance time
                current_time = min(
                    [release_dates[j] for j in remaining_jobs if release_dates[j] > current_time],
                    default=machine_available_time.min()
                )
                continue

            # Schedule the job
            start_time = max(machine_available_time[best_machine], release_dates[best_job])
            last_job = machine_last_job[best_machine]
            setup_time = setup_times[last_job + 1, best_job, best_machine]

            start_time += setup_time
            completion_time = start_time + processing_times[best_job, best_machine]

            # Update state
            machine_available_time[best_machine] = completion_time
            machine_last_job[best_machine] = best_job
            job_completion_times[best_job] = completion_time
            schedule[best_machine].append(best_job)
            remaining_jobs.remove(best_job)
            total_setup_time += setup_time

        # Calculate objectives
        total_weighted_tardiness = sum(
            weights[j] * max(0, job_completion_times[j] - due_dates[j])
            for j in range(n_jobs)
        )

        return {
            'schedule': schedule,
            'job_completion_times': job_completion_times,
            'total_weighted_tardiness': total_weighted_tardiness,
            'total_setup_time': total_setup_time,
        }

    def _calculate_priority(
        self, job_id: int, machine_id: int, current_time: float,
        processing_times, due_dates, weights, setup_times,
        machine_last_job, p_bar: float
    ) -> float:
        """Calculate ATCSR priority for job-machine pair."""

        p_jk = processing_times[job_id, machine_id]
        last_job = machine_last_job[machine_id]
        s_ijk = setup_times[last_job + 1, job_id, machine_id]

        # Slack time
        slack = due_dates[job_id] - current_time - p_jk - s_ijk

        # ATCSR formula components
        if slack < 0:
            slack_factor = np.exp(-max(0, slack) / (self.k2 * p_bar))
        else:
            slack_factor = np.exp(-slack / (self.k2 * p_bar))

        setup_factor = np.exp(-s_ijk / (self.k1 * p_bar))

        # Priority
        priority = (weights[job_id] / p_jk) * slack_factor * setup_factor

        return priority


class GeneticAlgorithmSolver:
    """
    Genetic Algorithm for multi-objective UPMSP.
    Minimizes weighted sum of normalized TWT and TST.
    """

    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        alpha: float = 0.5,  # weight for TWT in fitness
        time_limit: float = 60.0,  # seconds
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.alpha = alpha
        self.time_limit = time_limit

    def solve(self, instance: Dict) -> Dict:
        """
        Solve UPMSP instance using GA.

        Args:
            instance: problem instance

        Returns:
            Dictionary containing schedule and objectives
        """
        import time
        start_time = time.time()

        # Get baseline for normalization
        atcsr_solver = ATCSR_Rm_Solver()
        baseline_result = atcsr_solver.solve(instance)
        twt_baseline = baseline_result['total_weighted_tardiness']
        tst_baseline = baseline_result['total_setup_time']

        # Initialize population
        population = self._initialize_population(instance)

        best_solution = None
        best_fitness = np.inf

        generation = 0
        while generation < self.n_generations and (time.time() - start_time) < self.time_limit:
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                result = self._decode_solution(individual, instance)
                fitness = self._calculate_fitness(
                    result['total_weighted_tardiness'],
                    result['total_setup_time'],
                    twt_baseline,
                    tst_baseline
                )
                fitness_scores.append(fitness)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = copy.deepcopy(result)

            # Selection
            new_population = []
            for _ in range(self.population_size):
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # Crossover
                if np.random.rand() < self.crossover_rate:
                    offspring = self._crossover(parent1, parent2)
                else:
                    offspring = parent1

                # Mutation
                if np.random.rand() < self.mutation_rate:
                    offspring = self._mutate(offspring)

                new_population.append(offspring)

            population = new_population
            generation += 1

        return best_solution if best_solution else baseline_result

    def _initialize_population(self, instance: Dict) -> List[Dict]:
        """Initialize population with random solutions."""
        population = []
        n_jobs = instance['n_jobs']
        n_machines = instance['n_machines']
        eligibility = instance['eligibility']

        for _ in range(self.population_size):
            # Random job assignment and sequencing
            individual = {m: [] for m in range(n_machines)}
            for job_id in range(n_jobs):
                # Assign to random eligible machine
                eligible_machines = np.where(eligibility[job_id])[0]
                machine = np.random.choice(eligible_machines)
                individual[machine].append(job_id)

            # Shuffle sequences
            for machine in range(n_machines):
                np.random.shuffle(individual[machine])

            population.append(individual)

        return population

    def _decode_solution(self, individual: Dict, instance: Dict) -> Dict:
        """Decode chromosome to schedule and calculate objectives."""
        n_jobs = instance['n_jobs']
        n_machines = instance['n_machines']
        processing_times = instance['processing_times']
        release_dates = instance['release_dates']
        due_dates = instance['due_dates']
        weights = instance['weights']
        setup_times = instance['setup_times']

        machine_time = np.zeros(n_machines)
        job_completion_times = np.full(n_jobs, -1.0)
        total_setup_time = 0.0

        for machine_id in range(n_machines):
            last_job = -1
            for job_id in individual[machine_id]:
                start_time = max(machine_time[machine_id], release_dates[job_id])
                setup_time = setup_times[last_job + 1, job_id, machine_id]
                start_time += setup_time
                completion_time = start_time + processing_times[job_id, machine_id]

                machine_time[machine_id] = completion_time
                job_completion_times[job_id] = completion_time
                total_setup_time += setup_time
                last_job = job_id

        total_weighted_tardiness = sum(
            weights[j] * max(0, job_completion_times[j] - due_dates[j])
            for j in range(n_jobs)
        )

        return {
            'schedule': individual,
            'job_completion_times': job_completion_times,
            'total_weighted_tardiness': total_weighted_tardiness,
            'total_setup_time': total_setup_time,
        }

    def _calculate_fitness(self, twt: float, tst: float, twt_base: float, tst_base: float) -> float:
        """Calculate fitness as weighted sum of normalized objectives."""
        twt_norm = twt / (twt_base + 1e-6)
        tst_norm = tst / (tst_base + 1e-6)
        return self.alpha * twt_norm + (1 - self.alpha) * tst_norm

    def _tournament_selection(self, population: List, fitness_scores: List, k: int = 3) -> Dict:
        """Tournament selection."""
        indices = np.random.choice(len(population), k, replace=False)
        best_idx = indices[np.argmin([fitness_scores[i] for i in indices])]
        return copy.deepcopy(population[best_idx])

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Order crossover for job sequences."""
        offspring = {}
        n_machines = len(parent1)

        for machine in range(n_machines):
            seq1 = parent1[machine]
            seq2 = parent2[machine]

            if len(seq1) == 0 or len(seq2) == 0:
                offspring[machine] = copy.deepcopy(seq1 if np.random.rand() < 0.5 else seq2)
                continue

            # Simple crossover: take jobs from both parents
            all_jobs = set(seq1 + seq2)
            offspring[machine] = []
            for job in seq1:
                if job in all_jobs:
                    offspring[machine].append(job)
                    all_jobs.remove(job)

        return offspring

    def _mutate(self, individual: Dict) -> Dict:
        """Swap mutation."""
        mutated = copy.deepcopy(individual)
        n_machines = len(individual)

        # Random machine
        machine = np.random.randint(n_machines)
        if len(mutated[machine]) >= 2:
            # Swap two jobs
            idx1, idx2 = np.random.choice(len(mutated[machine]), 2, replace=False)
            mutated[machine][idx1], mutated[machine][idx2] = \
                mutated[machine][idx2], mutated[machine][idx1]

        return mutated
