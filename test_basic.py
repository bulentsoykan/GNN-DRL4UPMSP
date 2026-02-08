"""
Basic tests to verify implementation works correctly.
"""

import numpy as np
from src.instance_generator import InstanceGenerator
from src.environment import UPMSPEnvironment
from src.baselines import ATCSR_Rm_Solver, GeneticAlgorithmSolver

def test_instance_generation():
    """Test instance generator."""
    print("Testing instance generation...")
    generator = InstanceGenerator(seed=42)

    instance = generator.generate(
        n_jobs=10,
        n_machines=3,
        tau=0.4,
        R=0.6,
        beta=0.1,
        delta=0.75
    )

    assert instance['n_jobs'] == 10
    assert instance['n_machines'] == 3
    assert instance['processing_times'].shape == (10, 3)
    assert instance['eligibility'].shape == (10, 3)
    assert instance['setup_times'].shape == (11, 10, 3)  # (n_jobs+1, n_jobs, n_machines)

    print("  ✓ Instance generation working")


def test_environment():
    """Test environment."""
    print("Testing environment...")
    generator = InstanceGenerator(seed=42)
    instance = generator.generate(n_jobs=5, n_machines=2)

    env = UPMSPEnvironment(instance)
    obs, info = env.reset()

    assert 'current_time' in obs
    assert len(env.remaining_jobs) == 5

    # Test action masking
    mask = env.get_action_mask()
    assert mask.sum() > 0  # At least some actions should be feasible

    print("  ✓ Environment working")


def test_atcsr_solver():
    """Test ATCSR_Rm solver."""
    print("Testing ATCSR_Rm solver...")
    generator = InstanceGenerator(seed=42)
    instance = generator.generate(n_jobs=10, n_machines=3)

    solver = ATCSR_Rm_Solver()
    result = solver.solve(instance)

    assert 'schedule' in result
    assert 'total_weighted_tardiness' in result
    assert 'total_setup_time' in result
    assert result['total_weighted_tardiness'] >= 0
    assert result['total_setup_time'] >= 0

    print(f"  TWT: {result['total_weighted_tardiness']:.2f}")
    print(f"  TST: {result['total_setup_time']:.2f}")
    print("  ✓ ATCSR_Rm solver working")


def test_ga_solver():
    """Test GA solver."""
    print("Testing GA solver...")
    generator = InstanceGenerator(seed=42)
    instance = generator.generate(n_jobs=10, n_machines=3)

    solver = GeneticAlgorithmSolver(
        population_size=10,
        n_generations=5,
        time_limit=5.0
    )
    result = solver.solve(instance)

    assert 'schedule' in result
    assert 'total_weighted_tardiness' in result
    assert 'total_setup_time' in result
    assert result['total_weighted_tardiness'] >= 0
    assert result['total_setup_time'] >= 0

    print(f"  TWT: {result['total_weighted_tardiness']:.2f}")
    print(f"  TST: {result['total_setup_time']:.2f}")
    print("  ✓ GA solver working")


def test_comparison():
    """Test comparison of methods."""
    print("\nRunning comparison test...")
    generator = InstanceGenerator(seed=42)
    instance = generator.generate(n_jobs=20, n_machines=5)

    print(f"  Problem: {instance['n_jobs']} jobs, {instance['n_machines']} machines")

    # ATCSR_Rm
    atcsr_solver = ATCSR_Rm_Solver()
    atcsr_result = atcsr_solver.solve(instance)

    # GA
    ga_solver = GeneticAlgorithmSolver(
        population_size=20,
        n_generations=20,
        time_limit=10.0
    )
    ga_result = ga_solver.solve(instance)

    print(f"\n  ATCSR_Rm:")
    print(f"    TWT: {atcsr_result['total_weighted_tardiness']:.2f}")
    print(f"    TST: {atcsr_result['total_setup_time']:.2f}")

    print(f"\n  GA:")
    print(f"    TWT: {ga_result['total_weighted_tardiness']:.2f}")
    print(f"    TST: {ga_result['total_setup_time']:.2f}")

    # GA should generally be better than ATCSR_Rm
    print("\n  ✓ Comparison test complete")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Basic Functionality Tests")
    print("=" * 60)
    print()

    try:
        test_instance_generation()
        test_environment()
        test_atcsr_solver()
        test_ga_solver()
        test_comparison()

        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print()
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        raise


if __name__ == '__main__':
    run_all_tests()
