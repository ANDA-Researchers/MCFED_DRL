import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MCFED Simulation")
    # Simulation settings
    # Scenario settings
    parser.add_argument("--length", type=int, default=1000 * 4)
    parser.add_argument("--num_rsu", type=int, default=4)
    parser.add_argument("--num_vehicle", type=int, default=80)
    parser.add_argument("--time_step_per_round", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=30)
    parser.add_argument("--content_size", type=int, default=4e6)
    parser.add_argument("--deadline", type=int, default=1)

    # Mobility settings
    parser.add_argument("--min_velocity", type=int, default=5)
    parser.add_argument("--max_velocity", type=int, default=10)
    parser.add_argument("--std_velocity", type=float, default=2.5)

    # Connection settings
    parser.add_argument("--rsu_coverage", type=int, default=1000)
    parser.add_argument("--rsu_capacity", type=int, default=2000)
    parser.add_argument("--cloud_rate", type=float, default=100e6)
    parser.add_argument("--fiber_rate", type=float, default=150e6)
    parser.add_argument("--max_connections", type=int, default=10)

    # FL settings
    parser.add_argument("--num_clusters", type=int, default=2)
    parser.add_argument("--num_local_epochs", type=int, default=100)
    parser.add_argument("--parallel_update", type=bool, default=False)

    # DRL settings
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--capacity", type=int, default=20000)
    parser.add_argument("--target_update", type=int, default=500)
    parser.add_argument("--training_steps", type=int, default=2000)
    parser.add_argument("--epsilon_decay", type=float, default=0.99995)
    parser.add_argument("--epsilon_min", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--mu", type=float, default=0)

    return parser.parse_args()
