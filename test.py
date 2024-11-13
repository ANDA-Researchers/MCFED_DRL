import datetime
import os

from tqdm import tqdm

from simulation import Environment
from delivery import random_delivery
from utils import load_args, save_results

args, configs = load_args()


def main():
    env = Environment(
        args=args,
    )

    state, mask = env.reset()
    for round in range(args.num_rounds):
        print(f"Round: {round}")
        for step in tqdm(range(args.time_step_per_round)):
            action = random_delivery(env)

            for vehicle_idx in range(args.num_vehicle):
                print("=================")
                print(
                    "d from BS: {:.2f}, h: {:.10f}, V2N datarate: {:.2f}".format(
                        env.mobility.distance[vehicle_idx, 0],
                        env.channel.channel_gain[vehicle_idx, 0],
                        env.channel.data_rate[vehicle_idx, 0],
                    )
                )
                print(
                    "d from RSU: {:.2f}, h: {:.10f}, V2I datarate: {:.2f}".format(
                        env.mobility.distance[vehicle_idx, 1],
                        env.channel.channel_gain[vehicle_idx, 1],
                        env.channel.data_rate[vehicle_idx, 1],
                    )
                )

            _, _, reward, logs = env.step(action)


if __name__ == "__main__":
    main()
