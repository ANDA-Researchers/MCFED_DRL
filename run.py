import datetime
import os

import numpy as np
import torch
from tqdm import tqdm

from cache import avgfed, mcfed, random_cache, nocache
from delivery import greedy_delivery, nocache_delivery, random_delivery
from module.bdqn import BDQNAgent
from simulation import Environment
from utils import load_args, save_results

torch.manual_seed(0)
np.random.seed(0)


args, configs = load_args()


def load_weights(agent: BDQNAgent, path):
    agent.policy_net.load_state_dict(torch.load(path, weights_only=True))


def main(cache, delivery, cache_size):

    created_at = datetime.datetime.now().strftime(f"%Y%m%d-%H%M%S")

    args.rsu_capacity = cache_size

    env = Environment(
        args=args,
    )

    agent = BDQNAgent(
        state_dim=env.state_dim,
        num_actions=args.num_vehicle,
        action_dim=args.num_rsu + 2,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lr=args.lr,
        capacity=args.capacity,
        batch_size=args.batch_size,
        target_update=args.target_update,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        device=args.device,
        logger=None,
        dueling=True,
    )

    # Load the weights
    load_weights(agent, "./model.pth")

    delay_tracking = []
    round_avg_delay_tracking = []
    round_hit_ratio_tracking = []
    round_success_ratio_tracking = []

    global_total_request = 0
    global_total_hits = 0
    global_total_success = 0

    state = env.reset()
    for round in range(args.num_rounds):
        print(f"Round: {round}")
        round_total_request = 0
        round_total_hits = 0
        round_total_success = 0
        # Put cache replacement policy here
        if cache == "random":
            random_cache(env)
        elif cache == "mcfed":
            mcfed(env)
        elif cache == "avgfed":
            avgfed(env)
        elif cache == "nocache":
            nocache(env)

        state = env.state

        for timestep in tqdm(range(args.time_step_per_round), desc=f"timestep "):
            if delivery == "random":
                action = random_delivery(env)
            elif delivery == "greedy":
                action = greedy_delivery(env)
            elif delivery == "norsu":
                action = nocache_delivery(env)
            elif delivery == "drl":
                action = agent.act(state, inference=True)

            # _, reward, logs = env.step(action)
            next_state, reward, logs = env.step(action)
            avg_delay, total_request, total_hits, total_success = logs

            global_total_request += total_request
            global_total_hits += total_hits
            global_total_success += total_success

            round_total_request += total_request
            round_total_hits += total_hits
            round_total_success += total_success

            delay_tracking.append(avg_delay)

            state = next_state

        round_avg_delay = (
            sum(delay_tracking[-args.time_step_per_round :]) / args.time_step_per_round
        )
        round_hit_ratio = round_total_hits / round_total_request
        round_success_ratio = round_total_success / round_total_request

        round_avg_delay_tracking.append(round_avg_delay)
        round_hit_ratio_tracking.append(round_hit_ratio)
        round_success_ratio_tracking.append(round_success_ratio)

    # save the results as json
    results = {
        "total_request": global_total_request,
        "total_hits": global_total_hits,
        "total_success": global_total_success,
        "delay_tracking": delay_tracking,
        "round_avg_delay_tracking": round_avg_delay_tracking,
        "round_hit_ratio_tracking": round_hit_ratio_tracking,
        "round_success_ratio_tracking": round_success_ratio_tracking,
        "args": args.__dict__,
    }
    save_dir = os.path.join(
        "results",
        f"[{created_at}] {args.num_rsu}_{args.rsu_capacity}_{args.num_vehicle}_{cache}_{delivery}",
    )
    save_results(save_dir, cache, delivery, results)


if __name__ == "__main__":
    for cache_size in [5, 10, 30, 50, 80, 100]:
        for cache in [
            # "random",
            # "mcfed",
            # "avgfed",
            "nocache",
        ]:
            for delivery in ["random", "greedy", "drl", "norsu"]:
                if cache in ["random", "avgfed", "nocache"] and delivery not in [
                    "drl",
                    "norsu",
                ]:
                    continue
                if delivery == "norsu" and cache != "random":
                    continue
                main(cache, delivery, cache_size)
