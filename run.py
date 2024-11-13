import datetime
import os

from tqdm import tqdm

from cache import avgfed, mcfed, random_cache
from delivery import greedy_delivery, nocache_delivery, random_delivery
from simulation import Environment
from utils import load_args, save_results

args, configs = load_args()


def main():
    created_at = datetime.datetime.now().strftime(f"%Y%m%d-%H%M%S")

    env = Environment(
        args=args,
    )

    cache = "random"
    delivery = "random"

    delay_tracking = []
    global_total_request = 0
    global_total_hits = 0
    global_total_success = 0

    state, mask = env.reset()
    for round in range(args.num_rounds):
        print(f"Round: {round}")
        # Put cache replacement policy here
        if cache == "random":
            random_cache(env)
        elif cache == "mcfed":
            mcfed(env)
        elif cache == "avgfed":
            avgfed(env)

        for timestep in tqdm(range(args.time_step_per_round), desc=f"timestep "):
            if delivery == "random":
                action = random_delivery(env)
            elif delivery == "greedy":
                action = greedy_delivery(env)
            elif delivery == "nocache":
                action = nocache_delivery(env)

            _, _, reward, logs = env.step(action)

            avg_delay, total_request, total_hits, total_success = logs

            global_total_request += total_request
            global_total_hits += total_hits
            global_total_success += total_success

            delay_tracking.append(avg_delay)

    # save the results as json
    results = {
        "total_request": global_total_request,
        "total_hits": global_total_hits,
        "total_success": global_total_success,
        "delay_tracking": delay_tracking,
    }
    save_dir = os.path.join(
        "results",
        f"[{created_at}] {args.num_rsu}_{args.rsu_capacity}_{args.num_vehicle}_{cache}_{delivery}",
    )
    save_results(save_dir, cache, delivery, results)


if __name__ == "__main__":
    main()
