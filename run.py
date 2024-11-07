from simulation import Environment
from utils import load_args
from cache import random_cache, mcfed, avgfed

args, configs = load_args()


def main():

    env = Environment(
        args=args,
    )

    env.reset()

    # random_cache(env)
    # mcfed(env)
    avgfed(env)


if __name__ == "__main__":
    main()
