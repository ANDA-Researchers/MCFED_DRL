import yaml


def load_args():
    with open("./configs/simulation.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    args = type("args", (object,), config)()
    return args
