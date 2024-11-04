import yaml

with open("./configs/simulation.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

args = type("args", (object,), config)()

print(args.content_size)
