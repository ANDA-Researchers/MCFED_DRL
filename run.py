import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor

client_size = 10
comb = [
    ("False", "False"), 
    ("True", "False"), 
    ("False", "True"), 
    ("True", "True")
    ]

cmds = []

for sim in [False, True]:
    for c in comb:
        cmd = (
            f"python stage1_evaluate.py --num_clients {client_size}"
            + (" --similarity " if c[0] == "True" else "")
            + (" --use_semantic " if c[1] == "True" else "")
            + " > "
            + ("se_" if sim else "nose_")
            + c[0]
            + "_"
            + c[1]
            + f"_{size}.txt"
        )
        cmds.append(cmd)

# Function to handle keyboard interrupt
def signal_handler(sig, frame):
    print("KeyboardInterrupt caught. Exiting...")
    sys.exit(0)


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


# Execute the commands in parallel processes
def run_command(command):
    os.system(command)


# Execute commands in parallel with a maximum of 2 processes at a time
with ThreadPoolExecutor(max_workers=4) as executor:
    try:
        executor.map(run_command, cmds)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Exiting...")
        executor.shutdown(wait=False)
        sys.exit(0)
