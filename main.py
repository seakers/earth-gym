import argparse
import traceback
from datetime import datetime

from classes.instances import Gym

if __name__ == "__main__":
    try:
        # Gather the arguments
        argparse = argparse.ArgumentParser()

        argparse.add_argument("--host", default="localhost", type=str, help="Host address.")
        argparse.add_argument("--port", default=5555, type=int, help="Port number.")
        argparse.add_argument("--conf", type=str, help="Input file path.")
        argparse.add_argument("--evpt", type=str, help="Event points file path.")
        argparse.add_argument("--out", type=str, help="Output folder.")

        args = argparse.parse_args()

        # Create gym environment
        gym_env = Gym(args=args)

        # Time before the process
        start_time = datetime.now()

        # Start the environment
        gym_env.start(host=args.host, port=args.port)

        # Time after the process
        end_time = datetime.now()

        # Calculate the time difference
        time_diff = end_time - start_time
        print(f"Time elapsed: {time_diff}")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        print("Earth Gym was shut down. Bye!")