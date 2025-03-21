import io
import pstats
import cProfile
import argparse
import traceback
import tracemalloc
from datetime import datetime

from scripts.instances import Gym

if __name__ == "__main__":
    try:
        print("Starting the main script of ppo-eos...")

        # Time before the process
        start_time = datetime.now()

        # Gather the arguments
        argparse = argparse.ArgumentParser()

        argparse.add_argument("--host", default="localhost", type=str, help="Host address.")
        argparse.add_argument("--port", default=5555, type=int, help="Port number.")
        argparse.add_argument("--conf", type=str, help="Input file path.")
        argparse.add_argument("--evpt", type=str, help="Event points file path.")
        argparse.add_argument("--out", type=str, help="Output folder.")
        argparse.add_argument("--pro", type=int, help="Profiling and memory allocation mode.")

        args = argparse.parse_args()

        if args.pro:
            print("Tracking profile and memory allocation...")
            # Start tracing memory allocations
            tracemalloc.start()

        # Create gym environment
        gym_env = Gym(args=args)

        if args.pro:
            ###################### Gym with profiling ######################
            # Check the performance
            pr = cProfile.Profile()
            pr.enable()

            # Start the environment
            gym_env.start(host=args.host, port=args.port)

            pr.disable()

            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats("tottime") # sort by total time
            ps.print_stats(40) # show top n slowest functions
            print(s.getvalue())
            pr.dump_stats("src/main-profile.prof")
            ###################### Gym with profiling ######################
        else:
            # Start the environment
            gym_env.start(host=args.host, port=args.port)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        if args.pro:
            ###################### Memory allocation ################
            # Take a snapshot after executing the code
            snapshot = tracemalloc.take_snapshot()

            # Get the top memory allocations
            top_stats = snapshot.statistics('lineno')

            print("Top 100 memory-consuming lines:")
            for stat in top_stats[:100]:
                print(stat)

            # Stop tracing
            tracemalloc.stop()
            ###################### Memory allocation ################

        # Time after the process
        end_time = datetime.now()

        # Calculate the time difference
        time_diff = end_time - start_time
        print(f"Time elapsed: {time_diff}")
        print("Earth Gym was shut down. Bye!")