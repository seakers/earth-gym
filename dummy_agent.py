import socket
import json
import argparse
import time

class RLAgent:
    def __init__(self, gym_host="localhost", gym_port=5555):
        self.gym_host = gym_host
        self.gym_port = gym_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.gym_host, self.gym_port))
    
    def send_command(self, command, data=None):
        request = {"command": command}
        if data:
            request.update(data)
        self.sock.sendall(json.dumps(request).encode())
        response = self.sock.recv(1024).decode()
        return json.loads(response)
    
    def get_next_state(self, command, data):
        response = self.send_command(command, data)
        state = response["state"]
        reward = response["reward"]
        done = response["done"]
        return state, reward, done
    
    def shutdown_gym(self):
        response = self.send_command("shutdown")
        print(response["status"])

if __name__ == "__main__":
    # Gather the arguments
    argparse = argparse.ArgumentParser()

    argparse.add_argument("--host", default="localhost", type=str, help="Host address.")
    argparse.add_argument("--port", default=5555, type=int, help="Port number.")

    args = argparse.parse_args()

    # Create agent
    agent = RLAgent(gym_host=args.host, gym_port=args.port)
    
    for episode in range(5):  # Example of training loop
        data = {
            "agent_id": 0,
            "action": {
                "d_az": 0.5,
                "d_el": 0.0
            },
            "delta_time": 89.789
        }
        
        state, reward, done = agent.get_next_state("get_next", data)
        print(f"State: {state}, Reward: {reward}, Done: {done}")
        
        if done:
            break
    
    # Shutdown the environment
    agent.shutdown_gym()