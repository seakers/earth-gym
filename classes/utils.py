import os
import math
import pandas as pd
import matplotlib.pyplot as plt

class DataFromJSON():
    """
    Class to manage the data of the model. Functions:
    - __init__: iterates over the JSON dictionary and sets the attributes of the class.
                Parent objects are ommitted and only the leaf nodes are stored. Lists are stored.
                Adds the data type of the object in self.data_type.
    """
    def __init__(self, json_dict, data_type: str):
        self.loop(json_dict)
        self.data_type = data_type

    def loop(self, json_dict):
        if not isinstance(json_dict, dict):
            return
        for key, value in json_dict.items():
            if isinstance(value, dict):
                self.loop(value)
            else:
                if hasattr(self, key):
                    raise ValueError(f"Variable {key} already exists in the class. Rename the json key in your configuration file.")
                else:
                    setattr(self, key, value)

    def get_dict(self):
        """
        Return the dictionary of the class.
        """
        return self.__dict__

class DateManager():
    """
    Class to understand and manage the date and time of the simulation. Functions:
    - simplify_date: returns the date in a simplified, more readable format.
    - fancy_date: returns the date in the fancy stk-used format.
    - month_to_number: returns the number of the month.
    - number_to_month: returns the month of the number.
    - number_of_days_in_month: returns the number of days in the month.
    - update_date_after: returns the date after a given time increment.
    """
    def __init__(self, start_date: str, stop_date: str):
        self.class_name = "Date Manager"
        self.start_date = start_date
        self.stop_date = stop_date
        self.current_date = start_date # in fancy stk-used format
        self.last_date = start_date
        self.current_simplified_date = self.simplify_date(start_date) # all in numbers concatenated in a string
        self.last_simplified_date = self.current_simplified_date

    def simplify_date(self, date: str):
        """
        Return the date in a simplified, more readable format.
        """
        # Separate the date into year, month, day
        day, month, year, clock = date.split(" ")
        hour, minute, second = clock.split(":")
        return f"{day} {self.month_to_number(month)} {year} {hour} {minute} {second}"
    
    def fancy_date(self, date: str):
        """
        Return the date in the fancy stk-used format.
        """
        day, month, year, hour, minute, second = date.split(" ")
        return f"{day} {self.number_to_month(int(month))} {year} {hour}:{minute}:{second}"
    
    def month_to_number(self, month: str):
        """
        Return the number of the month.
        """
        months = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "Jun": 5, "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}
        if month not in months:
            raise ValueError("Month must be a valid month abbreviation.")
        
        return months[month]
    
    def number_to_month(self, number: int):
        """
        Return the month of the number.
        """
        months = {0: "Jan", 1: "Feb", 2: "Mar", 3: "Apr", 4: "May", 5: "Jun", 6: "Jul", 7: "Aug", 8: "Sep", 9: "Oct", 10: "Nov", 11: "Dec"}
        if number < 0 or number > 11:
            raise ValueError("Month number must be between 0 and 11.")

        return months[number]
    
    def number_of_days_in_month(self, month: str, year: int):
        """
        Return the number of days in the month.
        """
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if month not in months:
            raise ValueError("Month must be a valid month abbreviation.")
        
        if month in ["Jan", "Mar", "May", "Jul", "Aug", "Oct", "Dec"]:
            return 31
        elif month in ["Apr", "Jun", "Sep", "Nov"]:
            return 30
        elif month == "Feb":
            if year % 4 == 0:
                return 29
            else:
                return 28
        else:
            raise ValueError("Month must be a valid month abbreviation.")
        
    def number_of_days_in_year(self, year: int):
        """
        Return the number of days in the year.
        """
        if year % 4 == 0:
            return 366
        else:
            return 365
        
    def get_current_date_after(self, delta_time, return_simplified: bool=False):
        """
        Return the date after the given number of time.
        """
        # Get the current date
        day, month, year, hour, minute, second = self.current_simplified_date.split(" ")
        day = int(day)
        month = int(month)
        year = int(year)
        hour = int(hour)
        minute = int(minute)
        second = float(second)

        # Identify the delta time data type
        if isinstance(delta_time, dict):
            delta_seconds = delta_time["seconds"]
            delta_seconds += delta_time["minutes"] * 60
            delta_seconds += delta_time["hours"] * 3600
            delta_seconds += delta_time["days"] * 86400
            delta_seconds += delta_time["months"] * 2592000
            delta_seconds += delta_time["years"] * 31536000
        elif isinstance(delta_time, int) or isinstance(delta_time, float):
            delta_seconds = delta_time
        else:
            raise ValueError("Delta time must be a dictionary or a number.")

        # Add the increments
        second += delta_seconds
        minute += int(second / 60)
        second = second % 60
        hour += int(minute / 60)
        minute = minute % 60
        day += int(hour / 24)
        hour = hour % 24
        month += int(day / self.number_of_days_in_month(self.number_to_month(month), year))
        day = day % self.number_of_days_in_month(self.number_to_month(month), year)
        year += int(month / 12)
        month = month % 12

        # Simplified date
        simplified = f"{day} {month} {year} {hour} {minute} {second}"

        if return_simplified:
            return simplified
        else:
            return self.fancy_date(simplified)

    def update_date_after(self, delta_time):
        """
        Return the date after the given number of days.
        """
        # Store the last date
        self.last_date = self.current_date
        self.last_simplified_date = self.current_simplified_date

        # Store and return the new date
        self.current_simplified_date = self.get_current_date_after(delta_time, return_simplified=True)
        self.current_date = self.fancy_date(self.current_simplified_date)

    def time_ended(self) -> bool:
        """
        Check if the time has ended.
        """
        # Get the stop date
        simplified_stop_date = self.simplify_date(self.stop_date)
        num_stop_date = self.num_of_date(simplified_stop_date)

        # Get the current date
        num_current_date = self.num_of_date(self.current_simplified_date)

        # Check if the time has ended
        if num_current_date > num_stop_date:
            return True
        else:
            return False

    def num_of_date(self, date: str) -> float:
        """
        Return the number of the simplified date used by converting:
        - seconds: just as they are.
        - minutes: x60 seconds added to the seconds.
        - hours: x3600 seconds added to the seconds.
        - days: x86400 seconds added to the seconds.
        - months: x86400 number of days in month added to the seconds.
        - years: x86400 number of days in year added to the seconds.
        """
        day, month, year, hour, minute, _ = [int(float(i)) for i in date.split(" ")]
        _, _, _, _, _, second = [float(i) for i in date.split(" ")]
        
        # Calculate the seconds
        month_seconds = self.number_of_days_in_month(self.number_to_month(month), year) * 86400
        year_seconds = self.number_of_days_in_year(year) * 86400
        seconds = second + minute * 60 + hour * 3600 + day * 86400 + month * month_seconds + year * year_seconds
        return seconds

class SensorManager():
    """
    Class to understand and manage the date and time of the simulation. Functions:
    - get_item: return the value of the item.
    - update_azimuth: update the azimuth of the sensor within the boundaries.
    - update_elevation: update the elevation of the sensor within the boundaries.
    """
    def __init__(self, agent, sensor):
        self.class_name = "Sensor Manager"
        self.sensor = sensor
        self.pattern = agent["pattern"]
        self.cone_angle = agent["cone_angle"]
        self.resolution = agent["resolution"]
        self.current_azimuth = agent["initial_azimuth"]
        self.current_elevation = agent["initial_elevation"]

    def get_item(self, name):
        """
        Return the value of the item.
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise ValueError(f"Variable {name} does not exist in the class. Check the configuration file.")

    def update_azimuth(self, delta_azimuth):
        """
        Update the azimuth of the sensor within the boundaries.
        """
        self.current_azimuth += delta_azimuth

        # Correct the azimuth if out of boundaries
        if self.current_azimuth > 360:
            self.current_azimuth -= 360
        elif self.current_azimuth < 0:
            self.current_azimuth += 360

        return self.current_azimuth
    
    def update_elevation(self, delta_elevation):
        """
        Update the elevation of the sensor within the boundaries.
        """
        self.current_elevation += delta_elevation

        # Correct the elevation if out of boundaries
        if self.current_elevation > 90:
            self.current_elevation = 90
        elif self.current_elevation < -90:
            self.current_elevation = -90

        return self.current_elevation
    
class FeaturesManager():
    """
    Class to manage the features of the model. Functions:
    - set_properties: return the properties of the agent.
    - update_state: update the state properties of the agent.
    - update_action: update the action properties of the agent.
    """
    def __init__(self, agent):
        self.class_name = "Features Manager"
        self.agent_config = agent
        self.set_properties(agent)

    def set_properties(self, agent):
        """
        Set the properties of the agent in the states and actions objects.
        """
        # Initialize the states and actions objects
        self.state = {}
        self.action = {}
        self.states_features = agent["states_features"]
        self.actions_features = agent["actions_features"]

        # Iterate over the states
        for state in self.states_features:
            if state in agent.keys():
                self.state[state] = agent[state]
            else:
                raise ValueError(f"State {state} does not exist in the agent initial state. Check the configuration file.")
            
        # Iterate over the actions
        for action in self.actions_features:
            self.action[action] = 0
    
    def get_state(self):
        """
        Return the properties of the agent.
        """
        return self.state
                   
    def update_state(self, name, value):
        """
        Update the state properties of the agent.
        """
        if name in self.state.keys():
            self.state[name] = value
        else:
            raise ValueError(f"Variable {name} does not exist in the class.")
        
    def update_action(self, name, value):
        """
        Update the action properties of the agent.
        """
        if name in self.action.keys():
            self.action[name] = value
        else:
            raise ValueError(f"Variable {name} does not exist in the class.")
        
    def update_orbital_elements(self, orbital_elements):
        """
        Update the orbital elements of the agent.
        """
        # Update into the feature manager
        if self.agent_config["coordinate_system"] == "Classical":
            for key in ["a", "e", "i", "raan", "aop", "ta"]:
                self.update_state(key, orbital_elements.DataSets.GetDataSetByName(self.long_name_of(key)).GetValues()[0])
        elif self.agent_config["coordinate_system"] == "Cartesian":
            for key in ["x", "y", "z", "vx", "vy", "vz"]:
                self.update_state(key, orbital_elements.DataSets.GetDataSetByName(self.long_name_of(key)).GetValues()[0])
        else:
            raise ValueError("Invalid coordinate system. Please use 'Classical' or 'Cartesian'.")
        
    def long_name_of(self, short_name):
        """
        Return the long name of the short name.
        """
        short_to_long = {"a": "Semi-major Axis", "e": "Eccentricity", "i": "Inclination", "raan": "RAAN", "aop": "Arg of Perigee", "ta": "True Anomaly"}
        return short_to_long[short_name]
    
    def short_name_of(self, long_name):
        """
        Return the short name of the long name.
        """
        long_to_short = {"Semi-major Axis": "a", "Eccentricity": "e", "Inclination": "i", "RAAN": "raan", "Arg of Perigee": "aop", "True Anomaly": "ta"}
        return long_to_short[long_name]

class Rewarder():
    """
    Class to manage the reward of the model. Functions:
    - calculate_reward: return the reward of the state-action pair.
    - f_ri: return the reward of the observation.
    - f_theta: return the reward of the angle between the event and the satellite.
    - f_reobs: return the reward of the reobservation of the same event.
    """
    def __init__(self, agents_config):
        self.class_name = "Rewarder"
        self.seen_events = []
        self.agents_config = agents_config

    def calculate_reward(self, data_providers, date_mg : DateManager):
        """
        Return the reward of the state-action pair given the proper data providers (acces and aer).
        """
        reward = 0

        # Iterate over the access data providers
        for access_data_provider, aer_data_provider in data_providers:
            # Check if the access is valid
            if access_data_provider.Intervals.Count > 0:
                for i in range(access_data_provider.Intervals.Count):
                    # Information from the access data provider
                    start_time = access_data_provider.Intervals.Item(i).DataSets.GetDataSetByName("Start Time").GetValues()
                    stop_time = access_data_provider.Intervals.Item(i).DataSets.GetDataSetByName("Stop Time").GetValues()
                    to_object = access_data_provider.Intervals.Item(i).DataSets.GetDataSetByName("To Object").GetValues()

                    # Information from the aer data provider
                    zen_angles = aer_data_provider.Intervals.Item(i).DataSets.GetDataSetByName("Elevation").GetValues()

                    # Iterate over the unlike case of the target being accessed multiple times in the step
                    for j in range(len(start_time)):
                        # Get the event name by eliminating the "To Target" string from STK output
                        event_name = to_object[j].replace("To Target", "").strip()

                        # Find the maximum (best) angle of elevation
                        max_zen_angle = max([abs(el) for el in zen_angles])

                        # Prints
                        print(f"\nEvent: {event_name}")
                        print(f"Seen from {start_time[j]} to {stop_time[j]}.")
                        print("Date difference in start", -date_mg.num_of_date(date_mg.last_simplified_date) + date_mg.num_of_date(date_mg.simplify_date(start_time[j])))
                        print("Date difference in stop", -date_mg.num_of_date(date_mg.simplify_date(start_time[j])) + date_mg.num_of_date(date_mg.simplify_date(stop_time[j])))

                        # Minimum event duration
                        min_duration = self.agents_config["min_duration"]

                        # Check is long enough based on min_duration
                        if (date_mg.num_of_date(date_mg.simplify_date(stop_time[j])) - date_mg.num_of_date(date_mg.simplify_date(start_time[j]))) > min_duration:
                            reward_was_seen = False

                            # Check if the event has been seen before and how many times
                            for i, seen in enumerate(self.seen_events):
                                if seen[0] == event_name:
                                    # The event has been seen before
                                    last_seen = date_mg.num_of_date(date_mg.simplify_date(seen[2])) # need this line to avoid pointer to tuple
                                    reward_was_seen = True
                                    self.seen_events[i][2] = stop_time[j]

                                    # This filters concatenated observations (which indeed are one observation)
                                    if (last_seen - min_duration) >= date_mg.num_of_date(date_mg.simplify_date(start_time[j])): # min_duration added because of added in .Exec() too
                                        break

                                    # Event is seen again
                                    self.seen_events[i][1] += 1

                                    # Reward the observation
                                    n_obs = self.seen_events[i][1]
                                    ri = self.f_ri(max_zen_angle, n_obs)
                                    reward += ri
                                    print(f"Observed {event_name} with zenith {max_zen_angle:0.2f}º and reward of {ri:0.4f} (total of {reward:0.4f}).")
                                    break
                            
                            # If the event has not been seen before
                            if not reward_was_seen:
                                self.seen_events.append([event_name, 1, stop_time[j]])
                                
                                # Reward the observation
                                ri = self.f_ri(max_zen_angle, 1)
                                reward += ri
                                print(f"First observed {event_name} with zenith {max_zen_angle:0.2f}º and reward of {ri:0.4f} (total of {reward:0.4f}).")
        
        return reward
    
    def f_ri(self, max_zen_angle: float, n_obs: int):
        """
        Function rewarding a certain observation. Inputs given in the form of tuples.
        - event_pos: tuple of the event position (latitude, longitude, altitude).
        - satellite_pos: tuple of the satellite position (latitude, longitude, altitude).
        - event_name: name of the event target ('target1', for instance).
        - max_zen_angle: maximum elevation angle of the event.
        """
        # Arbitrary profit TODO: make it a parameter
        profit = 1

        # Each of the value functions
        f_theta = self.f_theta(max_zen_angle)
        f_reobs = self.f_reobs(n_obs)

        return profit * f_reobs * f_theta

    def f_theta(self, max_zen_angle: float):
        """
        Function rewarding the angle between the event and the satellite. Inputs given in the form of a list.
        - el_angles: list of elevation angles.
        """
        return self.agents_config["zenith_weight"] * math.sin(math.radians(max_zen_angle)) # the higher the better, the angle is in degrees
    
    def f_reobs(self, n_obs: int):
        """
        Function rewarding the reobservation of the same event. Inputs given in the form of an integer.
        - n_obs: number of times the event has been observed.
        """
        return (1 / n_obs**self.agents_config["reobs_decay"]) if n_obs > 0 else 1
    
class Plotter():
    """
    Class to manage the plotting of the model
    """
    def __init__(self, out_folder_path: str="${workspaceFolder}\\output"):
        self.class_name = "Plotter"
        self.rewards = pd.DataFrame()
        self.out_folder_path = out_folder_path

    def store_reward(self, reward):
        """
        Store the reward in the list.
        """
        reward = pd.DataFrame([reward])
        self.rewards = pd.concat([self.rewards, reward], ignore_index=True)

    def plot_rewards(self):
        """
        Plot the rewards as they are.
        """
        if self.rewards.empty:
            raise ValueError("No rewards to plot.")
        
        # Clear the plot
        plt.clf()
        
        # Plot
        plt.plot(self.rewards)
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Rewards over time")

        # Save the plot
        if not os.path.exists(self.out_folder_path):
            os.makedirs(self.out_folder_path)
        plt.savefig(f"{self.out_folder_path}\\rewards.png", dpi=500)
    
    def plot_rewards_smoothed(self, window_size: int=10):
        """
        Plot the rewards within smoothed windows of size window_size.
        """
        if self.rewards.empty:
            raise ValueError("No rewards to plot.")
        
        # Clear the plot
        plt.clf()
        
        # Smoothed
        smoothed_rewards = self.rewards.rolling(window=window_size).mean()

        # Plot
        plt.plot(smoothed_rewards)
        plt.xlabel("Step")
        plt.ylabel("Smoothed reward")
        plt.title("Smoothed rewards over time")
        
        # Save the plot
        if not os.path.exists(self.out_folder_path):
            os.makedirs(self.out_folder_path)
        plt.savefig(f"{self.out_folder_path}\\rewards_smoothed.png", dpi=500)

    def plot_cumulative_rewards(self):
        """
        Plot the cumulative rewards.
        """
        if self.rewards.empty:
            raise ValueError("No rewards to plot.")
        
        # Clear the plot
        plt.clf()
        
        # Cumulative sum
        cumulative_rewards = self.rewards.cumsum()

        # Plot
        plt.plot(cumulative_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Cumulative reward")
        plt.title("Cumulative reward over time")
        
        # Save the plot
        if not os.path.exists(self.out_folder_path):
            os.makedirs(self.out_folder_path)
        plt.savefig(f"{self.out_folder_path}\\cumulative_rewards.png", dpi=500)

    def plot_cumulative_rewards_smoothed_per_steps(self, window_size: int=10):
        """
        Plot the cumulative rewards per steps.
        """
        if self.rewards.empty:
            raise ValueError("No rewards to plot.")
        
        # Clear the plot
        plt.clf()
        
        # Smoothed  and cumulative sum divided by the step
        smoothed_rewards = self.rewards.rolling(window=window_size).mean()
        cumulative_rewards = smoothed_rewards.cumsum()
        cumulative_rewards = cumulative_rewards.div(pd.Series(range(1, len(cumulative_rewards))), axis=0)

        # Plot
        plt.plot(cumulative_rewards)
        plt.xlabel("Step")
        plt.ylabel("Cumulative reward")
        plt.title("Cumulative reward per step done over time")
        
        # Save the plot
        if not os.path.exists(self.out_folder_path):
            os.makedirs(self.out_folder_path)
        plt.savefig(f"{self.out_folder_path}\\cumulative_rewards_smoothed_per_steps.png", dpi=500)

    def plot_all(self, window_size: int=10):
        """
        Plot all the rewards.
        """
        self.plot_rewards()
        self.plot_rewards_smoothed(window_size=window_size)
        self.plot_cumulative_rewards()
        self.plot_cumulative_rewards_smoothed_per_steps()
