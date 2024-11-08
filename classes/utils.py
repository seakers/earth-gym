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
    - number_of_days_in: returns the number of days in the month.
    - update_date_after: returns the date after a given time increment.
    """
    def __init__(self, start_date: str, stop_date: str):
        self.class_type = "Date Manager"
        self.start_date = start_date
        self.stop_date = stop_date
        self.current_date = start_date # in fancy stk-used format
        self.current_simplified_date = self.simplify_date(start_date) # all in numbers concatenated in a string

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
    
    def number_of_days_in(self, month: str, year: int):
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

    def update_date_after(self, delta_time):
        """
        Return the date after the given number of days.
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
        month += int(day / self.number_of_days_in(self.number_to_month(month), year))
        day = day % self.number_of_days_in(self.number_to_month(month), year)
        year += int(month / 12)
        month = month % 12

        # Store and return the new date
        self.current_simplified_date = f"{day} {month} {year} {hour} {minute} {second}"
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
        Return the number of the simplified date used.
        """
        day, month, year, hour, minute, _ = [int(float(i)) for i in date.split(" ")]
        _, _, _, _, _, second = [float(i) for i in date.split(" ")]
        return float(f"{year:04}{month:02}{day:02}{hour:02}{minute:02}{second:08.5f}")

class SensorManager():
    """
    Class to understand and manage the date and time of the simulation. Functions:
    - get_item: return the value of the item.
    - update_azimuth: update the azimuth of the sensor within the boundaries.
    - update_elevation: update the elevation of the sensor within the boundaries.
    """
    def __init__(self, agent, sensor):
        self.class_type = "Sensor Manager"
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
        self.class_type = "Features Manager"
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
    """
    def __init__(self):
        self.class_type = "Rewarder"

    def calculate_reward(self, access_data_providers, satellite, sensor_mg, feature_mg, date_mg):
        """
        Return the reward of the state-action pair.
        """
        reward = 0

        # Initiate number of observations
        n_obs = 0

        # Iterate over the access data providers
        for access_data_provider in access_data_providers:
            # Check if the access is valid
            if access_data_provider.Intervals.Count > 0:
                n_obs += 1
                reward += self.f_ri(n_obs , n_obs, n_obs)

        return reward
    
    def f_ri(self, event_pos: tuple[float, float], satellite_pos: tuple[float, float], alt: float):
        """
        Function rewarding a certain observation.
        """
        reward = 0

        return reward
