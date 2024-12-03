import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agi.stk12.stkobjects import *

# Constants
RT = 6371  # radius of the earth in km

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

    def is_in_time_range(self, first: str, last: str, current: str):
        """
        Check if the current date is in the time range.
        """
        first = self.simplify_date(first)
        last = self.simplify_date(last)
        current = self.simplify_date(current)
        return self.num_of_date(first) <= self.num_of_date(current) <= self.num_of_date(last)
    
    def is_newer_than(self, first: str, second: str):
        """
        Check if the first date is older than the second date.
        """
        first = self.simplify_date(first)
        second = self.simplify_date(second)
        return self.num_of_date(first) > self.num_of_date(second)

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
        
    def get_date_after(self, delta_time, current_date, return_simplified: bool=False):
        """
        Return the date after the given number of days.
        """
        # Get the current date
        day, month, year, hour, minute, second = self.simplify_date(current_date).split(" ")
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
        
    def get_current_date_after(self, delta_time, return_simplified: bool=False):
        """
        Return the date after the given number of time.
        """
        return self.get_date_after(delta_time, self.current_date, return_simplified)
        
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

class AttitudeManager():
    """
    Class to manage the attitude of the model. Functions:
    - get_item: return the value of the item.
    - update_pitch: update the pitch of the agent within the boundaries.
    - update_roll: update the roll of the agent within the boundaries.
    """
    def __init__(self, agent):
        self.class_name = "Attitude Manager"
        self.current_pitch = agent["initial_pitch"]
        self.current_roll = agent["initial_roll"]
        self.max_slew = agent["max_slew_speed"]
        self.max_accel = agent["max_slew_accel"]
        if agent["attitude_align"] == "Nadir(Centric)":
            self.align_reference = "Nadir(Centric)"
            self.unallowed_angles = {"pitch": [90, -90]}
            self.constraint_reference = "Velocity"
            self.constraint_axes = "1 0 0"
        else:
            raise NotImplementedError("Invalid attitude alignment. Please use 'Nadir(Centric)'.")

    def get_item(self, name):
        """
        Return the value of the item.
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise ValueError(f"Variable {name} does not exist in the class. Check the configuration file.")
        
    def get_transition_command(self, satellite, time):
        """
        Return the transition command of the agent.
        """
        return f"""AddAttitude {satellite.Path} Profile MyProfile "{time}" VariableTimeSlew Mode Constrained SlewSegmentTiming Earliest SlewType 2ndOrderSpline RateMagnitude {self.max_slew} RateAxisX Off RateAxisY Off RateAxisZ Off AccelMagnitude {self.max_accel} AccelAxisX Off AccelAxisY Off AccelAxisZ Off"""

    def get_orientation_command(self, satellite, time):
        """
        Return the orientation command of the agent.
        """
        return f"""AddAttitude {satellite.Path} Profile MyProfile "{time}" AlignConstrain PR {self.current_pitch} {self.current_roll} "Satellite/{satellite.InstanceName} {self.align_reference}" Axis {self.constraint_axes} "Satellite/{satellite.InstanceName} {self.constraint_reference}" """

    def update_pitch(self, delta_pitch):
        """
        Update the pitch of the agent within the boundaries.
        """
        self.current_pitch += delta_pitch

        if "pitch" in self.unallowed_angles.keys() and self.current_pitch in self.unallowed_angles["pitch"]:
            self.current_pitch += 1e-4

        # Correct the pitch if out of boundaries
        if self.current_pitch > 90:
            self.current_pitch -= 180
            self.current_roll += 180
        elif self.current_pitch < -90:
            self.current_pitch += 180
            self.current_roll += 180

        return self.current_pitch
    
    def update_roll(self, delta_roll):
        """
        Update the roll of the agent within the boundaries.
        """
        self.current_roll += delta_roll

        if "roll" in self.unallowed_angles.keys() and self.current_roll in self.unallowed_angles["roll"]:
            self.current_roll += 1e-4

        # Correct the roll if out of boundaries
        while self.current_roll > 180 or self.current_roll < -180:
            if self.current_roll > 180:
                self.current_roll -= 360
            elif self.current_roll < -180:
                self.current_roll += 360

        return self.current_roll

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
        self.max_slew = agent["max_sensor_slew"]
        self.current_azimuth = agent["initial_azimuth"]
        self.current_elevation = agent["initial_elevation"]

        if hasattr(agent, "resolution"):
            self.resolution = agent["resolution"]
        else:
            self.resolution = 0.1

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
        self.target_memory = 0

        # Iterate over the states
        for state in self.states_features:
            if state in agent.keys():
                self.state[state] = agent[state]
            else:
                self.state[state] = None
            
            if state.startswith("lat_"):
                self.target_memory += 1
            
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
        if self.agent_config["coordinate_system"] == "Classical":
            for key in ["a", "e", "i", "raan", "aop", "ta"]:
                self.update_state(key, orbital_elements.DataSets.GetDataSetByName(self.long_name_of(key)).GetValues()[0])
        elif self.agent_config["coordinate_system"] == "Cartesian":
            for key in ["x", "y", "z", "vx", "vy", "vz"]:
                self.update_state(key, orbital_elements.DataSets.GetDataSetByName(self.long_name_of(key)).GetValues()[0])
        else:
            raise ValueError("Invalid coordinate system. Please use 'Classical' or 'Cartesian'.")
    
    def update_attitude_state(self, pitch, roll):
        """
        Update the attitude state of the agent.
        """
        if "pitch" in self.state.keys():
            self.update_state("pitch", pitch)
        if "roll" in self.state.keys():
            self.update_state("roll", roll)
        
    def update_sensor_state(self, az, el):
        """
        Update the sensor state of the agent.
        """
        if "az" in self.state.keys():
            self.update_state("az", az)
        if "el" in self.state.keys():
            self.update_state("el", el)
        
    def update_detic_state(self, satellite, time):
        """
        Update the LLA state of the agent.
        """
        detic_dataset = satellite.DataProviders.Item("LLA State").Group.Item(1).ExecSingle(time).DataSets
        detic_lat = detic_dataset.GetDataSetByName("Lat").GetValues()[0] # Group Items --> 0: TrueOfDateRotating, 1: Fixed
        detic_lon = detic_dataset.GetDataSetByName("Lon").GetValues()[0]
        detic_alt = detic_dataset.GetDataSetByName("Alt").GetValues()[0]
        if "detic_lat" in self.state.keys():
            self.update_state("detic_lat", detic_lat)
        if "detic_lon" in self.state.keys():
            self.update_state("detic_lon", detic_lon)
        if "detic_alt" in self.state.keys():
            self.update_state("detic_alt", detic_alt)

    def update_target_memory(self, preferred_zones, all_zones):
        """
        Update the target memory of the agent.
        """
        if not preferred_zones.empty:
            n_selected = self.target_memory if self.target_memory <= preferred_zones.shape[0] else preferred_zones.shape[0]
            seeking_zones = preferred_zones.sample(n_selected, ignore_index=True)
            if n_selected != self.target_memory:
                seeking_zones = pd.concat([seeking_zones, all_zones.sample(self.target_memory - n_selected, ignore_index=True)], ignore_index=True)
        else:
            seeking_zones = all_zones.sample(self.target_memory, ignore_index=True)

        for i in range(self.target_memory):
            self.update_state(f"lat_{i+1}", seeking_zones["lat [deg]"][i])
            self.update_state(f"lon_{i+1}", seeking_zones["lon [deg]"][i])
            self.update_state(f"priority_{i+1}", seeking_zones["priority"][i])
        
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

class TargetManager():
    """
    Class to manage the targets of the model.
    """
    def __init__(self, start_time, n_of_visible_targets):
        self.class_name = "Target Manager"
        self.df = pd.DataFrame()
        self.df_last = pd.DataFrame()
        self.date_mg = DateManager(start_time, start_time)
        self.newest_time = start_time
        self.n_of_visible_targets = n_of_visible_targets

    def n_of_zones_to_add(self, time):
        """
        Update the zones of the dataframe.
        """
        if self.date_mg.is_newer_than(time, self.newest_time):
            self.newest_time = time
            self.df_last = self.df_last[self.df_last["numeric_end_date"] >= self.date_mg.num_of_date(self.date_mg.simplify_date(time))]
            return self.n_of_visible_targets - self.df_last.shape[0]
            
        return 0
    
    def erase_zone(self, name: str):
        """
        Erase a zone from the dataframe.
        """
        self.df = self.df[self.df["name"] != name]
        self.df_last = self.df_last[self.df_last["name"] != name]

    def append_zone(self, name: str, target, type: str, lat: float, lon: float, priority: float, start_time: str, end_time: str, n_obs: int=0, last_seen: str="", erase_first: bool=False):
        """
        Append a zone to the dataframe.
        """
        self.df = pd.concat([self.df, pd.DataFrame({"name": [name], "object": [target], "type": [type], "lat [deg]": [lat], "lon [deg]": [lon], "priority": [priority], "start_time": [start_time], "end_time": [end_time], "numeric_start_date": [self.date_mg.num_of_date(self.date_mg.simplify_date(start_time))], "numeric_end_date": [self.date_mg.num_of_date(self.date_mg.simplify_date(end_time))], "n_obs": [n_obs], "last seen": [last_seen]})], ignore_index=True)
        self.df_last = pd.concat([self.df_last, pd.DataFrame({"name": [name], "object": [target], "type": [type], "lat [deg]": [lat], "lon [deg]": [lon], "priority": [priority], "start_time": [start_time], "end_time": [end_time], "numeric_start_date": [self.date_mg.num_of_date(self.date_mg.simplify_date(start_time))], "numeric_end_date": [self.date_mg.num_of_date(self.date_mg.simplify_date(end_time))], "n_obs": [n_obs], "last seen": [last_seen]})], ignore_index=True)

    def plus_one_obs(self, name: str):
        """
        Increase the number of observations of the zone by one.
        """
        self.df.loc[self.df["name"] == name, "n_obs"] += 1

    def update_last_seen(self, name: str, date: str):
        """
        Update the last seen date of the zone.
        """
        self.df.loc[self.df["name"] == name, "last seen"] = date

    def get_n_obs(self, name: str):
        """
        Return the number of observations of the zone.
        """
        return self.get_zone_by_name(name)["n_obs"].values[0]
    
    def get_last_seen(self, name: str):
        """
        Return the last seen date of the zone.
        """
        return self.get_zone_by_name(name)["last seen"].values[0]
    
    def get_priority(self, name: str):
        """
        Return the priority of the zone.
        """
        return self.get_zone_by_name(name)["priority"].values[0]

    def get_zone_by_row(self, i: int) -> pd.DataFrame:
        """
        Return the zone by row.
        """
        return pd.DataFrame(self.df.iloc[i]).T
    
    def get_zone_by_name(self, name: str) -> pd.DataFrame:
        """
        Return the zone by name.
        """
        zone = self.df[self.df["name"] == name]

        if zone.empty:
            raise ValueError(f"Zone {name} not found in the dataframe.")
        elif zone.shape[0] > 1:
            raise ValueError(f"Zone {name} found multiple times in the dataframe.")
        
        return zone
    
    def get_FoR_window_df(self, satellite: IAgStkObject, date_mg: DateManager, margin_pct: float=10) -> pd.DataFrame:
        """
        Return the Field of Regard (FoR) window dataframe.
        """
        # Get the window of targets
        FoR_window_df = self.df[self.df["numeric_end_date"] >= date_mg.num_of_date(date_mg.simplify_date(date_mg.last_date))]
        FoR_window_df = FoR_window_df[FoR_window_df["numeric_start_date"] <= date_mg.num_of_date(date_mg.simplify_date(date_mg.current_date))]

        # Get the satellite's geodetic coordinates (deg, deg, km)
        detic_dataset = satellite.DataProviders.Item("LLA State").Group.Item(1).ExecSingle(date_mg.last_date).DataSets
        detic_lat = detic_dataset.GetDataSetByName("Lat").GetValues()[0]
        detic_lon = detic_dataset.GetDataSetByName("Lon").GetValues()[0]
        detic_alt = detic_dataset.GetDataSetByName("Alt").GetValues()[0]

        detic_dataset = satellite.DataProviders.Item("LLA State").Group.Item(1).ExecSingle(date_mg.current_date).DataSets
        detic_lat = (detic_lat + detic_dataset.GetDataSetByName("Lat").GetValues()[0])/2
        detic_lon = (detic_lon + detic_dataset.GetDataSetByName("Lon").GetValues()[0])/2
        detic_alt = (detic_alt + detic_dataset.GetDataSetByName("Alt").GetValues()[0])/2

        # Find the distance between the satellite's nadir and the targets
        FoR_window_df["distance"] = FoR_window_df.apply(lambda row: self.haversine(detic_lat, detic_lon, row["lat [deg]"], row["lon [deg]"]), axis=1)

        # Calculate the field of regard (km)
        D_FoR = self.calculate_D_FoR(detic_alt) # distance of the field of regard on the ground

        # Filter the targets based on the field of regard
        FoR_window_df = FoR_window_df[FoR_window_df["distance"] <= D_FoR * (1 + margin_pct/100)] # 10% margin

        return FoR_window_df
    
    def get_FoR_zones(self, satellite_lat: float, satellite_lon: float, altitude: float, FoR: float):
        """
        Return the zones within the Field of Regard (FoR).
        """
        zones = []
        for i in range(self.df.shape[0]):
            zone = self.get_zone_by_row(i)
            zone_lat = zone["lat [deg]"].values[0]
            zone_lon = zone["lon [deg]"].values[0]
            distance = self.haversine(satellite_lat, satellite_lon, zone_lat, zone_lon)
            if distance <= FoR:
                zones.append(zone)
        
        return zones
    
    def calculate_D_FoR(self, altitude: float):
        """
        Calculate the distance of the Field of Regard (FoR).
        """
        return RT * np.arccos(RT / (RT + altitude))
    
    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points on the earth (specified in decimal degrees).
        """
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return RT * c

class Rewarder():
    """
    Class to manage the reward of the model. Functions:
    - calculate_reward: return the reward of the state-action pair.
    - f_ri: return the reward of the observation.
    - f_theta: return the reward of the angle between the event and the satellite.
    - f_reobs: return the reward of the reobservation of the same event.
    """
    def __init__(self, agents_config, target_mg: TargetManager):
        self.class_name = "Rewarder"
        self.seen_events = []
        self.target_mg = target_mg
        self.agents_config = agents_config

    def calculate_reward(self, data_providers, delta_time: float, date_mg: DateManager, sensor_mg: SensorManager, features_mg: FeaturesManager):
        """
        Return the reward of the state-action pair given the proper data providers (acces and aer).
        """
        reward = 0

        reward += self.slew_constraint(delta_time, sensor_mg, features_mg)

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

                        # Get the zone information
                        zone_n_obs = self.target_mg.get_n_obs(event_name)
                        zone_last_seen = self.target_mg.get_last_seen(event_name)
                        zone_priority = self.target_mg.get_priority(event_name)

                        # Check is long enough based on min_duration
                        if (date_mg.num_of_date(date_mg.simplify_date(stop_time[j])) - date_mg.num_of_date(date_mg.simplify_date(start_time[j]))) > min_duration:                            
                            # Check if the event has been seen before and how many times
                            if zone_n_obs != 0:
                                # Check the number of observations is not negative
                                if zone_n_obs < 0:
                                    raise ValueError("Number of observations cannot be negative.")
                                
                                # Change the last seen date format
                                last_seen = date_mg.num_of_date(date_mg.simplify_date(zone_last_seen))
                                self.target_mg.update_last_seen(event_name, stop_time[j])

                                # This filters concatenated observations (which indeed are one observation)
                                if (last_seen - min_duration) >= date_mg.num_of_date(date_mg.simplify_date(start_time[j])): # min_duration added because of added in .Exec() too
                                    break

                                self.target_mg.plus_one_obs(event_name)
                                n_obs = self.target_mg.get_n_obs(event_name)
                                ri = self.f_ri(zone_priority, max_zen_angle, n_obs)
                                reward += ri
                                print(f"Observed {event_name} with zenith {max_zen_angle:0.2f}º and reward of {ri:0.4f} (total of {reward:0.4f}).")
                            else:
                                self.target_mg.plus_one_obs(event_name)
                                self.target_mg.update_last_seen(event_name, stop_time[j])
                                ri = self.f_ri(zone_priority, max_zen_angle, 1)
                                reward += ri
                                print(f"First observed {event_name} with zenith {max_zen_angle:0.2f}º and reward of {ri:0.4f} (total of {reward:0.4f}).")

                        print()
        
        return reward
    
    def f_ri(self, priority: float, max_zen_angle: float, n_obs: int):
        """
        Function rewarding a certain observation. Inputs given in the form of tuples.
        - event_pos: tuple of the event position (latitude, longitude, altitude).
        - satellite_pos: tuple of the satellite position (latitude, longitude, altitude).
        - event_name: name of the event target ('target1', for instance).
        - max_zen_angle: maximum elevation angle of the event.
        """
        # Target-specific profit
        profit = self.agents_config["priority_weight"] * priority

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
    
    def slew_constraint(self, delta_time, sensor_mg: SensorManager, features_mg: FeaturesManager):
        """
        Function penalizing the slew rates. Inputs given in the form of a list.
        """
        # Get the slew rates
        r = 0
        for diff in features_mg.action.keys():
            if diff in ["d_az", "d_el"]:
                slew_rate = features_mg.action[diff] / delta_time
                slew_rate = abs(slew_rate)
                r += -10 if slew_rate > sensor_mg.max_slew else 5
            
            # if diff in ["d_pitch", "d_roll"]:
            #     slew_rate = features_mg.action[diff] / delta_time
            #     r += -10 if slew_rate > sensor_mg.max_slew else 5

        return r
    
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
    
    def plot_rewards_smoothed(self, window_size: int=0):
        """
        Plot the rewards within smoothed windows of size window_size.
        """
        if self.rewards.empty:
            raise ValueError("No rewards to plot.")
        
        window_size = self.correct_window_size(window_size)
        
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
        
        window_size = self.correct_window_size(window_size)
        
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

    def plot_all(self, window_size: int=0):
        """
        Plot all the rewards.
        """
        self.plot_rewards()
        self.plot_rewards_smoothed(window_size=window_size)
        self.plot_cumulative_rewards()
        self.plot_cumulative_rewards_smoothed_per_steps(window_size=window_size)

    def correct_window_size(self, window_size: int):
        """
        Correct the window size if it is negative.
        """
        # Correct the window size
        if window_size == 0:
            window_size = int(len(self.rewards) / 10)
        
        # Check the window size
        if window_size < 1:
            window_size = 1

        return window_size
