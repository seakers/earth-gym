import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R

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
        days_that_month = self.number_of_days_in_month(self.number_to_month(month), year)
        month += int(day / (days_that_month + 1))
        day = day % (days_that_month + 1)
        day = day if day != 0 else 1
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
        seconds = hour * 3600 + minute * 60 + second
        day_of_the_year = self.day_of_the_year(day, month, year)
        seconds += (day_of_the_year - 1) * 86400
        seconds += self.seconds_before_year(year)

        return seconds
    
    def day_of_the_year(self, day: int, month: int, year: int) -> int:
        """
        Return which day in the year is the one given.
        """
        # Calculate the days
        day_of_the_year = day
        for i in range(month):
            day_of_the_year += self.number_of_days_in_month(self.number_to_month(i), year)

        return day_of_the_year
    
    def seconds_before_year(self, year: int) -> float:
        """
        Return the number of seconds before the given year.
        """
        seconds = 0
        year -= 1

        if year / 4 == 0:
            seconds += 1/4 * year * 31622400
            seconds += 3/4 * year * 31536000
        else:
            add_years = year % 4
            year -= add_years
            seconds += 1/4 * year * 31622400
            seconds += 3/4 * year * 31536000
            seconds += add_years * 31536000

        return seconds

class AttitudeManager():
    """
    Class to manage the attitude of the model.
    """
    def __init__(self, agent):
        self.class_name = "Attitude Manager"
        self.current_pitch = agent["initial_pitch"]
        self.current_roll = agent["initial_roll"]
        self.max_slew = agent["max_slew_speed"]
        self.max_accel = agent["max_slew_accel"]
        if agent["attitude_align"] == "Nadir(Centric)":
            self.align_reference = "Nadir(Centric)"
            self.angle_domains = {"pitch": [-90, 90], "roll": [-180, 180]}
            self.unallowed_angles = {"pitch": [90, -90]}
            self.constraint_reference = "Velocity"
            self.constraint_axes = "1 0 0"
        else:
            raise NotImplementedError("Invalid attitude alignment. Please use 'Nadir(Centric)'.")
        
        self.previous_orientation_cmd = None

    def get_item(self, name):
        """
        Return the value of the item.
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise ValueError(f"Variable {name} does not exist in the class. Check the configuration file.")
        
    def get_clear_data_command(self, satellite):
        """
        Return the clear data command of the agent.
        """
        return f"SetAttitude {satellite.Path} ClearData AllProfiles"
    
    def get_segments_command(self, satellite):
        """
        Return the segments command of the agent.
        """
        return f"""GetAttitude {satellite.Path} Segments"""
        
    def get_transition_command(self, satellite, time):
        """
        Return the transition command of the agent.
        """
        return f"""AddAttitude {satellite.Path} Profile MyProfile "{time}" VariableTimeSlew Mode Constrained SlewSegmentTiming Earliest SlewType 2ndOrderSpline RateMagnitude {self.max_slew} RateAxisX Off RateAxisY Off RateAxisZ Off AccelMagnitude {self.max_accel} AccelAxisX Off AccelAxisY Off AccelAxisZ Off"""

    def get_new_orientation_command(self, satellite, time):
        """
        Return the orientation command of the agent.
        """
        self.previous_orientation_cmd = f"""AddAttitude {satellite.Path} Profile MyProfile "{time}" AlignConstrain PR {self.current_pitch} {self.current_roll} "Satellite/{satellite.InstanceName} {self.align_reference}" Axis {self.constraint_axes} "Satellite/{satellite.InstanceName} {self.constraint_reference}" """
        return self.previous_orientation_cmd
    
    def get_previous_orientation_command(self):
        """
        Return the previous orientation command of the agent.
        """
        return self.previous_orientation_cmd
    
    def update_roll_pitch(self, delta_pitch, delta_roll):
        """
        Update the current pitch and roll by applying incremental rotations.
        Order of rotations: roll, pitch, yaw.
        Yaw is assumed to be zero.
        
        Parameters:
        delta_pitch: incremental change in pitch (degrees)
        delta_roll: incremental change in roll (degrees)
        
        Returns:
        new_pitch: updated pitch (degrees)
        new_roll: updated roll (degrees)
        """
        # Gather the current orientation
        current_pitch = self.current_pitch
        current_roll = self.current_roll

        # Represent current orientation with yaw=0 using a full 3D Euler representation.
        # The sequence 'xyz' means: rotation about x (roll), then y (pitch), then z (yaw).
        current_rot = R.from_euler('xyz', [current_roll, current_pitch, 0], degrees=True)
        
        # Create the incremental rotation; again, yaw change is zero.
        incremental_rot = R.from_euler('xyz', [delta_roll, delta_pitch, 0], degrees=True)
        
        # Apply the incremental rotation (body-fixed update).
        new_rot = current_rot * incremental_rot
        
        # Convert back to Euler angles in the same sequence.
        new_euler = new_rot.as_euler('xyz', degrees=True)
        
        # new_euler contains [new_roll, new_pitch, new_yaw]. We ignore new_yaw.
        new_roll = new_euler[0]
        new_pitch = new_euler[1]

        # Set the current pitch and roll to the new values.
        self.current_pitch = new_pitch
        self.current_roll = new_roll

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
        self.aux_state = {"detic_lat": None, "detic_lon": None, "detic_alt": None}
        self.detic_log = {"max_samples": 50, "all_time_counter": 0, "prev_lat": [], "prev_lon": [], "prev_alt": [], "prev_times": np.array([]), "curr_lat": None, "curr_lon": None, "curr_alt": None, "counter": 0, "curr_step_gap": 1, "prev_step_gap": 1}

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
    
    def get_state(self, name: str=None):
        """
        Return the properties of the agent.
        """
        return self.state if name is None else self.state[name]
                   
    def update_state(self, name, value):
        """
        Update the state properties of the agent.
        """
        if name in self.state.keys():
            self.state[name] = value
        else:
            raise ValueError(f"Variable {name} does not exist in the class.")
        
    def update_aux_state(self, name, value):
        """
        Update the auxiliar state properties of the agent.
        """
        if name in self.aux_state.keys():
            self.aux_state[name] = value
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
        
    def update_detic_state(self):
        """
        Update the LLA state of the agent.
        """
        detic_lat = self.get_aux_state("detic_lat")
        detic_lon = self.get_aux_state("detic_lon")
        detic_alt = self.get_aux_state("detic_alt")

        if "detic_lat" in self.state.keys():
            self.update_state("detic_lat", detic_lat)
        if "detic_lon" in self.state.keys():
            self.update_state("detic_lon", detic_lon)
        if "detic_alt" in self.state.keys():
            self.update_state("detic_alt", detic_alt)

    def update_entire_aux_state(self, satellite: IAgStkObject, target_mg, time: str):
        """
        Update the auxiliar state of the agent.
        """
        detic_lat, detic_lon, detic_alt = self.get_LLA_state(satellite, target_mg, time)

        if "detic_lat" in self.aux_state.keys():
            self.update_aux_state("detic_lat", detic_lat)
        if "detic_lon" in self.aux_state.keys():
            self.update_aux_state("detic_lon", detic_lon)
        if "detic_alt" in self.aux_state.keys():
            self.update_aux_state("detic_alt", detic_alt)
        
    def get_aux_state(self, name: str=None):
        """
        Get the auxiliar state of the agent.
        """
        return self.aux_state if name is None else self.aux_state[name]
    
    def get_LLA_state(self, satellite: IAgStkObject, target_mg, time: str):
        """
        Get the LLA state of the agent. Interpolation is used to get the LLA state, given the high computational cost.
        """
        if self.detic_log["counter"] == 0 or self.detic_log["counter"] >= self.detic_log["curr_step_gap"] or len(self.detic_log["prev_times"]) <= 2 or abs(self.detic_log["prev_lon"][-1] - self.detic_log["prev_lon"][-2]) > 180: # it takes high computational cost to get the LLA state
            # print("Getting LLA state...")
            # Get the LLA state of the agent
            detic_dataset = satellite.DataProviders.Item("LLA State").Group.Item(1).ExecSingle(time).DataSets
            detic_lat = detic_dataset.GetDataSetByName("Lat").GetValues()[0] # Group Items --> 0: TrueOfDateRotating, 1: Fixed
            detic_lon = detic_dataset.GetDataSetByName("Lon").GetValues()[0]
            detic_alt = detic_dataset.GetDataSetByName("Alt").GetValues()[0]

            # Store the current LLA state
            self.detic_log["curr_lat"] = detic_lat
            self.detic_log["curr_lon"] = detic_lon
            self.detic_log["curr_alt"] = detic_alt
            self.detic_log["counter"] = 1

            # Correct discontinuities in the longitude
            if len(self.detic_log["prev_lon"]) >= 2 and abs(self.detic_log["prev_lon"][-1] - self.detic_log["prev_lon"][-2]) > 180:
                self.detic_log["prev_lon"] = self.detic_log["prev_lon"][-1:]
                self.detic_log["prev_lat"] = self.detic_log["prev_lat"][-1:]
                self.detic_log["prev_alt"] = self.detic_log["prev_alt"][-1:]
                self.detic_log["prev_times"] = self.detic_log["prev_times"][-1:]

            # Save the previous LLA state
            self.detic_log["prev_lat"] += [self.detic_log["curr_lat"]]
            self.detic_log["prev_lon"] += [self.detic_log["curr_lon"]]
            self.detic_log["prev_alt"] += [self.detic_log["curr_alt"]]
            self.detic_log["prev_times"] = np.concatenate([self.detic_log["prev_times"], np.array([self.detic_log["all_time_counter"]])])

            # Limit the number of samples used for interpolation
            self.detic_log["prev_lat"] = self.detic_log["prev_lat"][-self.detic_log["max_samples"]:]
            self.detic_log["prev_lon"] = self.detic_log["prev_lon"][-self.detic_log["max_samples"]:]
            self.detic_log["prev_alt"] = self.detic_log["prev_alt"][-self.detic_log["max_samples"]:]
            self.detic_log["prev_times"] = self.detic_log["prev_times"][-self.detic_log["max_samples"]:]

            # Update the step gap progressively
            if len(self.detic_log["prev_times"]) <= 2:
                self.detic_log["prev_step_gap"] = self.detic_log["curr_step_gap"] = 1
            else:
                self.detic_log["prev_step_gap"] = self.detic_log["curr_step_gap"]
                self.detic_log["curr_step_gap"] = 2 * self.detic_log["curr_step_gap"]
                self.detic_log["curr_step_gap"] = self.detic_log["curr_step_gap"] if self.detic_log["curr_step_gap"] <= self.agent_config["LLA_step_gap"] else self.agent_config["LLA_step_gap"]

        else:
            # Reset all the time points
            self.detic_log["all_time_counter"] -= self.detic_log["prev_times"][0]
            self.detic_log["prev_times"] -= self.detic_log["prev_times"][0]

            # Perform interpolation with scipy
            cs_lat = PchipInterpolator(self.detic_log["prev_times"], self.detic_log["prev_lat"])
            cs_lon = PchipInterpolator(self.detic_log["prev_times"], self.detic_log["prev_lon"])
            cs_alt = PchipInterpolator(self.detic_log["prev_times"], self.detic_log["prev_alt"])

            # Get the interpolated LLA state at the current time
            detic_lat = cs_lat(self.detic_log["all_time_counter"])
            detic_lon = cs_lon(self.detic_log["all_time_counter"])
            detic_alt = cs_alt(self.detic_log["all_time_counter"])

            # Correct latitude if it exceeds its physical boundaries
            if detic_lat > 90:
                detic_lat = 180 - detic_lat
                detic_lon += 180  # adjust longitude when crossing the north pole
            elif detic_lat < -90:
                detic_lat = -180 - detic_lat
                detic_lon += 180  # adjust longitude when crossing the south pole

            # Normalize the longitude to be within [-180, 180]
            detic_lon = ((detic_lon + 180) % 360) - 180

            ############## DEBUGGING ############## Use to check if the interpolation is correct
            # detic_dataset = satellite.DataProviders.Item("LLA State").Group.Item(1).ExecSingle(time).DataSets
            # adetic_lat = detic_dataset.GetDataSetByName("Lat").GetValues()[0] # Group Items --> 0: TrueOfDateRotating, 1: Fixed
            # adetic_lon = detic_dataset.GetDataSetByName("Lon").GetValues()[0]
            # adetic_alt = detic_dataset.GetDataSetByName("Alt").GetValues()[0]
            # print("Real LLA state: ", adetic_lat, adetic_lon, adetic_alt)
            # print(f"PCHIP: {detic_lat}, {detic_lon}, {detic_alt}. Counter: {self.detic_log['counter']}/{self.detic_log['curr_step_gap']}.")
            # diff = abs(detic_lat - adetic_lat) % 360
            # diff = min(diff, 360 - diff)
            # if diff > 10:
            #     raise ValueError("Longitude is not being interpolated correctly.")
            ############## DEBUGGING ##############

            self.detic_log["counter"] += 1

        self.detic_log["all_time_counter"] += 1

        return float(detic_lat), float(detic_lon), float(detic_alt)
    
    # def get_LLA_state_OLD(self, satellite: IAgStkObject, target_mg, time: str):
    #     """
    #     Get the LLA state of the agent. Interpolation is used to get the LLA state, given the high computational cost.
    #     """
    #     if self.detic_log["counter"] == 0 or self.detic_log["counter"] >= self.detic_log["curr_step_gap"]: # it takes high computational cost to get the LLA state
    #         # Save the previous LLA state if it is not the first time
    #         if self.detic_log["counter"] != 0:
    #             self.detic_log["prev_lat"] = self.detic_log["curr_lat"]
    #             self.detic_log["prev_lon"] = self.detic_log["curr_lon"]
    #             self.detic_log["prev_alt"] = self.detic_log["curr_alt"]

    #         # Get the LLA state of the agent
    #         detic_dataset = satellite.DataProviders.Item("LLA State").Group.Item(1).ExecSingle(time).DataSets
    #         detic_lat = detic_dataset.GetDataSetByName("Lat").GetValues()[0] # Group Items --> 0: TrueOfDateRotating, 1: Fixed
    #         detic_lon = detic_dataset.GetDataSetByName("Lon").GetValues()[0]
    #         detic_alt = detic_dataset.GetDataSetByName("Alt").GetValues()[0]

    #         # Store the current LLA state
    #         self.detic_log["curr_lat"] = detic_lat
    #         self.detic_log["curr_lon"] = detic_lon
    #         self.detic_log["curr_alt"] = detic_alt
    #         self.detic_log["counter"] = 1

    #         # Update the step gap progressively
    #         self.detic_log["prev_step_gap"] = self.detic_log["curr_step_gap"]
    #         self.detic_log["curr_step_gap"] = 2 * self.detic_log["curr_step_gap"]
    #         self.detic_log["curr_step_gap"] = self.detic_log["curr_step_gap"] if self.detic_log["curr_step_gap"] <= self.agent_config["LLA_step_gap"] else self.agent_config["LLA_step_gap"]

    #     else:
    #         detic_lat = self.detic_log["curr_lat"]
    #         detic_lon = self.detic_log["curr_lon"]
    #         detic_alt = self.detic_log["curr_alt"]

    #         # If there are previous LLA states, use them to interpolate the current LLA state
    #         if self.detic_log["prev_lat"] != None:
    #             prev_detic_lat = self.detic_log["prev_lat"]
    #             prev_detic_lon = self.detic_log["prev_lon"]
    #             prev_detic_alt = self.detic_log["prev_alt"]

    #             fraction = self.detic_log["counter"] / self.detic_log["prev_step_gap"]
    #             detic_lat, detic_lon, detic_alt = self.interpolate_LLA_state(prev_detic_lat, prev_detic_lon, prev_detic_alt, detic_lat, detic_lon, detic_alt, target_mg, fraction)

    #         self.detic_log["counter"] += 1

    #     return detic_lat, detic_lon, detic_alt
    
    # def interpolate_LLA_state(self, lat_0: float, lon_0: float, alt_0: float, lat_1: float, lon_1: float, alt_1: float, target_mg, fraction: float):
    #     """
    #     Interpolate the LLA state of the agent, using circular and linear interpolation.
    #     """
    #     # ------------------------------------- CLARIFICATION -------------------------------------
    #     # The interpolation follows a circular path on the Earth's surface for coordinates
    #     #   and a linear path for the altitude. The interpolation is done as follows:
    #     # lat_2 = A * p_2 + B
    #     # lon_2 = C * p_2 + D
    #     # alt_2 = (alt_1 - alt_0) * (1 + fr) + alt_0
    #     # Where, by means of 2 points (4 equations):
    #     # A = (lat_1 - lat_0) / p1
    #     # B = lat_0
    #     # C = (lon_1 - lon_0) / p1
    #     # D = lon_0
    #     # And the distance in radians between the 2 points is:
    #     # p1 = haversine_angle(lat_0, lon_0, lat_1, lon_1)
    #     # p2 = p1 * (1 + fr)
    #     # -----------------------------------------------------------------------------------------
    #     p1 = target_mg.haversine_angle(lat_0, lon_0, lat_1, lon_1)
    #     p2 = p1 * (1 + fraction)
    #     lat = (lat_1 - lat_0) / p1 * p2 + lat_0
    #     lon = (lon_1 - lon_0) / p1 * p2 + lon_0
    #     alt = (alt_1 - alt_0) * (1 + fraction) + alt_0

    #     return lat, lon, alt

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
        self.date_mg = DateManager(start_time, start_time)
        self.newest_time = start_time
        self.n_of_visible_targets = n_of_visible_targets
        self.max_id = 0

    def n_of_zones_to_add(self, time):
        """
        Update the zones of the dataframe.
        """
        if self.date_mg.is_newer_than(time, self.newest_time):
            self.newest_time = time
            unloadable_df = self.get_unloadable_zones_before(self.date_mg.num_of_date(self.date_mg.simplify_date(time)))
            return self.n_of_visible_targets - self.df.shape[0] + unloadable_df.shape[0]
        
        return 0
    
    def get_unloadable_zones_before(self, lowest_current_date):
        """
        Return the unloadable zones of the dataframe. Lowest current date has to be in single number format.
        """
        return self.df[self.df["numeric_end_date"] < lowest_current_date]
    
    def unload_zones_before(self, lowest_current_date):
        """
        Unload the zones of the dataframe.
        """
        self.df = self.df[self.df["numeric_end_date"] >= lowest_current_date]
    
    def erase_zone(self, name: str):
        """
        Erase a zone from the dataframe.
        """
        self.df = self.df[self.df["name"] != name]

    def append_zone(self, name: str, target, type: str, lat: float, lon: float, priority: float, start_time: str, end_time: str, n_obs: int=0, last_seen: str="", erase_first: bool=False):
        """
        Append a zone to the dataframe.
        """
        self.df = pd.concat([self.df, pd.DataFrame({"name": [name], "object": [target], "type": [type], "lat [deg]": [lat], "lon [deg]": [lon], "priority": [priority], "start_time": [start_time], "end_time": [end_time], "numeric_start_date": [self.date_mg.num_of_date(self.date_mg.simplify_date(start_time))], "numeric_end_date": [self.date_mg.num_of_date(self.date_mg.simplify_date(end_time))], "n_obs": [n_obs], "last seen": [last_seen]})], ignore_index=True)
        self.max_id += 1

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
    
    def get_FoR_window_df(self, date_mg: DateManager, features_mg: FeaturesManager, margin_pct: float=10, return_D_FoR: bool=False) -> pd.DataFrame:
        """
        Return the Field of Regard (FoR) window dataframe.
        """
        # Get the window of targets
        FoR_window_df = self.df[self.df["numeric_end_date"] >= date_mg.num_of_date(date_mg.simplify_date(date_mg.last_date))]
        FoR_window_df = FoR_window_df[FoR_window_df["numeric_start_date"] <= date_mg.num_of_date(date_mg.simplify_date(date_mg.current_date))]

        # Get the satellite's geodetic coordinates (deg, deg, km)
        detic_lat = features_mg.get_aux_state("detic_lat")
        detic_lon = features_mg.get_aux_state("detic_lon")
        detic_alt = features_mg.get_aux_state("detic_alt")

        # Find the distance between the satellite's nadir and the targets
        FoR_window_df["distance"] = FoR_window_df.apply(lambda row: self.haversine(detic_lat, detic_lon, row["lat [deg]"], row["lon [deg]"]), axis=1)

        # Calculate the field of regard (km)
        D_FoR = self.calculate_D_FoR(detic_alt) # distance of the field of regard on the ground

        # Filter the targets based on the field of regard
        FoR_window_df = FoR_window_df[FoR_window_df["distance"] <= D_FoR * (1 + margin_pct/100)] # 10% margin

        if return_D_FoR:
            return FoR_window_df, D_FoR * (1 + margin_pct/100)
        else:
            return FoR_window_df

    def calculate_D_FoR(self, altitude: float):
        """
        Calculate the distance of the Field of Regard (FoR).
        """
        return RT * np.arccos(RT / (RT + altitude))
    
    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points on the earth (specified in decimal degrees).
        """
        # Haversine angle
        c = self.haversine_angle(lat1, lon1, lat2, lon2)

        return RT * c
    
    def haversine_angle(self, lat1, lon1, lat2, lon2):
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

        return c
    
class GridManager():
    """
    Class to manage the grid of the model.
    """
    def __init__(self):
        self.class_name = "Grid Manager"
        self.grids = {}

    def set_grid_data(self, nums: list, lats: list, lons: list):
        """
        Set the grid data (nums, lats, lons).
        """
        self.grid_data = {"nums": np.array(tuple(nums)), "lats [deg]": np.array(tuple(lats)), "lons [deg]": np.array(tuple(lons))}

    def get_seen_points(self, value_by_point_FoM: list):
        """
        Get the seen points of the grid. Retuns a list of tuples (lat, lon).
        """
        value_by_point_FoM = np.array(value_by_point_FoM)

        # Get the seen points
        seen_lats = self.grid_data["lats [deg]"][value_by_point_FoM > 0]
        seen_lons = self.grid_data["lons [deg]"][value_by_point_FoM > 0]
        return [(lat, lon) for lat, lon in zip(seen_lats, seen_lons)]

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

    def calculate_reward(self, data_providers, delta_time: float, grid_points_seen: list, FoR_window_df: pd.DataFrame, D_FoR: float, date_mg: DateManager, sensor_mg: SensorManager, features_mg: FeaturesManager, angle_domains: dict):
        """
        Return the reward of the state-action pair given the proper data providers (acces and aer).
        """
        reward = 0

        # Add the negative reward of the slew constraint
        reward += self.slew_constraint(delta_time, sensor_mg, features_mg, angle_domains) * self.agents_config["slew_weight"]

        if self.agents_config["use_grid"] and grid_points_seen is not None:
            # Add the grid rewards
            reward += self.grid_rewards(grid_points_seen, FoR_window_df, D_FoR)

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
                                if (last_seen - min_duration) < date_mg.num_of_date(date_mg.simplify_date(start_time[j])): # min_duration added because of added in .Exec() too
                                    self.target_mg.plus_one_obs(event_name)
                                    n_obs = self.target_mg.get_n_obs(event_name)
                                    ri = self.f_ri(zone_priority, max_zen_angle, n_obs)
                                    reward += ri
                                    print(f"Observed {event_name} with zenith {max_zen_angle:0.2f}º and reward of {ri:0.4f} (total of {reward:0.4f}).")
                                else:
                                    print(f"Observation of {event_name} not counted because it belongs to a previous one.")
                            else:
                                self.target_mg.plus_one_obs(event_name)
                                self.target_mg.update_last_seen(event_name, stop_time[j])
                                ri = self.f_ri(zone_priority, max_zen_angle, 1)
                                reward += ri
                                print(f"First observed {event_name} with zenith {max_zen_angle:0.2f}º and reward of {ri:0.4f} (total of {reward:0.4f}).")
                        else:
                            print(f"Observation of {event_name} has insufficient duration.")

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
        profit = priority**self.agents_config["priority_weight"]

        # Each of the value functions
        f_theta = self.f_theta(max_zen_angle)
        f_reobs = self.f_reobs(n_obs)

        return profit * f_reobs * f_theta

    def f_theta(self, max_zen_angle: float):
        """
        Function rewarding the angle between the event and the satellite. Inputs given in the form of a list.
        - el_angles: list of elevation angles.
        """
        return math.sin(math.radians(max_zen_angle))**self.agents_config["zenith_weight"] # the higher the better, the angle is in degrees
    
    def f_reobs(self, n_obs: int):
        """
        Function rewarding the reobservation of the same event. Inputs given in the form of an integer.
        - n_obs: number of times the event has been observed.
        """
        return (1 / n_obs**self.agents_config["reobs_decay"]) if n_obs > 0 else 1
    
    def slew_constraint(self, delta_time, sensor_mg: SensorManager, features_mg: FeaturesManager, angle_domains: dict):
        """
        Function penalizing the slew rates and the movement of the agent. Inputs given in the form of a list.
        """
        # Get the slew rates
        r = 0
        for diff in features_mg.action.keys():
            if diff in ["d_az", "d_el"]:
                slew_rate = features_mg.action[diff] / delta_time
                slew_rate = abs(slew_rate)
                r += -10 if slew_rate > sensor_mg.max_slew else 5

            # Penalize for simply moving
            key = diff.split("_")[1]
            movement = abs(features_mg.action[diff])
            domain = abs(angle_domains[key][1] - angle_domains[key][0])
            r += -movement/domain

        return r
    
    def grid_rewards(self, grid_points_seen: list, FoR_window_df: pd.DataFrame, D_FoR: float):
        """
        Function rewarding the grid points seen by the agent. Distances in km.
        """
        reward = 0
        # count = 0
        # for_count = 0

        # Reset the indices of the dataframe to ensure proper index access
        FoR_window_df.reset_index(drop=True, inplace=True)

        # Compute distance to each target
        for lat, lon in grid_points_seen:
            count += 1
            for i in range(FoR_window_df.shape[0]):
                distance = self.target_mg.haversine(lat, lon, FoR_window_df["lat [deg]"][i], FoR_window_df["lon [deg]"][i])
                if distance < D_FoR:
                    event_name = FoR_window_df["name"][i]

                    # Get the zone information
                    zone_n_obs = self.target_mg.get_n_obs(event_name)
                    zone_priority = self.target_mg.get_priority(event_name)

                    # Calculate cosine function
                    A = self.agents_config["grid_weight"] * zone_priority * self.f_reobs(zone_n_obs) # amplitude
                    B = np.pi / (2 * D_FoR) # argument
                    r = A * np.cos(B * distance)**self.agents_config["grid_decay"] # reward
                    r = r if r > 0 else 0

                    if r < 0: # security check
                        raise ValueError("Grid reward cannot be negative.")

                    reward += r
                    # for_count += 1
                    # print(f"Grid point {count} seen. Distance to {event_name}: {distance:0.2f} km. Reward: {r:0.4f} (total of {reward:0.4f}). DFoR: {D_FoR:0.2f} km.")

        # print(f"Grid points seen: {count}. Total of {for_count} calculations. Reward: {reward:0.4f}.")

        return reward
    
class Plotter():
    """
    Class to manage the plotting of the model
    """
    def __init__(self, out_folder_path: str="${workspaceFolder}/output"):
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
        plt.savefig(f"{self.out_folder_path}/rewards.png", dpi=500)
    
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
        plt.savefig(f"{self.out_folder_path}/rewards_smoothed.png", dpi=500)

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
        plt.savefig(f"{self.out_folder_path}/cumulative_rewards.png", dpi=500)

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
        plt.savefig(f"{self.out_folder_path}/cumulative_rewards_smoothed_per_steps.png", dpi=500)

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
