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
    - __init__: sets the current date and time of the simulation.
    - simplify_date: returns the date in a simplified, more readable format.
    - fancy_date: returns the date in the fancy stk-used format.
    - month_to_number: returns the number of the month.
    - number_to_month: returns the month of the number.
    - number_of_days_in: returns the number of days in the month.
    - get_date_after: returns the date after a given time increment.
    """
    def __init__(self, current_date):
        self.class_type = "Date Manager"
        self.current_date = current_date
        self.current_simplified_date = self.simplify_date(current_date) # all in numbers concatenated in a string

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

    def get_date_after(self, delta_time):
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
        return self.current_date

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
    - get_properties: return the properties of the agent.
    - update_property: update the property of the agent if it exists.
    """
    def __init__(self, agent):
        self.class_type = "Features Manager"
        self.get_properties(agent)

    def get_properties(self, agent):
        """
        Return the properties of the agent.
        """
        self.states = agent["states_features"]
        self.actions = agent["actions_features"]

        # Iterate over the states
        for key in self.states:
            if key in agent.keys():
                setattr(self, key, agent[key])
            else:
                raise ValueError(f"In the configuration file, {key} appears in the states_features but not in the initial_state.")
            
    def update_property(self, property_name, value):
        """
        Update the property of the agent if it exists.
        """
        if hasattr(self, property_name):
            setattr(self, property_name, value)

