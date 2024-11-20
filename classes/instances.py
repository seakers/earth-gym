import json
import socket
import pandas as pd
from datetime import datetime

from agi.stk12.stkengine import STKEngine
from agi.stk12.stkobjects import *
from agi.stk12.stkutil import *

from classes.utils import *

class Gym():
    """
    Class to simulate a gym environment for training an agent.
    """
    def __init__(self, args):
        self.initialize_args(args)
        self.running = True

    def initialize_args(self, args):
        """
        Store arguments in class and check output folder exists.
        """
        self.host = args.host
        self.port = args.port

        # Check if input file is specified
        if args.conf is None:
            raise ValueError("Configuration file not specified in launch.json.")
        else:
            self.conf_file_path = args.conf

        # Check if events zones file is specified
        if args.evpt is None:
            raise ValueError("Events zones file not specified in launch.json.")
        else:
            self.evpt_file_path = args.evpt

        # Check if output folder is specified
        if args.out is None:
            raise ValueError("Output folder not specified in launch.json.")
        else:
            self.out_folder_path = args.out

    def initialize_world(self, file_path):
        """
        Initialize the agent with the given configuration.
        """
        with open(file_path, "r") as f:
            agents_config = json.load(f)

        if not agents_config:
            raise ValueError("Agent configuration is empty.")
        
        # Initialize the agent with the configuration in a simplified dictionary
        self.stk_env = STKEnvironment(DataFromJSON(agents_config, "configuration").get_dict(), self.evpt_file_path, self.out_folder_path)
    
    def get_next_state_and_reward(self, agent_id, action, delta_time):
        """
        Return the next state and reward of the agent.
        """
        return self.stk_env.step(agent_id, action, delta_time)
    
    def generate_output(self):
        """
        Perform final displays of the period.
        """
        # Plot all the reward graphics available
        self.stk_env.plotter.plot_all()
    
    def handle_request(self, request):
        """
        Deal with the request by calling different actions based on the incoming command. Options are:
        - get_next: Get the next state and reward based on the action.
        - shutdown: Shutdown the environment and generate the GIF.
        """
        # Load the data from the request
        request_data = json.loads(request)

        print(f"Received request: {request_data}")

        # Handle the request based on the command
        if request_data["command"] == "get_next":
            state, reward, done = self.get_next_state_and_reward(request_data["agent_id"], request_data["action"], request_data["delta_time"])
            return json.dumps({"state": state, "reward": reward, "done": done})
        elif request_data["command"] == "shutdown":
            self.stk_env.stk_root.SaveScenario()
            self.generate_output()
            self.running = False
            return json.dumps({"status": "shutdown_complete"})
        else:
            raise ValueError("Invalid command. Please use 'get_next' or 'shutdown'.")
    
    def start(self, host="localhost", port=5555):
        """
        Start to listen for incoming connections.
        """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(1)
        print("Gym environment started. Waiting for connections...")

        # Accept the connection
        conn, addr = server_socket.accept()
        print(f"Connected to: {addr}")

        # Initialize the world before starting the loop
        self.initialize_world(self.conf_file_path)
        
        # Loop to handle the requests
        while self.running:
            data = conn.recv(1024).decode()
            if not data:
                break
            response = self.handle_request(data)
            conn.sendall(response.encode())
        
        # Close the connection
        conn.close()
        server_socket.close()

class STKEnvironment():
    """
    Class to simulate the environment using STK.
    """
    def __init__(self, agents_config, evpt_file_path, out_folder_path):
        self.agents_config = agents_config
        self.evpt_file_path = evpt_file_path
        self.out_folder_path = out_folder_path
        stk_app = STKEngine().StartApplication(noGraphics=False)
        stk_root = stk_app.NewObjectRoot()
        self.stk_app = stk_app
        self.stk_root = stk_root
        self.scenario = self.build_scenario(self.stk_root, self.agents_config)
        self.satellites_tuples = []
        self.target_mg = TargetManager()
        self.rewarder = Rewarder(agents_config, self.target_mg)
        self.plotter = Plotter(out_folder_path)

        # Add the zones of interest
        self.draw_initial_event_zones(evpt_file_path, self.scenario)

        # Build the satellites by iterating over the agents
        for i, agent in enumerate(agents_config["agents"]):
            agent = DataFromJSON(agent, "agent").get_dict()
            satellite, sensor_mg, features_mg = self.build_satellite(agent, self.scenario, i)
            date_mg = DateManager(self.scenario.StartTime, self.scenario.StopTime)
            self.satellites_tuples.append((satellite, sensor_mg, features_mg, date_mg)) # append the satellite, its sensor manager and its date manager

    def build_scenario(self, root: AgStkObjectRoot, agents_config) -> IAgStkObject:
        """
        Build the scenario based on the agent configuration.
        """
        # Create a new scenario
        root.NewScenario(agents_config["scenario_name"])
        scenario = root.CurrentScenario
        scenario.StartTime = agents_config["start_time"]
        scenario.StopTime = agents_config["stop_time"]
        scenario.SetTimePeriod(scenario.StartTime, scenario.StopTime)
        root.Rewind()
        return scenario

    def build_satellite(self, agent, scenario: IAgStkObject, idx) -> tuple[IAgStkObject, SensorManager, FeaturesManager]:
        """
        Add a satellite to the scenario.
        """
        # Add a satellite with orbit propagation
        if hasattr(agent, "name"):
            satellite_name = agent["name"]
        else:
            satellite_name = f"MySatellite{idx}"

        # Create the satellite and set the propagator type depending on the configuration
        satellite = scenario.Children.New(AgESTKObjectType.eSatellite, satellite_name)
        self.set_propagator_type(satellite)
        prop = satellite.Propagator
        self.set_prop_initial_state(prop, agent)
        prop.Propagate()

        # Add a Field of View (FOV) sensor to the satellite
        cone_angle = agent["cone_angle"]

        # Set the sensor resolution based on the configuration
        if hasattr(agent, "resolution"):
            resolution = agent["resolution"]
        else:
            resolution = 0.1

        # Create the sensor
        sensor = satellite.Children.New(AgESTKObjectType.eSensor, f"{satellite_name}_sensor")

        # Set the sensor pattern based on the configuration
        if hasattr(agent, "pattern"):
            self.set_sensor_pattern(sensor, cone_angle, resolution, agent["pattern"])
        else:
            self.set_sensor_pattern(sensor, cone_angle, resolution)

        # Add dynamic pointing using azimuth and elevation (custom pointing model)
        az = agent["initial_azimuth"] # azimuth (coordinate range is 0 to 360)
        el = agent["initial_elevation"] # elevation (coordinate range is -90 to 90)
        sensor.CommonTasks.SetPointingFixedAzEl(az, el, AgEAzElAboutBoresight.eAzElAboutBoresightRotate)
        
        # Create the sensor manager
        sensor_mg = SensorManager(agent, sensor)

        # Create the features manager
        features_mg = FeaturesManager(agent)

        # Fill the custom states features for all those which do not belong ot the initial state from the agent configuration
        checked_var = []
        for var in agent["states_features"]:
            if var not in checked_var:
                if var in ["a", "e", "i", "raan", "aop", "ta"] + ["x", "y", "z", "vx", "vy", "vz"]:
                    checked_var += ["a", "e", "i", "raan", "aop", "ta"] + ["x", "y", "z", "vx", "vy", "vz"]
                    pass # These have their own update functions because they are included in the initial state
                elif var in ["az", "el"]:
                    features_mg.update_sensor_state(sensor_mg.current_azimuth, sensor_mg.current_elevation)
                    checked_var += ["az", "el"]
                elif var in ["detic_lat", "detic_lon", "detic_alt"]:
                    features_mg.update_detic_state(satellite, scenario.StartTime)
                    checked_var += ["detic_lat", "detic_lon", "detic_alt"]
                elif var.startswith("lat_") or var.startswith("lon_") or var.startswith("priority_"):
                    features_mg.update_target_memory(self.target_mg.df)
                    target_number = int(var.split("_")[1])
                    checked_var += [f"lat_{target_number}", f"lon_{target_number}", f"priority_{target_number}"]
                    pass
                else:
                    raise ValueError(f"State feature {var} not recognized. Please use orbital features, 'az', 'el', 'detic_lat', 'detic_lon' or 'detic_alt'.")

        return satellite, sensor_mg, features_mg
        
    def set_propagator_type(self, satellite):
        """
        Set the propagator of the satellite based on the agent configuration.
        - HPOP: High Precision Orbit Propagator
        - J2Perturbation: J2 Perturbation Model
        """
        # Set the propagator type depending on the configuration
        if self.agents_config["propagator"] == "HPOP":
            satellite.SetPropagatorType(AgEVePropagatorType.ePropagatorHPOP)
        elif self.agents_config["propagator"] == "J2Perturbation":
            satellite.SetPropagatorType(AgEVePropagatorType.ePropagatorJ2Perturbation)
        else:
            raise ValueError("Invalid propagator type. Please use 'HPOP' or 'J2Perturbation'.")
        
    def get_reference_frame_obj(self, agent):
        """
        Get the reference frame of the agent depending on configuration.
        """
        # Determine which of the reference frames
        if agent["reference_frame"] == "ICRF":
            return AgECoordinateSystem.eCoordinateSystemICRF
        elif agent["reference_frame"] == "Fixed":
            return AgECoordinateSystem.eCoordinateSystemFixed
        else:
            raise ValueError("Invalid reference frame. Please use 'ICRF' or 'Fixed'.")
        
    def get_reference_frame_str(self, agent):
        """
        Get the reference frame of the agent depending on configuration.
        """
        return agent["reference_frame"]
    
    def get_coordinate_system(self, agent):
        """
        Get the coordinate system of the agent depending on configuration.
        """
        # Determine which of the coordinate systems
        if agent["coordinate_system"] == "Classical":
            return "Classical Elements"
        elif agent["coordinate_system"] == "Cartesian":
            return "Cartesian"
        else:
            raise ValueError("Invalid coordinate system. Please use 'Classical' or 'Cartesian'.")
        
    def set_prop_initial_state(self, prop, agent):
        """
        Set the initial state of the satellite based on the agent configuration.
        - Classical: Orbital elements (a, e, i, raan, aop, ta)
        - Cartesian: Position and velocity (x, y, z, vx, vy, vz)
        """
        # Set the initial state depending on the coordinate system
        if agent["coordinate_system"] == "Classical":
            a, e, i, raan, aop, ta = [agent[key] for key in ["a", "e", "i", "raan", "aop", "ta"]]
            prop.InitialState.Representation.AssignClassical(self.get_reference_frame_obj(agent), a, e, i, raan, aop, ta)
        elif agent["coordinate_system"] == "Cartesian":
            x, y, z, vx, vy, vz = [agent[key] for key in ["x", "y", "z", "vx", "vy", "vz"]]
            prop.InitialState.Representation.AssignCartesian(self.get_reference_frame_obj(agent), x, y, z, vx, vy, vz)
        else:
            raise ValueError("Invalid coordinate system. Please use 'Classical' or 'Cartesian'.")
        
    def set_sensor_pattern(self, sensor, cone_angle, resolution, pattern="Simple Conic"):
        """
        Set the sensor pattern based on the agent configuration.
        """
        # Set the sensor pattern based on the configuration
        if pattern == "Simple Conic":
            sensor.SetPatternType(AgESnPattern.eSnSimpleConic)
            sensor.CommonTasks.SetPatternSimpleConic(cone_angle, resolution)
        else:
            raise ValueError("Invalid sensor pattern. Please use 'Simple Conic'.")
        
    def draw_initial_event_zones(self, file_path, scenario):
        """
        Draw the event zones (points or areas) on the scenario map.
        """
        # Create the events zones
        self.all_event_zones = pd.read_csv(file_path)

        # Draw 100 zones on the scenario map
        self.draw_n_zones(self.agents_config["visible_targets"], self.all_event_zones, scenario)

    def draw_n_zones(self, n: int, given_zones: pd.DataFrame, scenario, first_id: int=0):
        """
        Draw n event zones on the scenario map.
        """
        if n > given_zones.shape[0]:
            raise ValueError("The number of zones to draw is higher than the number of zones in the file.")
        elif n == 0:
            return
        
        # Sample n zones from the dataframe
        zones = given_zones.sample(n, ignore_index=True)

        # Define specific objects or grid zones to check for coverage
        for i in range(zones.shape[0]):
            # Get down to 0 if i+first_id is equal to the number of visible targets
            if i+first_id == self.agents_config["visible_targets"]:
                first_id = -i

            # See if a certain column exists in the dataframe
            if "lat [deg]" and "lon [deg]" in zones.columns:
                lat = float(zones.loc[i, "lat [deg]"])
                lon = float(zones.loc[i, "lon [deg]"])
                priority = float(zones.loc[i, "priority [1, 10]"])

                # Check if altitude is specified
                if "alt [m]" in zones.columns:
                    alt = float(zones.loc[i, "lat [deg]"])
                    self.point_drawing(scenario, i+first_id, lat, lon, priority, alt)
                else:
                    self.point_drawing(scenario, i+first_id, lat, lon, priority, alt=0)
            elif "lat 1 [deg]" and "lon 1 [deg]" in zones.columns:
                lats = [float(zones.loc[i, f"lat {j} [deg]"]) for j in range(int(len(zones.columns)))]
                lons = [float(zones.loc[i, f"lon {j} [deg]"]) for j in range(int(len(zones.columns)))]
                priority = float(zones.loc[i, "priority [1, 10]"])
                self.area_drawing(scenario, i+first_id, lats, lons, priority)
            else:
                raise ValueError("The column names for the event zones file is are not recognized. Please use 'lat [deg]' and 'lon [deg]' format or 'lat 1 [deg]', 'lon 1 [deg]', ... format.")

    def point_drawing(self, scenario: IAgStkObject, idx: int, lat, lon, priority, alt=0):
        """
        Draw a point target on the scenario map.
        """
        # Create the point target
        target = scenario.Children.New(AgESTKObjectType.eTarget, f"target{idx}")
        target.Position.AssignGeodetic(lat, lon, alt)

        # Store the zones
        self.target_mg.append_zone(f"target{idx}", "Point", lat, lon, priority)

    def area_drawing(self, scenario, idx: int, lats, lons, priority):
        """
        Draw an area target on the scenario map.
        """
        # Create the area target
        target = scenario.Children.New(AgESTKObjectType.eAreaTarget, f"target{idx}")
        target.AreaType = AgEAreaType.ePattern

        if len(lats) != len(lons):
            raise ValueError("Latitude and longitude lists must have the same length.")
        elif len(lats) < 3:
            raise ValueError("Area target must have at least 3 points.")
        
        lat_lon_added = 0
        
        # Set the area boundary points
        for lat, lon in zip(lats, lons):
            # Check for None values (dataframes require same columns for all rows)
            if lat == None or lon == None:
                if lat_lon_added < 3:
                    raise ValueError("Area target must have at least 3 points.")
                break

            # Add the latitude and longitude to the area target
            target.AreaTypeData.Add(lat, lon)
            lat_lon_added += 1

        target.AutoCentroid = True

        # Store the zones
        self.target_mg.append_zone(f"target{idx}", "Area", lats[0], lons[0], priority)

    def delete_object(self, scenario: IAgStkObject, name: str, type: str):
        """
        Delete n event zones from the scenario map.
        """
        # Unload the object from the scenario
        if type == "Point":
            scenario.Children.Unload(AgESTKObjectType.eTarget, name)
        elif type == "Area":
            scenario.Children.Unload(AgESTKObjectType.eAreaTarget, name)
        
    def step(self, agent_id, action, delta_time):
        """
        Forward method. Return the next state and reward based on the current state and action taken.
        """
        # Update the agent's features
        done = self.update_agent(agent_id, action, delta_time)

        # Return None if the episode is done
        if done:
            print(f"Episode for agent {agent_id} is done.")
            return None, None, done

        # Get the next state
        state = self.get_state(agent_id, as_dict=True)

        # If zero delta time, return only the state and do not enter the reward calculation
        if delta_time == 0.0:
            return state, None, None

        # Get the reward
        reward = self.get_reward(agent_id, self.scenario, delta_time)

        # Store the reward
        self.plotter.store_reward(reward)

        # Delete and draw n zones
        n = int(delta_time/self.agents_config["target_renew_time"])
        self.erase_and_draw_zones(n)

        return state, reward, done
    
    def erase_and_draw_zones(self, n):
        """
        Erase and draw n event zones on the scenario map.
        """
        # Check it is not too many changes
        n = n if n < len(self.target_mg.df) else len(self.target_mg.df)

        first_id = 0

        # Delete n zones
        for i in range(n):
            zone = self.target_mg.get_zone_by_row(i)
            name = zone["name"].values[0]
            type = zone["type"].values[0]
            self.delete_object(self.scenario, name, type)

            if i == 0:
                first_id = int(name.split("target")[1])

        # Erase these from the target memory
        self.target_mg.erase_first_n_zones(n)
        
        # Draw n zones
        self.draw_n_zones(n, self.all_event_zones, self.scenario, first_id)
    
    def update_agent(self, agent_id, action, delta_time):
        """
        Class to update the agent's features based on the action taken and the time passed.
        """
        # Get the satellite tuple
        satellite, sensor_mg, features_mg, date_mg = self.get_satellite(agent_id)

        # Iterate over all actions taken
        for key in action.keys():
            # Update the actions in the features manager
            features_mg.update_action(key, action[key])

            # Perform sensor changes
            if key == "d_az":
                sensor_mg.update_azimuth(action[key])
            elif key == "d_el":
                sensor_mg.update_elevation(action[key])
            else:
                raise ValueError("Invalid action. Please use 'd_az' or 'd_el'.")
            
        az = sensor_mg.get_item("current_azimuth")
        el = sensor_mg.get_item("current_elevation")
        sensor_mg.sensor.CommonTasks.SetPointingFixedAzEl(az, el, AgEAzElAboutBoresight.eAzElAboutBoresightRotate)

        # Update the date manager
        date_mg.update_date_after(delta_time)

        # Check if the episode is done
        done = self.check_done(agent_id)

        # Return True if the episode is done
        if done:
            return True
        else:
            # Find the orbital elements and update the features manager
            orbital_elements = self.get_orbital_elements(satellite, date_mg.current_date, features_mg.agent_config)

            # Fill the custom states features for all those which do not belong ot the initial state from the agent configuration
            checked_var = []
            for var in features_mg.state.keys():
                if var not in checked_var:
                    if var in ["a", "e", "i", "raan", "aop", "ta"] + ["x", "y", "z", "vx", "vy", "vz"]:
                        features_mg.update_orbital_elements(orbital_elements)
                        checked_var += ["a", "e", "i", "raan", "aop", "ta"] + ["x", "y", "z", "vx", "vy", "vz"]
                    elif var in ["az", "el"]:
                        features_mg.update_sensor_state(sensor_mg.current_azimuth, sensor_mg.current_elevation)
                        checked_var += ["az", "el"]
                    elif var in ["detic_lat", "detic_lon", "detic_alt"]:
                        features_mg.update_detic_state(satellite, date_mg.current_date)
                        checked_var += ["detic_lat", "detic_lon", "detic_alt"]
                    elif var.startswith("lat_") or var.startswith("lon_") or var.startswith("priority_"):
                        features_mg.update_target_memory(self.target_mg.df)
                        target_number = int(var.split("_")[1])
                        checked_var += [f"lat_{target_number}", f"lon_{target_number}", f"priority_{target_number}"]
                        pass
                    else:
                        raise ValueError(f"State feature {var} not recognized. Please use orbital features, 'az', 'el', 'detic_lat', 'detic_lon' or 'detic_alt'.")
                    
            return False

    def get_state(self, agent_id, as_dict=False):
        """
        Get the state of the agent based on the current features.
        """
        # Get the satellite tuple
        _, _, features_mg, _ = self.get_satellite(agent_id)

        # Get the features of the agent
        state = features_mg.get_state()

        return state if as_dict else [value for value in state.values()]

    def get_reward(self, agent_id, scenario: IAgStkObject, delta_time: float) -> float:
        """
        Get the reward of the agent based on its state-action pair.
        """
        # Get the satellite tuple
        satellite, _, features_mg, date_mg = self.get_satellite(agent_id)

        # Create the rewarder
        rewarder = self.rewarder

        # Create the access data providers
        data_providers = []
        sensor = satellite.Children.Item(f"{satellite.InstanceName}_sensor")

        # minduration-adjusted current date
        adj_current_date = date_mg.get_current_date_after(self.agents_config["min_duration"])

        # Iterate over all point targets
        if scenario.Children.GetElements(AgESTKObjectType.eTarget) is not None:
            for target in scenario.Children.GetElements(AgESTKObjectType.eTarget):
                access = sensor.GetAccessToObject(target)
                access_data_provider = access.DataProviders.Item("Access Data").Exec(date_mg.last_date, adj_current_date)
                aer_data_provider = access.DataProviders.Item("AER Data").Group.Item("NorthEastDown").Exec(date_mg.last_date, adj_current_date, delta_time/10)
                data_providers.append((access_data_provider, aer_data_provider))

        # Iterate over all area targets
        if scenario.Children.GetElements(AgESTKObjectType.eAreaTarget) is not None:
            for target in scenario.Children.GetElements(AgESTKObjectType.eAreaTarget):
                access = sensor.GetAccessToObject(target)
                access_data_provider = access.DataProviders.Item("Access Data").Exec(date_mg.last_date, adj_current_date)
                aer_data_provider = access.DataProviders.Item("AER Data").Group.Item("NorthEastDown").Exec(date_mg.last_date, adj_current_date, delta_time/10)
                data_providers.append((access_data_provider, aer_data_provider))

        # Get the slew rates
        slew_rates = []
        for diff in features_mg.action.keys():
            if diff in ["d_az", "d_el"]:
                slew_rate = features_mg.action[diff] / delta_time
                slew_rates.append(abs(slew_rate))

        # Call the rewarder to calculate the reward
        reward = rewarder.calculate_reward(data_providers, date_mg, slew_rates)

        return reward

    def check_done(self, agent_id):
        """
        Check if the episode is done based on the current date.
        """
        # Get the satellite tuple
        _, _, _, date_mg = self.get_satellite(agent_id)

        # Check if the simulation time ended
        if date_mg.time_ended():
            return True
        else:
            return False
    
    def get_satellite(self, agent_id) -> tuple[IAgStkObject, SensorManager, FeaturesManager, DateManager]:
        """
        Get the satellite based on the agent ID.
        """
        # See whether the input is a string or an integer
        if isinstance(agent_id, str):
            for tuple in self.satellites_tuples:
                if tuple[0].InstanceName == agent_id:
                    return tuple
        elif isinstance(agent_id, int):
            for tuple in self.satellites_tuples:
                if tuple[0].InstanceName == f"MySatellite{agent_id}":
                    return tuple
        else:
            raise ValueError("Invalid agent ID. Please use a string or an integer.")
        
        raise ValueError(f"Satellite with ID {agent_id} not found.")
    
    def get_orbital_elements(self, satellite, specific_time, agent):
        """
        Get the orbital elements of the satellite at a specific time.
        """
        return satellite.DataProviders.Item(self.get_coordinate_system(agent)).Group.Item(self.get_reference_frame_str(agent)).ExecSingle(specific_time)