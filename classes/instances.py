import json
import socket
import pandas as pd

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
        self.initialize_agent(self.conf_file_path)
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
            self.output_folder = args.out

    def initialize_agent(self, file_path):
        """
        Initialize the agent with the given configuration.
        """
        with open(file_path, "r") as f:
            agents_config = json.load(f)

        if not agents_config:
            raise ValueError("Agent configuration is empty.")
        
        # Initialize the agent with the configuration in a simplified dictionary
        self.stk_env = STKEnvironment(DataFromJSON(agents_config, "configuration").get_dict(), self.evpt_file_path)
    
    def get_next_state_and_reward(self, agent_id, action, delta_time):
        """
        Return the next state and reward of the agent.
        """
        return self.stk_env.step(agent_id, action, delta_time)
    
    def generate_gif(self):
        """
        Perform final displays of the period.
        """
        self.stk_env.stk_root.SaveScenario()
    
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
            self.generate_gif()
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
    def __init__(self, agents_config, evpt_file_path):
        self.agents_config = agents_config
        stk_app = STKEngine().StartApplication(noGraphics=False)
        stk_root = stk_app.NewObjectRoot()
        self.stk_app = stk_app
        self.stk_root = stk_root
        self.scenario = self.build_scenario(self.stk_root, self.agents_config)
        self.satellites_tuples = []
        self.current_dates = []

        # Build the satellites by iterating over the agents
        for i, agent in enumerate(agents_config["agents"]):
            agent = DataFromJSON(agent, "agent").get_dict()
            satellite, sensor_mg, features_mg = self.build_satellite(agent, self.scenario, i)
            date_mg = DateManager(self.scenario.StartTime, self.scenario.StopTime)
            self.satellites_tuples.append((satellite, sensor_mg, features_mg, date_mg)) # append the satellite, its sensor manager and its date manager

        # Add the zones of interest
        self.draw_event_zones(evpt_file_path, self.stk_root, self.scenario)

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
        
    def draw_event_zones(self, file_path, root, scenario):
        """
        Draw the event zones (points or areas) on the scenario map.
        """
        # Create the events zones
        event_zones = pd.read_csv(file_path)

        # Trim the data to 1000 sampled zones
        zones = event_zones.sample(300, ignore_index=True)

        # Define specific objects or grid zones to check for coverage
        for i in range(zones.shape[0]):
            # See if a certain column exists in the dataframe
            if "lat [deg]" and "lon [deg]" in zones.columns:
                lat = float(zones.loc[i, "lat [deg]"])
                lon = float(zones.loc[i, "lon [deg]"])

                # Check if altitude is specified
                if "alt [m]" in zones.columns:
                    alt = float(zones.loc[i, "lat [deg]"])
                    self.point_drawing(scenario, i, lat, lon, alt)
                else:
                    self.point_drawing(scenario, i, lat, lon, alt=0)
            elif "lat 1 [deg]" and "lon 1 [deg]" in zones.columns:
                lats = [float(zones.loc[i, f"lat {j} [deg]"]) for j in range(int(len(zones.columns)))]
                lons = [float(zones.loc[i, f"lon {j} [deg]"]) for j in range(int(len(zones.columns)))]
                self.area_drawing(scenario, i, lats, lons)
            else:
                raise ValueError("The column names for the event zones file is are not recognized. Please use 'lat [deg]' and 'lon [deg]' format or 'lat 1 [deg]', 'lon 1 [deg]', ... format.")

        root.EndUpdate()

    def point_drawing(self, scenario, idx: int, lat, lon, alt=0):
        """
        Draw a point target on the scenario map.
        """
        # Create the point target
        target = scenario.Children.New(AgESTKObjectType.eTarget, f"target{idx+1}")
        target.Position.AssignGeodetic(lat, lon, alt)

    def area_drawing(self, scenario, idx: int, lats, lons):
        """
        Draw an area target on the scenario map.
        """
        # Create the area target
        target = scenario.Children.New(AgESTKObjectType.eAreaTarget, f"target{idx+1}")
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
        
    def step(self, agent_id, action, delta_time):
        """
        Forward method. Return the next state and reward based on the current state and action taken.
        """
        # Update the agent's features
        self.update_agent(agent_id, action, delta_time)

        # Get the next state
        state = self.get_state(agent_id)

        # Get the reward
        reward = self.get_reward(agent_id, self.scenario)

        # Check if the episode is done
        done = self.check_done(agent_id)

        return state, reward, done
    
    def update_agent(self, agent_id, action, delta_time):
        """
        Class to update the agent's features based on the action taken and the time passed.
        """
        # Get the satellite tuple
        satellite, sensor_mg, feature_mg, date_mg = self.get_satellite(agent_id)

        # Iterate over all actions taken
        for key in action.keys():
            # Update the actions in the features manager
            feature_mg.update_action(key, action[key])

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

        # Update the features manager sensor data
        for var in ["az", "el"]:
            feature_mg.update_state(var, locals()[var])

        # Update the date manager
        date_mg.update_date_after(delta_time)

        # Find the orbital elements and update the features manager
        orbital_elements = self.get_orbital_elements(satellite, date_mg.current_date, feature_mg.agent_config)
        feature_mg.update_orbital_elements(orbital_elements)

        # HERE TO PUT ADDITIONAL FEATURES IF THEY EXIST IN THE FEATURE MANAGER

    def get_state(self, agent_id):
        """
        Get the state of the agent based on the current features.
        """
        # Get the satellite tuple
        _, _, feature_mg, _ = self.get_satellite(agent_id)

        # Get the features of the agent
        state = feature_mg.get_state()

        return [value for value in state.values()]

    def get_reward(self, agent_id, scenario: IAgStkObject):
        """
        Get the reward of the agent based on its state-action pair.
        """
        # Get the satellite tuple
        satellite, sensor_mg, feature_mg, date_mg = self.get_satellite(agent_id)

        # Create the rewarder
        rewarder = Rewarder()

        # The reward is based on the sensor's coverage of the event zones
        targets = []

        # Append the points targets to the list
        if scenario.Children.GetElements(AgESTKObjectType.eTarget) is not None:
            now_targets = [t for t in scenario.Children.GetElements(AgESTKObjectType.eTarget)]
            for target in now_targets:
                targets.append(target)
        
        # Append the area targets to the list
        if scenario.Children.GetElements(AgESTKObjectType.eAreaTarget) is not None:
            now_targets = [t for t in scenario.Children.GetElements(AgESTKObjectType.eAreaTarget)]
            for target in now_targets:
                targets.append(target)

        # Check access to Target 1 at the specific moment
        sensor = satellite.Children.Item(f"{satellite.InstanceName}_sensor")

        access_data_providers = []

        # Get the access data for each target
        for target in targets:
            access = sensor.GetAccessToObject(target)
            access_data_provider = access.DataProviders.Item("Access Data").Exec(scenario.StartTime, date_mg.current_date)
            access_data_providers.append(access_data_provider)

        n_obs = 0

        # Iterate over the access data providers
        for access_data_provider in access_data_providers:
            # Check if the access is valid
            if access_data_provider.Intervals.Count > 0:
                n_obs += 1
                data = 0

            # CALCULATE THE REWARD BASED ON THE ACCESS DATA
            # ---------------------------------------------
            reward = rewarder.calculate_reward(data, satellite, sensor_mg, feature_mg, date_mg)

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