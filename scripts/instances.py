import os
import json
import socket
import psutil
import pandas as pd
from time import perf_counter

from agi.stk12.stkengine import STKEngine
from agi.stk12.stkobjects import *
from agi.stk12.stkutil import *

from scripts.utils import *

# Constants
RT = 6371  # radius of the earth in km

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

        # Memory used
        process = psutil.Process(os.getpid())
        memory_used = process.memory_info().rss
        print(f"Memory used: {memory_used / (1024 ** 2):.2f} MB")
    
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
            if not self.stk_env.agents_config["deep_training"]:
                self.stk_env.stk_root.SaveScenario()
            self.stk_env.stk_root.CloseScenario()
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
        self.stk_engine = STKEngine()
        self.stk_app = self.stk_engine.StartApplication(noGraphics=False)
        self.stk_root = self.stk_app.NewObjectRoot()
        self.scenario = self.build_scenario(self.stk_root, self.agents_config)
        self.satellites_tuples = []
        self.date_mg = DateManager(self.scenario.StartTime, self.scenario.StopTime)
        self.target_mg = TargetManager(self.scenario.StartTime, agents_config["visible_targets"])
        self.rewarder = Rewarder(agents_config, self.target_mg)
        self.plotter = Plotter(out_folder_path)

        # Add the zones of interest
        self.draw_initial_event_zones(evpt_file_path, self.scenario)

        # Build the satellites by iterating over the agents
        for i, agent in enumerate(agents_config["agents"]):
            agent = DataFromJSON(agent, "agent").get_dict()
            satellite, sensor_mg, features_mg, date_mg, attitude_mg = self.build_satellite(agent, self.scenario, i)
            self.satellites_tuples.append((satellite, sensor_mg, features_mg, date_mg, attitude_mg)) # append the satellite, its sensor manager and its date manager

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

    def build_satellite(self, agent, scenario: IAgStkObject, idx) -> tuple[IAgStkObject, SensorManager, FeaturesManager, DateManager, AttitudeManager]:
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

        # Create the sensor
        sensor = satellite.Children.New(AgESTKObjectType.eSensor, f"{satellite_name}_sensor")

        # Create the sensor manager
        sensor_mg = SensorManager(agent, sensor)

        # Set the sensor pattern based on the configuration
        if hasattr(agent, "pattern"):
            self.set_sensor_pattern(sensor, sensor_mg.cone_angle, sensor_mg.resolution, sensor_mg.pattern)
        else:
            self.set_sensor_pattern(sensor, sensor_mg.cone_angle, sensor_mg.resolution)

        # Add static pointing using azimuth and elevation (custom pointing model)
        sensor.CommonTasks.SetPointingFixedAzEl(sensor_mg.current_azimuth, sensor_mg.current_elevation, AgEAzElAboutBoresight.eAzElAboutBoresightRotate)

        # Create the features manager
        features_mg = FeaturesManager(agent)

        # Create the date manager
        date_mg = DateManager(scenario.StartTime, scenario.StopTime)

        # Create the attitude manager
        attitude_mg = AttitudeManager(agent)

        # Set attitude profile
        cmd = f"""SetAttitude {satellite.Path} Profile AlignConstrain PR {attitude_mg.current_pitch} {attitude_mg.current_roll} "Satellite/{satellite.InstanceName} {attitude_mg.align_reference}" Axis {attitude_mg.constraint_axes} "Satellite/{satellite.InstanceName} {attitude_mg.constraint_reference}" """
        self.stk_root.ExecuteCommand(cmd)

        # Update all necessary auxiliar features
        features_mg.update_entire_aux_state(satellite, self.target_mg, scenario.StartTime)

        # Fill the custom states features for all those which do not belong ot the initial state from the agent configuration
        checked_var = []
        for var in agent["states_features"]:
            if var not in checked_var:
                if var in ["a", "e", "i", "raan", "aop", "ta"] + ["x", "y", "z", "vx", "vy", "vz"]:
                    checked_var += ["a", "e", "i", "raan", "aop", "ta"] + ["x", "y", "z", "vx", "vy", "vz"]
                    pass # These have their own update functions because they are included in the initial state
                elif var in ["pitch", "roll"]:
                    features_mg.update_attitude_state(attitude_mg.current_pitch, attitude_mg.current_roll)
                    checked_var += ["pitch", "roll"]
                elif var in ["az", "el"]:
                    features_mg.update_sensor_state(sensor_mg.current_azimuth, sensor_mg.current_elevation)
                    checked_var += ["az", "el"]
                elif var in ["detic_lat", "detic_lon", "detic_alt"]:
                    features_mg.update_detic_state()
                    checked_var += ["detic_lat", "detic_lon", "detic_alt"]
                elif var.startswith("lat_") or var.startswith("lon_") or var.startswith("priority_"):
                    features_mg.update_target_memory(self.target_mg.get_FoR_window_df(date_mg, features_mg, margin_pct=0), self.target_mg.df)
                    target_number = int(var.split("_")[1])
                    checked_var += [f"lat_{target_number}", f"lon_{target_number}", f"priority_{target_number}"]
                    pass
                else:
                    raise ValueError(f"State feature {var} not recognized. Please use orbital features, 'az', 'el', 'detic_lat', 'detic_lon' or 'detic_alt'.")

        return satellite, sensor_mg, features_mg, date_mg, attitude_mg
        
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

        # Draw X zones on the scenario map
        self.draw_n_zones(self.agents_config["visible_targets"], self.all_event_zones, scenario, scenario.StartTime)

    def draw_n_zones(self, n: int, given_zones: pd.DataFrame, scenario, start_date, first_id: int=0):
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
            # Calculate the end date of the zone
            end_date = self.date_mg.get_date_after(float(zones.loc[i, "duration [s]"]), start_date)

            # See if a certain column exists in the dataframe
            if "lat [deg]" and "lon [deg]" in zones.columns:
                lat = float(zones.loc[i, "lat [deg]"])
                lon = float(zones.loc[i, "lon [deg]"])
                priority = float(zones.loc[i, "priority"])

                # Check if altitude is specified
                if "alt [m]" in zones.columns:
                    alt = float(zones.loc[i, "lat [deg]"])
                    self.point_drawing(scenario, i+first_id, lat, lon, priority, start_date, end_date, alt)
                else:
                    self.point_drawing(scenario, i+first_id, lat, lon, priority, start_date, end_date, alt=0)
            elif "lat 1 [deg]" and "lon 1 [deg]" in zones.columns:
                lats = [float(zones.loc[i, f"lat {j} [deg]"]) for j in range(int(len(zones.columns)))]
                lons = [float(zones.loc[i, f"lon {j} [deg]"]) for j in range(int(len(zones.columns)))]
                priority = float(zones.loc[i, "priority"])
                self.area_drawing(scenario, i+first_id, lats, lons, priority, start_date, end_date)
            else:
                raise ValueError("The column names for the event zones file is are not recognized. Please use 'lat [deg]' and 'lon [deg]' format or 'lat 1 [deg]', 'lon 1 [deg]', ... format.")

    def point_drawing(self, scenario: IAgStkObject, idx: int, lat, lon, priority, start_date, end_date, alt=0):
        """
        Draw a point target on the scenario map.
        """
        # Create the point target
        target = scenario.Children.New(AgESTKObjectType.eTarget, f"target{idx}")

        self.stk_root.BeginUpdate()
        target.Position.AssignGeodetic(lat, lon, alt)
        self.stk_root.EndUpdate()

        if not self.agents_config["deep_training"]:
            # Add the time intervals
            cmd = f"""DisplayTimes {target.Path} Intervals Add 1 "{start_date}" "{end_date}" """
            self.stk_root.ExecuteCommand(cmd)

        # Store the zones
        self.target_mg.append_zone(f"target{idx}", target, "Point", lat, lon, priority, start_date, end_date)

    def area_drawing(self, scenario, idx: int, lats, lons, priority, start_date, end_date):
        """
        Draw an area target on the scenario map.
        """
        # Create the area target
        target = scenario.Children.New(AgESTKObjectType.eAreaTarget, f"target{idx}")
        target.AreaType = AgEAreaType.ePattern

        if not self.agents_config["deep_training"]:
            # Add the time intervals
            cmd = f"""DisplayTimes {target.Path} Intervals Add 1 "{start_date}" "{end_date}" """
            self.stk_root.ExecuteCommand(cmd)

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
        self.target_mg.append_zone(f"target{idx}", target, "Area", lats[0], lons[0], priority, start_date, end_date)

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
        # Check if time is over 0.5 seconds
        if delta_time < 0.5 and delta_time != 0.0:
            raise ValueError("Delta time must be at least 0.5 seconds.")
        
        if self.agents_config["debug"]:
            before = perf_counter()

        # Update the agent's features
        done = self.update_agent(agent_id, action, delta_time)

        if self.agents_config["debug"]:
            print(f"Time taken to update agent: {perf_counter() - before:.2f} seconds.")

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
        reward = self.get_reward(agent_id, delta_time)

        # Store the reward
        self.plotter.store_reward(reward)

        if self.agents_config["debug"]:
            before = perf_counter()

        # Update the target zones
        self.update_target_zones(agent_id)

        if self.agents_config["debug"]:
            print(f"Time taken to update target zones: {perf_counter() - before:.2f} seconds.")

        return state, reward, done
    
    def update_target_zones(self, agent_id):
        """
        Update the target zones based on the current date.
        """
        # Get the satellite tuple
        _, _, _, date_mg, _ = self.get_satellite(agent_id)

        # Delete and draw n zones
        n = self.target_mg.n_of_zones_to_add(date_mg.current_date)
        self.draw_n_zones(n, self.all_event_zones, self.scenario, date_mg.current_date, self.target_mg.max_id)
        self.unload_expired_zones(self.satellites_tuples, self.scenario, self.target_mg)

    def unload_expired_zones(self, satellites_tuples: tuple, scenario: IAgStkObject, target_mg: TargetManager):
        """
        Unload the zones which have already expired.
        """
        # Gather the lowest current date of all satellites
        lowest_current_date = min([date_mg.num_of_date(date_mg.simplify_date(date_mg.current_date)) for _, _, _, date_mg, _ in satellites_tuples])

        if self.agents_config["deep_training"]:
            # Get the zones which are unloadable because they will no longer be visible
            unloadable_df = target_mg.get_unloadable_zones_before(lowest_current_date)

            # Unload the zones
            for _, row in unloadable_df.iterrows():
                self.delete_object(scenario, row["name"], row["type"])

        target_mg.unload_zones_before(lowest_current_date)
    
    def update_agent(self, agent_id, action, delta_time):
        """
        Class to update the agent's features based on the action taken and the time passed.
        """
        # Get the satellite tuple
        satellite, sensor_mg, features_mg, date_mg, attitude_mg = self.get_satellite(agent_id)

        change_sensor = False
        change_attitude = False

        self.stk_root.BeginUpdate()

        if delta_time != 0.0:
            # Iterate over all actions taken
            for key in action.keys():
                # Update the actions in the features manager
                features_mg.update_action(key, action[key])

                # Perform sensor changes
                if key == "d_az":
                    _ = sensor_mg.update_azimuth(action[key])
                    change_sensor = True
                elif key == "d_el":
                    _ = sensor_mg.update_elevation(action[key])
                    change_sensor = True
                elif key == "d_pitch":
                    attitude_mg.update_pitch(action[key])
                    change_attitude = True
                elif key == "d_roll":
                    attitude_mg.update_roll(action[key])
                    change_attitude = True
                else:
                    raise ValueError("Invalid action. Please use 'd_az' or 'd_el'.")
            
            if change_sensor:
                az = sensor_mg.get_item("current_azimuth")
                el = sensor_mg.get_item("current_elevation")
                sensor_mg.sensor.CommonTasks.SetPointingFixedAzEl(az, el, AgEAzElAboutBoresight.eAzElAboutBoresightRotate)

            if change_attitude:
                if date_mg.current_date == date_mg.start_date:
                    # Transition command
                    cmd = attitude_mg.get_transition_command(satellite, date_mg.get_current_date_after(0.1))
                    self.stk_root.ExecuteCommand(cmd)
                else:
                    if self.agents_config["deep_training"]:
                        # Get the attitude profile
                        cmd = attitude_mg.get_segments_command(satellite)

                        # Execute the command
                        segments = self.stk_root.ExecuteCommand(cmd)

                        if segments.Count > 12: # number obtained from testing study
                            # First, clear all but the last attitude command
                            cmd = attitude_mg.get_clear_data_command(satellite)
                            self.stk_root.ExecuteCommand(cmd)
                            
                            # Then, add the previous orientation command
                            cmd = attitude_mg.get_previous_orientation_command()
                            self.stk_root.ExecuteCommand(cmd)

                    # Transition command
                    cmd = attitude_mg.get_transition_command(satellite, date_mg.current_date)
                    self.stk_root.ExecuteCommand(cmd)

                # Orientation command
                cmd = attitude_mg.get_new_orientation_command(satellite, date_mg.get_current_date_after(delta_time - 0.5))
                self.stk_root.ExecuteCommand(cmd)

            # Update the date manager
            date_mg.update_date_after(delta_time)
            if self.agents_config["debug"]:
                print(f"Current date: {date_mg.current_date}")

        self.stk_root.EndUpdate()

        # Check if the episode is done
        done = self.check_done(agent_id)

        # Return True if the episode is done
        if done:
            return True

        # Find the orbital elements and update the features manager
        orbital_elements = self.get_orbital_elements(satellite, date_mg.current_date, features_mg.agent_config)

        # Update all necessary auxiliar features
        features_mg.update_entire_aux_state(satellite, self.target_mg, date_mg.current_date)

        # Fill the custom states features for all those which do not belong ot the initial state from the agent configuration
        checked_var = []
        for var in features_mg.state.keys():
            if var not in checked_var:
                if var in ["a", "e", "i", "raan", "aop", "ta"] + ["x", "y", "z", "vx", "vy", "vz"]:
                    features_mg.update_orbital_elements(orbital_elements)
                    checked_var += ["a", "e", "i", "raan", "aop", "ta"] + ["x", "y", "z", "vx", "vy", "vz"]
                elif var in ["pitch", "roll"]:
                    features_mg.update_attitude_state(attitude_mg.current_pitch, attitude_mg.current_roll)
                    checked_var += ["pitch", "roll"]
                elif var in ["az", "el"]:
                    features_mg.update_sensor_state(sensor_mg.current_azimuth, sensor_mg.current_elevation)
                    checked_var += ["az", "el"]
                elif var in ["detic_lat", "detic_lon", "detic_alt"]:
                    features_mg.update_detic_state()
                    checked_var += ["detic_lat", "detic_lon", "detic_alt"]
                elif var.startswith("lat_") or var.startswith("lon_") or var.startswith("priority_"):
                    features_mg.update_target_memory(self.target_mg.get_FoR_window_df(date_mg, features_mg, margin_pct=0), self.target_mg.df)
                    target_number = int(var.split("_")[1])
                    checked_var += [f"lat_{target_number}", f"lon_{target_number}", f"priority_{target_number}"]
                else:
                    raise ValueError(f"State feature {var} not recognized. Please use orbital features, 'az', 'el', 'detic_lat', 'detic_lon' or 'detic_alt'.")

        return False

    def get_state(self, agent_id, as_dict=False):
        """
        Get the state of the agent based on the current features.
        """
        # Get the satellite tuple
        _, _, features_mg, _, _ = self.get_satellite(agent_id)

        # Get the features of the agent
        state = features_mg.get_state()

        return state if as_dict else [value for value in state.values()]

    def get_reward(self, agent_id, delta_time: float) -> float:
        """
        Get the reward of the agent based on its state-action pair.
        """
        # Get the satellite tuple
        satellite, sensor_mg, features_mg, date_mg, attitude_mg = self.get_satellite(agent_id)

        # Create the rewarder
        rewarder = self.rewarder

        # Create the access data providers
        data_providers = []
        sensor = satellite.Children.Item(f"{satellite.InstanceName}_sensor")

        # minduration-adjusted current date
        adj_current_date = date_mg.get_current_date_after(self.agents_config["min_duration"])

        # Get the dataframe filtered on the window and the FoR
        FoR_window_df = self.target_mg.get_FoR_window_df(date_mg=date_mg, features_mg=features_mg)

        # Iterate over all targets in the window
        for _, target in FoR_window_df.iterrows():
            access = sensor.GetAccessToObject(target["object"])
            access.AccessTimePeriod = AgEAccessTimeType.eUserSpecAccessTime
            access.SpecifyAccessTimePeriod(target["start_time"], target["end_time"])
            access.ComputeAccess()
            access_data_provider = access.DataProviders.Item("Access Data").Exec(date_mg.last_date, adj_current_date)
            aer_data_provider = access.DataProviders.Item("AER Data").Group.Item("NorthEastDown").Exec(date_mg.last_date, adj_current_date, delta_time/10)
            data_providers.append((access_data_provider, aer_data_provider))

        # Call the rewarder to calculate the reward
        reward = rewarder.calculate_reward(data_providers, delta_time, date_mg, sensor_mg, features_mg, attitude_mg.angle_domains)

        return reward

    def check_done(self, agent_id):
        """
        Check if the episode is done based on the current date.
        """
        # Get the satellite tuple
        _, _, _, date_mg, _ = self.get_satellite(agent_id)

        # Check if the simulation time ended
        if date_mg.time_ended():
            return True
        else:
            return False
    
    def get_satellite(self, agent_id) -> tuple[IAgStkObject, SensorManager, FeaturesManager, DateManager, AttitudeManager]:
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