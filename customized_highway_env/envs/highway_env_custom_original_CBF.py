import numpy as np
from typing import Tuple
from gym.envs.registration import register

from highway_env import utils
from customized_highway_env.envs.common.abstract_original_CBF import AbstractEnv_original_CBF
from highway_env.road.road import RoadNetwork
from customized_highway_env.road.road_customized import Road_original
from highway_env.utils import near_split
from customized_highway_env.vehicle.controller_customized import ControlledVehicle_original
from customized_highway_env.vehicle.controller_customized import clone_MDPVehicle
from customized_highway_env.vehicle.behavior_customized import no_input_IDMVehicle


class HighwayEnv_CBF_1(AbstractEnv_original_CBF):
    RIGHT_LANE_REWARD: float = 0.1
    """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

    HIGH_SPEED_REWARD: float = 0.4  #
    """The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"]."""

    LANE_CHANGE_REWARD: float = 0
    """The reward received at each lane change action."""

    steps = 0

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics_original",
                "vehicles_count": 7,  # specific environment
            },
            "action": {
                "type": "DiscreteMetaAction_original",
            },
            "lanes_count": 4,
            "vehicles_count": 20,
            "controlled_vehicles": 1,
            "other_vehicles_type": "customized_highway_env.vehicle.behavior_customized.IDMVehicle_original",
            "initial_lane_id": None,
            "duration": 100,  # we can double check if it is second
            "ego_spacing": 2,
            "vehicles_density": 1.5,  ## this is something I changed
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False,
            "simulation_frequency": 10,
            "policy_frequency": 2
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road_original(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                                  np_random=self.np_random, record_history=self.config["show_trajectories"])

        self.road_clone = Road_original(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                                        np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                self.road.vehicles.append(
                    other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                )

    def _reward(self, action) -> float:

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle_original) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * lane / max(len(neighbours) - 1, 1) \
            + self.HIGH_SPEED_REWARD * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                            [self.config["collision_reward"], self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
                            [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _do_the_prediction(self):

        record_dict = dict()  # initialize the information saver

        for action_clone in ['LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER', 'SLOWER']:
            # build the controlled vehicles
            self.road_clone.vehicles = []
            self.controlled_vehicle_clone = clone_MDPVehicle.clone_from(self.road_clone, self.vehicle)
            self.controlled_vehicle_clone.act(action_clone)

            self.road_clone.vehicles.append(self.controlled_vehicle_clone)

            # find the current lane index
            record_dict[action_clone] = dict()
            record_dict[action_clone]["controlled_vehicle"] = []
            record_dict[action_clone]["front_current"] = []
            record_dict[action_clone]["rear_current"] = []
            record_dict[action_clone]["front_target"] = []
            record_dict[action_clone]["rear_target"] = []

            # record_dict[action_clone]["controlled_vehicle"] = self._simulate_clone_controlled_vehicles(controlled_vehicle_clone)

            original_vehicle_list = self.road.close_vehicles_to_CBF(self.vehicle, 180, 6, False)

            clone_vehicle_list = [no_input_IDMVehicle.clone_from(self.road_clone, i) for i in original_vehicle_list]

            self.road_clone.vehicles.extend(clone_vehicle_list)


            for index_j in range(int(self.config["simulation_frequency"] // self.config["policy_frequency"])):
                # print(self._get_clone_controlled_vehicle_info())
                # print(self.controlled_vehicle_clone.speed_index)
                record_dict[action_clone]["controlled_vehicle"].append(self._get_clone_controlled_vehicle_info())

                front_vehicle_c_clone, rear_vehicle_c_clone = self.road_clone.neighbour_vehicles(
                    self.controlled_vehicle_clone, self.controlled_vehicle_clone.lane_index)

                record_dict[action_clone]["front_current"].append(
                    self._get_clone_front_vehicles_info(front_vehicle_c_clone))
                record_dict[action_clone]["rear_current"].append(
                    self._get_clone_rear_vehicles_info(rear_vehicle_c_clone))

                front_vehicle_t_clone, rear_vehicle_t_clone = self.road_clone.neighbour_vehicles(
                    self.controlled_vehicle_clone, self.controlled_vehicle_clone.target_lane_index)

                record_dict[action_clone]["front_target"].append(
                    self._get_clone_front_vehicles_info(front_vehicle_t_clone))
                record_dict[action_clone]["rear_target"].append(
                    self._get_clone_rear_vehicles_info(rear_vehicle_t_clone))

                self.road_clone.act()
                self.road_clone.step_pred(1 / self.config["simulation_frequency"])  # update the motion information

                # we also need to predict the states
                if index_j == int(self.config["simulation_frequency"] // self.config["policy_frequency"]) - 1:
                    record_dict[action_clone]["obs_pred"] = self.observation_type.observe_CBF_clone()

            # del self.controlled_vehicle_clone

        return record_dict

    def _get_clone_front_vehicles_info(self, vehicle_clone):
        if vehicle_clone:
            return [vehicle_clone.position[0], vehicle_clone.speed * np.cos(vehicle_clone.heading)]
        else:
            return [self.controlled_vehicle_clone.position[0] + 100, 100]

    def _get_clone_rear_vehicles_info(self, vehicle_clone):
        if vehicle_clone:
            return [vehicle_clone.position[0], vehicle_clone.speed * np.cos(vehicle_clone.heading)]
        else:
            return [self.controlled_vehicle_clone.position[0] - 100, 0]

    def _get_clone_controlled_vehicle_info(self):
        beta = np.arctan(1 / 2 * np.tan(self.controlled_vehicle_clone.action['steering']))

        if self.controlled_vehicle_clone.lane_index[2] == self.controlled_vehicle_clone.target_lane_index[2]:
            status = 0
        elif self.controlled_vehicle_clone.lane_index[2] > self.controlled_vehicle_clone.target_lane_index[2]:
            status = -1
        else:
            status = 1

        return [self.controlled_vehicle_clone.position[0], self.controlled_vehicle_clone.position[1],
                self.controlled_vehicle_clone.speed, self.controlled_vehicle_clone.heading,
                self.controlled_vehicle_clone.action['acceleration'], beta, status,
                self.controlled_vehicle_clone.velocity[0],
                self.controlled_vehicle_clone.velocity[1]]  # we also need to add the lane index

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
               self.steps >= self.config["duration"] or \
               (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


register(
    id='highway_cbf-v0',
    entry_point='customized_highway_env.envs:HighwayEnv_CBF_1',
)
