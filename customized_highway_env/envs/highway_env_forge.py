import numpy as np
from gym.envs.registration import register
from highway_env import utils
from highway_env.utils import near_split
from customized_highway_env.envs.common.abstract_original import AbstractEnv_original
from highway_env.road.road import RoadNetwork
from customized_highway_env.road.road_customized import Road_original
from customized_highway_env.vehicle.controller_customized import ControlledVehicle_original
import pandas as pd

def get_the_info_to_plot(env):
    feature_needed = ['x', 'y', 'heading']
    # we want to record the x-pos, y-pos, heading
    vehicle_list = env.road.vehicles
    df = pd.DataFrame()
    df = df.append(pd.DataFrame.from_records(
        [v.to_dict() for v in vehicle_list])[feature_needed], ignore_index=True)
    return df.values.copy()



class HighwayEnv_forge(AbstractEnv_original):

    RIGHT_LANE_REWARD: float = 0.1
    HIGH_SPEED_REWARD: float = 0.6  # change from 0.4 to 0.6
    LANE_CHANGE_REWARD: float = 0

    SPEED_SPREAD = np.array([2, -2, 0, 2]) + 17
    POS_SPREAD = np.array([25, 30, 25, 30]) + 10

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics_original",
                "vehicles_count": 7, # specific environment
            },
            "action": {
                "type": "DiscreteMetaAction_original",
            },
            "lanes_count": 4,
            "vehicles_count": 4, # other cars for this case, we need to change it to 4
            "controlled_vehicles": 1,
            "other_vehicles_type": "customized_highway_env.vehicle.behavior_customized.IDMVehicle_original",
            "initial_lane_id": None,
            "duration": 100,  # we can double check if it is second
            "ego_spacing": 2,
            "vehicles_density": 1.5,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False,
            "simulation_frequency": 10,
            "policy_frequency": 2,
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:

        self.road = Road_original(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road,
                lane_index=("0", "1", 2),
                longitudinal=10,
                speed=25
            ) #instead of
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)



            for index_c in range(others):
                self.road.vehicles.append(
                    other_vehicles_type.make_on_lane(self.road, lane_index=("0", "1", index_c),
                longitudinal=self.POS_SPREAD[index_c],
                speed=self.SPEED_SPREAD[index_c] )
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

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
               self.steps >= self.config["duration"] or \
               (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


    def _info(self, obs, action) -> dict:

        # here, we just want to return one thing: the
        return get_the_info_to_plot(self)


register(
    id='highway_forge-v0',
    entry_point='customized_highway_env.envs:HighwayEnv_forge',
)
