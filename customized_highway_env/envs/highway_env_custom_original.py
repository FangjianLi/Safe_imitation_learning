import numpy as np
from gym.envs.registration import register
from highway_env import utils
from highway_env.utils import near_split
from customized_highway_env.envs.common.abstract_original import AbstractEnv_original
from highway_env.road.road import RoadNetwork
from customized_highway_env.road.road_customized import Road_original
from customized_highway_env.vehicle.controller_customized import ControlledVehicle_original



class HighwayEnv_original(AbstractEnv_original):

    RIGHT_LANE_REWARD: float = 0.1
    HIGH_SPEED_REWARD: float = 0.6  # change from 0.4 to 0.6
    LANE_CHANGE_REWARD: float = 0

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
            "vehicles_count": 20,
            "controlled_vehicles": 1,
            "other_vehicles_type": "customized_highway_env.vehicle.behavior_customized.IDMVehicle_original",
            "initial_lane_id": None,
            "duration": 100,  # we can double check if it is second
            "ego_spacing": 2,
            "vehicles_density": 1.5, ## this is something I changed
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

        self.road = Road_original(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
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
                vehicle = other_vehicles_type.create_random(self.road, speed=np.random.uniform(low=20, high=25),  spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)  ## this is something I changed

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


register(
    id='highway_original-v0',
    entry_point='customized_highway_env.envs:HighwayEnv_original',
)
