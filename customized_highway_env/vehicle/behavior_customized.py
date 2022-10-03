from highway_env.vehicle.behavior import IDMVehicle
import numpy as np
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class IDMVehicle_original(IDMVehicle):
    @classmethod
    def create_random(cls, road,
                      speed=None,
                      lane_from=None,
                      lane_to=None,
                      lane_id=None,
                      spacing=1):

        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(0.7 * lane.speed_limit, lane.speed_limit)
            else:
                speed = road.np_random.uniform(IDMVehicle_original.DEFAULT_SPEEDS[0],
                                               IDMVehicle_original.DEFAULT_SPEEDS[1])
        default_spacing = 15 + 1.2 * speed
        offset = spacing * default_spacing * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))

        if not len(road.vehicles):
            x0 = 3 * offset
            x0 += offset * road.np_random.uniform(0.9, 1.1)
        elif np.random.rand(1) > 0.2:
            x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles])
            x0 += offset * road.np_random.uniform(0.9, 1.1)
        else:
            x0 = np.min([lane.local_coordinates(v.position)[0] for v in road.vehicles])
            x0 -= offset * road.np_random.uniform(0.9, 1.1)

        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)

        return v



class no_input_IDMVehicle(ControlledVehicle):

    def __init__(self,
                 road,
                 position,
                 heading: float = 0,
                 speed: float = 0):
        super().__init__(road, position, heading, speed)

    def randomize_behavior(self):
        pass

    @classmethod
    def create_from(cls,  vehicle: IDMVehicle) -> "no_input_vehicle":

        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed)
        return v

    @classmethod
    def clone_from(cls, road_clone, vehicle: IDMVehicle) -> "no_input_vehicle":

        v = cls(road_clone, vehicle.position, heading=vehicle.heading, speed=vehicle.speed)
        return v

    def act(self, action = None):

        action = {}
        action['steering'] = 0
        action['acceleration'] = 0
        Vehicle.act(self, action)  # Skip ControlledVehicle.act(), or the command will be overriden.

    def step(self, dt: float):

        super().step(dt)