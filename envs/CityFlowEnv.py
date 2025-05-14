import json
import cityflow
import gym
import numpy as np
from gymnasium import spaces


class CityFlowEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, configPath, episodeSteps):
        self.steps_per_episode = episodeSteps
        self.is_done = False
        self.current_step = 0
        self.configDict = json.load(open(configPath))
        self.roadnetDict = json.load(open(self.configDict['dir'] + self.configDict['roadnetFile']))
        self.flowDict = json.load(open(self.configDict['dir'] + self.configDict['flowFile']))
        self.intersections = {}

        for intersec in self.roadnetDict['intersections']:
            if not intersec['virtual']:
                incomingLanes = []
                outgoingLanes = []
                directions = []
                for roadLink in intersec['roadLinks']:
                    in_roads = []
                    out_roads = []
                    directions.append(roadLink['direction'])
                    for lane_link in roadLink['laneLinks']:
                        in_roads.append(roadLink['startRoad'] + '_' + str(lane_link['startLaneIndex']))
                        out_roads.append(roadLink['endRoad'] + '_' + str(lane_link['endLaneIndex']))
                    incomingLanes.append(in_roads)
                    outgoingLanes.append(out_roads)
                num_phases = len(intersec['trafficLight']['lightphases'])
                self.intersections[intersec['id']] = [num_phases, incomingLanes, outgoingLanes, directions]

        self.intersectionNames = sorted(list(self.intersections.keys()))
        self.fixed_num_intersections = len(self.intersectionNames)
        self.phase_list = [self.intersections[name][0] for name in self.intersectionNames]
        self.action_space = spaces.MultiDiscrete(np.array(self.phase_list))

        total_dim = 0
        for name in self.intersectionNames:
            for lane_group in self.intersections[name][1]:
                total_dim += 2 * len(lane_group)  

        maxVehicles = len(self.flowDict)
        self.observation_space = spaces.Box(low=0, high=maxVehicles, shape=(total_dim,), dtype=np.int32)
        self.eng = cityflow.Engine(configPath, thread_num=1)
        self.waiting_vehicles_reward = {}

    def step(self, action):
        for i, tl_id in enumerate(self.intersectionNames):
            self.eng.set_tl_phase(tl_id, action[i])

        self.eng.next_step()
        observation = self._get_observation()
        reward = self.getReward()
        self.current_step += 1
        if self.current_step >= self.steps_per_episode:
            self.is_done = True
        return observation, reward, self.is_done, {}

    def reset(self):
        self.eng.reset(seed=False)
        self.is_done = False
        self.current_step = 0
        return self._get_observation()

    def render(self, mode='human'):
        print("Current time:", self.eng.get_current_time())

    def _get_observation(self):
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        obs_list = []
        for name in self.intersectionNames:
            _, incoming, outgoing, _ = self.intersections[name]
            for in_group, out_group in zip(incoming, outgoing):
                for in_lane, out_lane in zip(in_group, out_group):
                    obs_list.append(lane_waiting[in_lane])
                    obs_list.append(lane_waiting[out_lane])
        return np.array(obs_list, dtype=np.int32)

    def getReward(self):
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        total_pressure = 0.0
        for name in self.intersectionNames:
            _, incoming, outgoing, _ = self.intersections[name]
            sum_in = 0
            sum_out = 0
            for group in incoming:
                for lane in group:
                    sum_in += lane_vehicle_count[lane]
            for group in outgoing:
                for lane in group:
                    sum_out += lane_vehicle_count[lane]
            total_pressure += abs(sum_in - sum_out)
        if self.fixed_num_intersections > 0:
            total_pressure /= self.fixed_num_intersections
        return -total_pressure

    def seed(self, seed=None):
        self.eng.set_random_seed(seed)
