import itertools
import math
import os
import random

import numpy as np
from tqdm import tqdm


class CoNE:
    def __init__(self,
                 grid_dimension: int,
                 grid_size: int,
                 num_total_pits: int,
                 num_clusters: int,
                 distribute_pits_evenly: bool,
                 max_steps_per_episode: int,
                 load_terminal_states: bool = False,
                 save_terminal_states: bool = False,
                 randomize_initial_state: bool = False,
                 small_state: bool = True,
                 generate_all_interior_pits=False,
                 use_random_pits: bool = False):
        self.grid_dimension = grid_dimension
        self.grid_size = grid_size
        self.num_total_pits = num_total_pits
        self.num_pit_clusters = num_clusters
        self.distribute_pits_evenly = distribute_pits_evenly
        self.max_steps_per_episode = max_steps_per_episode
        self.randomize_initial_state = randomize_initial_state
        self.generate_all_interior_pits = generate_all_interior_pits
        self.use_random_pits = use_random_pits
        self.grid = {}
        self.episode_reward = 0
        self.episode_length = 0
        self.current_location = np.zeros(grid_dimension, dtype=int)
        self.action_size = 2 * grid_dimension
        self.total_actions = 2 ** self.action_size
        self.small_state = small_state
        self.episode_steps = []

        run_directory = f"{grid_dimension}-{grid_size}-{num_total_pits}-90"
        terminal_states_file = f"envs/pit_locations/{run_directory}/terminal_states.npy"
        load_terminal_states = load_terminal_states if num_total_pits > 5 else False

        if load_terminal_states:
            self.terminal_states = self._load_terminal_states(terminal_states_file)
        else:
            self.terminal_states = self._place_goal_and_pits()
            if save_terminal_states:
                self._save_terminal_states(terminal_states_file)
                print(f"Saved terminal states to {terminal_states_file}")

        self._setup_terminal_states()
        print(f'Created gridworld with {len(self.terminal_states)} terminal states')

    def _save_terminal_states(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.terminal_states)

    @staticmethod
    def _load_terminal_states(filepath: str) -> np.ndarray:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Terminal states file not found at {filepath}")
        return np.load(filepath)

    def reset(self):
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_steps = []

        if self.randomize_initial_state:
            terminal_tuples = {tuple(state) for state in self.terminal_states}
            self.current_location = np.random.randint(0, self.grid_size, size=self.grid_dimension)
            while tuple(self.current_location) in terminal_tuples:
                self.current_location = np.random.randint(0, self.grid_size, size=self.grid_dimension)
        else:
            self.current_location = np.zeros(self.grid_dimension, dtype=int)

        if self.small_state:
            observation = self.current_location.copy() 
        else:
            distances = np.array([np.linalg.norm(self.current_location - ts) for ts in self.terminal_states])
            observation = np.append(self.current_location, distances)

        info = {} 
        return observation, info 

    def get_cell_value(self):
        distance_from_goal = np.linalg.norm(self.current_location - self.terminal_states[0])
        return self.grid.get(tuple(self.current_location), -distance_from_goal)

    def _get_pit_starting_points(self, goal):
        factors = (np.linspace(1 / (self.num_pit_clusters + 1), 1, self.num_pit_clusters) if self.distribute_pits_evenly
                   else np.random.choice(np.linspace(0, 1, 100), self.num_pit_clusters, replace=False))

        return [np.array([max(1, min(int(self.current_location[j] + (goal[j] - self.current_location[j]) * factor),
                                     self.grid_size - 2)) for j in range(self.grid_dimension)])
                for factor in factors]

    def _place_goal_and_pit_clusters(self, goal):
        deltas = [delta for delta in itertools.product([-1, 0, 1], repeat=self.grid_dimension)
                  if any(d != 0 for d in delta)]

        pit_starting_locations = self._get_pit_starting_points(goal)
        total_pits_to_place = self.num_total_pits
        pbar = tqdm(total=total_pits_to_place, desc="Adding pits")

        cluster_pits = []
        occupied_positions = {tuple(goal)}
        num_starting_locations = len(pit_starting_locations)

        if num_starting_locations == 0 and total_pits_to_place > 0:
            pbar.close()
            print(f"Warning: No valid starting locations found for pits, but {total_pits_to_place} requested.")
            return [np.array(goal)]

        for idx, pit in enumerate(pit_starting_locations):
            pit_tup = tuple(pit)
            
            if pit_tup in occupied_positions or not all(0 <= coord < self.grid_size for coord in pit):
                print(f"Warning: Skipping invalid starting pit {pit_tup}")
                continue

            cluster_pits.append(pit)
            occupied_positions.add(pit_tup)
            pbar.update(1)

            pits_placed_so_far = pbar.n
            pits_left_to_place_total = total_pits_to_place - pits_placed_so_far
            if pits_left_to_place_total <= 0:
                break

            clusters_remaining = num_starting_locations - (idx + 1)
            clusters_to_distribute_over = max(1, clusters_remaining + 1) 

            num_pits_for_this_cluster = math.ceil(pits_left_to_place_total / clusters_to_distribute_over)
            additional_pits_in_cluster = min(num_pits_for_this_cluster, pits_left_to_place_total)

            current_cluster_pits = [pit]

            for _ in range(additional_pits_in_cluster):
                possible_neighbors = set()
                for current_pit in current_cluster_pits:
                    for delta in deltas:
                        neighbor_coords = tuple(current_pit[j] + delta[j] for j in range(self.grid_dimension))
                        
                        if all(0 <= c < self.grid_size for c in neighbor_coords) and neighbor_coords not in occupied_positions:
                            possible_neighbors.add(neighbor_coords)

                if not possible_neighbors:
                    break 

                new_pit_tup = random.choice(list(possible_neighbors))
                new_pit_arr = np.array(new_pit_tup)

                cluster_pits.append(new_pit_arr)
                occupied_positions.add(new_pit_tup)
                current_cluster_pits.append(new_pit_arr) 
                pbar.update(1)

                if pbar.n >= total_pits_to_place: break 

            if pbar.n >= total_pits_to_place: break 

        if pbar.n < self.num_total_pits:
            print(f"Warning: Only placed {pbar.n}/{self.num_total_pits} pits.")

        pbar.close()
        return [np.array(goal)] + cluster_pits

    def _place_goal_and_pits(self):
        goal = np.array([self.grid_size - 1] * self.grid_dimension, dtype=int)

        if self.generate_all_interior_pits:
            return self._generate_all_interior_pits()

        if self.use_random_pits and self.num_total_pits > 0:
            return self._place_goal_and_random_pits(goal)

        return [goal] if self.num_total_pits == 0 else self._place_goal_and_pit_clusters(goal)

    def _place_goal_and_random_pits(self, goal: np.ndarray) -> list[np.ndarray]:
        D = self.grid_dimension
        S = self.grid_size
        P = self.num_total_pits
        N_int = (S - 2) ** D

        if P > N_int:
            raise ValueError(f"Cannot place {P} pits in only {N_int} interior cells")

        flat_idx = np.random.choice(N_int, size=P, replace=False)
        strides = (S - 2,) * D
        pits = []
        for idx in flat_idx:
            coord = np.unravel_index(idx, strides)  
            pits.append(np.array(coord, dtype=int) + 1)

        return [goal] + pits

    def _generate_all_interior_pits(self) -> list[np.ndarray]:
        goal_location = np.array([self.grid_size - 1] * self.grid_dimension, dtype=int)
        terminal_states = [goal_location]

        if self.grid_size >= 3:
            interior_coord_range = range(1, self.grid_size - 1)
            interior_pit_iterator = itertools.product(interior_coord_range, repeat=self.grid_dimension)

            pit_locations = [np.array(coords, dtype=int) for coords in interior_pit_iterator]
            terminal_states.extend(pit_locations)

            expected_pits = (self.grid_size - 2)**self.grid_dimension
            if len(pit_locations) != expected_pits:
                print(f"Warning: Pit count mismatch! Expected {expected_pits}, generated {len(pit_locations)}")

        return terminal_states

    def _setup_terminal_states(self):
        origin = np.zeros(self.grid_dimension, dtype=int)
        max_distance_from_goal = np.linalg.norm(origin - self.terminal_states[0])
        
        max_distance_from_goal = max(max_distance_from_goal, 1.0)

        self.grid = {} 
        goal_state_tuple = tuple(self.terminal_states[0])

        for state in self.terminal_states:
            state_tuple = tuple(state)
            if state_tuple == goal_state_tuple:
                self.grid[state_tuple] = 10.0 
            else:
                 
                self.grid[state_tuple] = max_distance_from_goal * -10.0
    
    def compute_action_from_index(self, idx):
        return np.array(np.unravel_index(idx, (2,) * (2 * self.grid_dimension))).reshape(-1)

    def step(self, action):
        self.episode_steps.append((self.current_location.copy(), action))  
        self.episode_length += 1
        movement = np.zeros(self.grid_dimension, dtype=int)
        for i in range(self.grid_dimension):
            if action[2 * i] == 1 and action[2 * i + 1] == 0:
                movement[i] = 1
            if action[2 * i] == 0 and action[2 * i + 1] == 1:
                movement[i] = -1

        new_location = np.clip(self.current_location + movement, 0, self.grid_size - 1)

        if self.small_state:
            observation = new_location.copy()  
        else:
            distances = np.array([np.linalg.norm(new_location - ts) for ts in self.terminal_states])
            observation = np.append(new_location.copy(), distances)

        self.current_location = new_location

        reward = self.get_cell_value()  
        self.episode_reward += reward

        in_terminal_state = tuple(self.current_location) in map(tuple, self.terminal_states)
        is_terminal_flag_for_info = in_terminal_state or self.episode_length >= self.max_steps_per_episode

        terminated = in_terminal_state
        truncated = self.episode_length >= self.max_steps_per_episode

        info = {}
        if is_terminal_flag_for_info:  
            
            final_obs_for_info = observation
            info = {
                'final_info': {  
                    'episode': {
                        'r': self.episode_reward,
                        'l': self.episode_length,
                        'final_observation': final_obs_for_info,
                        'episode_steps': self.episode_steps,  
                    }
                }
            }

        return observation, reward, terminated, truncated, info
