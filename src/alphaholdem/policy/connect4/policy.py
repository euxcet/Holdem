import numpy as np
import random
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import restore_original_dimensions


class HeuristicBase(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()

    def learn_on_batch(self, samples):
        pass

    @override(Policy)
    def get_weights(self):
        return {}

    @override(Policy)
    def set_weights(self, weights):
        pass

    @override(Policy)
    def export_model(self, export_dir: str, onnx=None) -> None:
        pass

    @override(Policy)
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        obs_batch = restore_original_dimensions(
            np.array(obs_batch, dtype=np.float32), self.observation_space, tensorlib=np
        )
        return self._do_compute_actions(obs_batch)

    def pick_legal_action(self, legal_action):
        legal_choices = np.arange(len(legal_action))[legal_action == 1]
        return np.random.choice(legal_choices)


class AlwaysSameHeuristic(HeuristicBase):
    _rand_choice = random.choice(range(7))

    def _do_compute_actions(self, obs_batch):
        def select_action(legal_action):
            legal_choices = np.arange(len(legal_action))[legal_action == 1]

            if self._rand_choice not in legal_choices:
                self._rand_choice = np.random.choice(legal_choices)

            return self._rand_choice

        return [select_action(x) for x in obs_batch["action_mask"]], [], {}


class LinearHeuristic(HeuristicBase):
    _rand_choice = random.choice(range(7))
    _rand_sign = np.random.choice([-1, 1])

    def _do_compute_actions(self, obs_batch):
        def select_action(legal_action):
            legal_choices = np.arange(len(legal_action))[legal_action == 1]

            self._rand_choice += 1 * self._rand_sign

            if self._rand_choice not in legal_choices:
                self._rand_choice = np.random.choice(legal_choices)

            return self._rand_choice

        return [select_action(x) for x in obs_batch["action_mask"]], [], {}


class BeatLastHeuristic(HeuristicBase):
    def _do_compute_actions(self, obs_batch):
        def select_action(legal_action, observation):
            legal_choices = np.arange(len(legal_action))[legal_action == 1]

            obs_sums = np.sum(observation, axis=0)

            desired_actions = np.squeeze(np.argwhere(obs_sums[:, 0] < obs_sums[:, 1]))
            if desired_actions.size == 0:
                return np.random.choice(legal_choices)

            if desired_actions.size == 1:
                desired_action = desired_actions[()]
            else:
                desired_action = np.random.choice(desired_actions)
            if desired_action in legal_choices:
                return desired_action

            return np.random.choice(legal_choices)

        return (
            [
                select_action(x, y)
                for x, y in zip(obs_batch["action_mask"], obs_batch["observation"])
            ],
            [],
            {},
        )


class RandomHeuristic(HeuristicBase):
    def _do_compute_actions(self, obs_batch):
        return [self.pick_legal_action(x) for x in obs_batch["action_mask"]], [], {}


class SmartHeuristic(HeuristicBase):
    row_count = 6
    column_count = 7
    win_count = 4

    def check_window(self, window, num_discs, mark):
        return (
            window.count(mark) == num_discs
            and window.count(0) == self.win_count - num_discs
        )

    def count_windows(self, grid, num_discs, mark):
        num_windows = 0
        for row in range(self.row_count):
            for col in range(self.column_count - self.win_count - 1):
                window = list(grid[row, col : col + self.win_count])
                if self.check_window(window, num_discs, mark):
                    num_windows += 1
        for row in range(self.row_count - self.win_count + 1):
            for col in range(self.column_count):
                window = list(grid[row : row + self.win_count, col])
                if self.check_window(window, num_discs, mark):
                    num_windows += 1

        for row in range(self.row_count - self.win_count + 1):
            for col in range(self.column_count - self.win_count + 1):
                negative_window = list(
                    grid[
                        range(row, row + self.win_count),
                        range(col, col + self.win_count),
                    ]
                )
                if self.check_window(negative_window, num_discs, mark):
                    num_windows += 1

                positive_window = list(
                    grid[
                        range(
                            self.row_count - row - 1,
                            self.row_count - row - self.win_count - 1,
                            -1,
                        ),
                        range(
                            self.column_count - col - self.win_count,
                            self.column_count - col,
                        ),
                    ]
                )
                if self.check_window(positive_window, num_discs, mark):
                    num_windows += 1

        return num_windows

    def get_heuristic(self, grid, mark):
        num_threes = self.count_windows(grid, 3, mark)
        num_fours = self.count_windows(grid, 4, mark)
        num_threes_opp = self.count_windows(grid, 3, mark % 2 + 1)
        score = num_threes - 1e2 * num_threes_opp + 1e6 * num_fours
        return score

    def drop_piece(self, grid, col, mark):
        next_grid = grid.copy()
        for row in range(self.row_count - 1, -1, -1):
            if next_grid[row, col] == 0:
                break
        next_grid[row, col] = mark
        return next_grid

    def select_action(self, legal_action, observation):
        legal_choices = np.arange(len(legal_action))[legal_action == 1]

        my_grid = np.array(observation[:, :, 0] + observation[:, :, 1] * 2, dtype=int)

        mark_1 = np.sum(my_grid == 1)
        mark_2 = np.sum(my_grid == 2)

        mark = 1 if mark_1 == mark_2 else 2

        scores = []
        for i in range(7):
            future_grid = self.drop_piece(my_grid, i, mark)
            scores.append(self.get_heuristic(future_grid, mark))

        scores = np.array(scores, dtype=int)
        if np.all(scores == 0):
            return np.random.choice(legal_choices)

        desired_action = [
            index for index, item in enumerate(scores) if item == max(scores)
        ]

        if len(desired_action) > 1:
            desired_action = np.random.choice(desired_action)
        else:
            desired_action = desired_action[0]

        if desired_action in legal_choices:
            return desired_action
        else:
            return np.random.choice(legal_choices)

    def _do_compute_actions(self, obs_batch):
        return (
            [
                self.select_action(x, y)
                for x, y in zip(obs_batch["action_mask"], obs_batch["observation"])
            ],
            [],
            {},
        )