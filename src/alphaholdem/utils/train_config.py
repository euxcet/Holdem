from __future__ import annotations

class TrainConfig():
    class TrainGameConfig():
        def __init__(
            self,
            type: str,
            num_players: int,
            showdown_street: str,
            legal_raise_pot_size: list[float],
            num_runs: int,
            initial_chips: int,
            payoff_max: int,
            circular_train: bool,
            custom_board_cards: list[str],
        ) -> None:
            self.type = type
            self.num_players = num_players
            self.showdown_street = showdown_street
            self.legal_raise_pot_size = legal_raise_pot_size
            self.num_runs = num_runs
            self.initial_chips = initial_chips
            self.payoff_max = payoff_max
            self.circular_train = circular_train
            self.custom_board_cards = custom_board_cards

        @staticmethod
        def load_from_dict(data: dict) -> TrainConfig.TrainGameConfig:
            return TrainConfig.TrainGameConfig(
                type=data['type'],
                num_players=data['num_players'],
                showdown_street=data['showdown_street'],
                legal_raise_pot_size=data['legal_raise_pot_size'],
                num_runs=data['num_runs'],
                initial_chips=data['initial_chips'],
                payoff_max=data['payoff_max'],
                circular_train=data['circular_train'],
                custom_board_cards=data['custom_board_cards'],
            )

    class TrainResourcesConfig():
        def __init__(
            self,
            num_cpus: int,
            num_gpus: int,
            num_rollout_workers: int,
            num_envs_per_worker: int,
            num_cpus_per_worker: float,
            num_gpus_per_worker: float,
            num_gpus_algorithm: float,
            num_learner_workers: int,
        ) -> None:
            self.num_cpus = num_cpus
            self.num_gpus = num_gpus
            self.num_rollout_workers = num_rollout_workers
            self.num_envs_per_worker = num_envs_per_worker
            self.num_cpus_per_worker = num_cpus_per_worker
            self.num_gpus_per_worker = num_gpus_per_worker
            self.num_gpus_algorithm = num_gpus_algorithm
            self.num_learner_workers = num_learner_workers
        
        @staticmethod
        def load_from_dict(data: dict) -> TrainConfig.TrainResourcesConfig:
            return TrainConfig.TrainResourcesConfig(
                num_cpus=data['num_cpus'],
                num_gpus=data['num_gpus'],
                num_rollout_workers=data['num_rollout_workers'],
                num_envs_per_worker=data['num_envs_per_worker'],
                num_cpus_per_worker=data['num_cpus_per_worker'],
                num_gpus_per_worker=data['num_gpus_per_worker'],
                num_gpus_algorithm=data['num_gpus_algorithm'],
                num_learner_workers=data['num_learner_workers'],
            )
    
    class TrainHyperConfig():
        def __init__(
            self,
            learning_rate: float,
            clip_param: float,
            entropy_coeff: float,
            train_batch_size: int,
            sgd_minibatch_size: int,
            num_sgd_iter: int,
            kl_coeff: float,
            kl_target: float,
            vf_clip_param: float,
            vf_loss_coeff: float,
            checkpoint_frequency: int,
            checkpoint_num_to_keep: int,
            opponent_count: int,
            num_update_iter: int,
            win_rate_threshold: int,
        ) -> None:
            self.learning_rate = learning_rate
            self.clip_param = clip_param
            self.entropy_coeff = entropy_coeff
            self.train_batch_size = train_batch_size
            self.sgd_minibatch_size = sgd_minibatch_size
            self.num_sgd_iter = num_sgd_iter
            self.kl_coeff = kl_coeff
            self.kl_target = kl_target
            self.vf_clip_param = vf_clip_param
            self.vf_loss_coeff = vf_loss_coeff
            self.checkpoint_frequency = checkpoint_frequency
            self.checkpoint_num_to_keep = checkpoint_num_to_keep
            self.opponent_count = opponent_count
            self.num_update_iter = num_update_iter
            self.win_rate_threshold = win_rate_threshold
        
        @staticmethod
        def load_from_dict(data: dict) -> TrainConfig.TrainHyperConfig:
            return TrainConfig.TrainHyperConfig(
                learning_rate=data['learning_rate'],
                clip_param=data['clip_param'],
                entropy_coeff=data['entropy_coeff'],
                train_batch_size=data['train_batch_size'],
                sgd_minibatch_size=data['sgd_minibatch_size'],
                num_sgd_iter=data['num_sgd_iter'],
                kl_coeff=data['kl_coeff'],
                kl_target=data['kl_target'],
                vf_clip_param=data['vf_clip_param'],
                vf_loss_coeff=data['vf_loss_coeff'],
                checkpoint_frequency=data['checkpoint_frequency'],
                checkpoint_num_to_keep=data['checkpoint_num_to_keep'],
                opponent_count=data['opponent_count'],
                num_update_iter=data['num_update_iter'],
                win_rate_threshold=data['win_rate_threshold'],
            )

    def __init__(
        self,
        game: TrainGameConfig,
        resources: TrainResourcesConfig,
        hyper: TrainHyperConfig,
    ) -> None:
        self.game = game
        self.resources = resources
        self.hyper = hyper

    @staticmethod
    def load_from_dict(data: dict):
        return TrainConfig(
            game=TrainConfig.TrainGameConfig.load_from_dict(data['game']),
            resources=TrainConfig.TrainResourcesConfig.load_from_dict(data['resources']),
            hyper=TrainConfig.TrainHyperConfig.load_from_dict(data['hyper']),
        )
