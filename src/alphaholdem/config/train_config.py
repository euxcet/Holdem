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
            num_action: int,
        ) -> None:
            from ..poker.component.street import Street
            self.type = type
            self.num_players = num_players
            self.showdown_street = Street.from_str(showdown_street)
            self.legal_raise_pot_size = legal_raise_pot_size
            self.num_runs = num_runs
            self.initial_chips = initial_chips
            self.payoff_max = payoff_max
            self.circular_train = circular_train
            self.custom_board_cards = custom_board_cards
            self.num_action = num_action

        @staticmethod
        def load_from_dict(data: dict | None) -> TrainConfig.TrainGameConfig:
            if data is None: return None
            return TrainConfig.TrainGameConfig(
                type=data.get('type'),
                num_players=data.get('num_players'),
                showdown_street=data.get('showdown_street'),
                legal_raise_pot_size=data.get('legal_raise_pot_size'),
                num_runs=data.get('num_runs'),
                initial_chips=data.get('initial_chips'),
                payoff_max=data.get('payoff_max'),
                circular_train=data.get('circular_train'),
                custom_board_cards=data.get('custom_board_cards'),
                num_action=data.get('num_action'),
            )

    class TrainResourcesConfig():
        def __init__(
            self,
            num_cpus: int,
            num_gpus: int,
        ) -> None:
            self.num_cpus = num_cpus
            self.num_gpus = num_gpus
        
        @staticmethod
        def load_from_dict(data: dict | None) -> TrainConfig.TrainResourcesConfig:
            if data is None: return None
            return TrainConfig.TrainResourcesConfig(
                num_cpus=data.get('num_cpus'),
                num_gpus=data.get('num_gpus'),
            )
    
    class TrainHyperConfig():
        def __init__(
            self,
            model: str,
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
            num_update_iter: int,
            win_rate_threshold: int,
            stop_timesteps_total: int,
            stop_training_iteration: int,
        ) -> None:
            self.model = model
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
            self.num_update_iter = num_update_iter
            self.win_rate_threshold = win_rate_threshold
            self.stop_timesteps_total = stop_timesteps_total
            self.stop_training_iteration = stop_training_iteration
        
        @staticmethod
        def load_from_dict(data: dict | None) -> TrainConfig.TrainHyperConfig:
            if data is None: return None
            return TrainConfig.TrainHyperConfig(
                model=data.get('model'),
                learning_rate=data.get('learning_rate'),
                clip_param=data.get('clip_param'),
                entropy_coeff=data.get('entropy_coeff'),
                train_batch_size=data.get('train_batch_size'),
                sgd_minibatch_size=data.get('sgd_minibatch_size'),
                num_sgd_iter=data.get('num_sgd_iter'),
                kl_coeff=data.get('kl_coeff'),
                kl_target=data.get('kl_target'),
                vf_clip_param=data.get('vf_clip_param'),
                vf_loss_coeff=data.get('vf_loss_coeff'),
                checkpoint_frequency=data.get('checkpoint_frequency'),
                checkpoint_num_to_keep=data.get('checkpoint_num_to_keep'),
                num_update_iter=data.get('num_update_iter'),
                win_rate_threshold=data.get('win_rate_threshold'),
                stop_timesteps_total=data.get('stop_timesteps_total'),
                stop_training_iteration=data.get('stop_training_iteration'),
            )

    class TrainRLlibConfig():
        def __init__(
            self,
            num_rollout_workers: int,
            num_envs_per_worker: int,
            num_cpus_per_worker: float,
            num_gpus_per_worker: float,
            num_gpus_algorithm: float,
            num_learner_workers: int,
            framework: str,
            log_level: str,
            evaluation_interval: int,
            verbose: int,
        ) -> None:
            self.num_rollout_workers = num_rollout_workers
            self.num_envs_per_worker = num_envs_per_worker
            self.num_cpus_per_worker = num_cpus_per_worker
            self.num_gpus_per_worker = num_gpus_per_worker
            self.num_gpus_algorithm = num_gpus_algorithm
            self.num_learner_workers = num_learner_workers
            self.framework = framework
            self.log_level = log_level
            self.evaluation_interval = evaluation_interval
            self.verbose = verbose
            
        @staticmethod
        def load_from_dict(data: dict | None) -> TrainConfig.TrainRLlibConfig:
            if data is None: return None
            return TrainConfig.TrainRLlibConfig(
                num_rollout_workers=data.get('num_rollout_workers'),
                num_envs_per_worker=data.get('num_envs_per_worker'),
                num_cpus_per_worker=data.get('num_cpus_per_worker'),
                num_gpus_per_worker=data.get('num_gpus_per_worker'),
                num_gpus_algorithm=data.get('num_gpus_algorithm'),
                num_learner_workers=data.get('num_learner_workers'),
                framework=data.get('framework'),
                log_level=data.get('log_level'),
                evaluation_interval=data.get('evaluation_interval'),
                verbose=data.get('verbose'),
            )
    
    class TrainSelfPlayConfig():
        def __init__(
            self,
            type: str,
            arena: str,
            policy_type: str,
            rule_based_policies: list[str],
            num_opponent_limit: int,
            num_update_iter: int,
            arena_runs: int,
        ) -> None:
            self.type = type
            self.arena = arena
            self.policy_type = policy_type
            self.rule_based_policies = rule_based_policies
            self.num_opponent_limit = num_opponent_limit
            self.num_update_iter = num_update_iter
            self.arena_runs = arena_runs
        
        @staticmethod
        def load_from_dict(data: dict | None) -> TrainConfig.TrainSelfPlayConfig:
            if data is None: return None
            return TrainConfig.TrainSelfPlayConfig(
                type=data.get('type'),
                arena=data.get('arena'),
                policy_type=data.get('policy_type'),
                rule_based_policies=data.get('rule_based_policies'),
                num_opponent_limit=data.get('num_opponent_limit'),
                num_update_iter=data.get('num_update_iter'),
                arena_runs=data.get('arena_runs'),
            )

    class TrainPolicyConfig():
        def __init__(
            self,
            alphaholdem: str,
            leduc_nash: str,
            kuhn_nash: str,
        ) -> None:
            self.alphaholdem = alphaholdem
            self.leduc_nash = leduc_nash
            self.kuhn_nash = kuhn_nash

        @staticmethod
        def load_from_dict(data: dict | None) -> TrainConfig.TrainPolicyConfig:
            if data is None: return None
            return TrainConfig.TrainPolicyConfig(
                alphaholdem=data.get('alphaholdem'),
                leduc_nash=data.get('leduc_nash'),
                kuhn_nash=data.get('kuhn_nash'),
            )
    
    def __init__(
        self,
        game: TrainGameConfig,
        resources: TrainResourcesConfig,
        hyper: TrainHyperConfig,
        rllib: TrainRLlibConfig,
        self_play: TrainSelfPlayConfig,
        policy: TrainPolicyConfig,
    ) -> None:
        self.game = game
        self.resources = resources
        self.hyper = hyper
        self.rllib = rllib
        self.self_play = self_play
        self.policy = policy

    @staticmethod
    def load_from_dict(data: dict | None) -> TrainConfig:
        if data is None: return None
        return TrainConfig(
            game=TrainConfig.TrainGameConfig.load_from_dict(data.get('game')),
            resources=TrainConfig.TrainResourcesConfig.load_from_dict(data.get('resources')),
            hyper=TrainConfig.TrainHyperConfig.load_from_dict(data.get('hyper')),
            rllib=TrainConfig.TrainRLlibConfig.load_from_dict(data.get('rllib')),
            self_play=TrainConfig.TrainSelfPlayConfig.load_from_dict(data.get('self_play')),
            policy=TrainConfig.TrainPolicyConfig.load_from_dict(data.get('policy')),
        )
