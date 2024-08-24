from __future__ import annotations
from .client import Client
from ..poker.no_limit_texas_holdem import NoLimitTexasHoldem
from ..poker.component.action import Action, ActionType

class Room():
    def __init__(
        self,
        id: str,
    ) -> None:
        self.id = id
        self.clients: set[Client] = set()
        self.ready_clients: list[Client] = []
        self.playing_clients: list[Client] = []
        self.game = NoLimitTexasHoldem(
            num_players=2,
            num_runs=1,
            raise_pot_size=[1],
            legal_raise_pot_size=[1],
            street_start_player=1,
        )

    def is_playing(self) -> bool:
        return len(self.playing_clients) == self.game.num_players

    def query_action(self) -> None:
        obs = self.game.observe_current()
        if obs.is_over:
            self.game_over()
        else:
            self.playing_clients[obs.player].handle_event(event_type='info', message=obs)

    def game_start(self) -> None:
        assert len(self.ready_clients) == self.game.num_players
        assert len(self.playing_clients) == 0
        for client in self.clients:
            client.handle_event(event_type='info', message='Game start.')
        for client in self.ready_clients:
            self.playing_clients.append(client)
        self.ready_clients = []
        self.game.reset()
        self.query_action()
    
    def game_over(self) -> None:
        for client in self.clients:
            client.handle_event(event_type='info', message='Game over.')
        self.ready_clients = []
        self.playing_clients = []

    def ready(self, client: Client) -> None:
        if self.is_playing():
            client.handle_event(event_type='error', message='Game is full.')
            return
        if client in self.ready_clients:
            client.handle_event(event_type='info', message='You are ready.')
            return
        self.ready_clients.append(client)
        client.handle_event(event_type='info', message='You are ready.')
        if len(self.ready_clients) == self.game.num_players:
            self.game_start()

    def step(self, client: Client, action: str) -> None:
        if client not in self.playing_clients:
            client.handle_event(event_type='error', message='You are not in the game.')
            return
        obs = self.game.step_str(action)
        if obs is None:
            client.handle_event(event_type='error', message='Invalid action.')
        self.query_action()
    
    def enter(self, client: Client) -> None:
        self.clients.add(client)
    
    def leave(self, client: Client) -> None:
        if client in self.playing_clients:
            self.playing_clients.remove(client)
            self.game_over()
        if client in self.ready_clients:
            self.ready_clients.remove(client)
        self.clients.remove(client)
