from lbforaging.foraging import Agent
from lbforaging.foraging.environment import Action


class IdleAgent(Agent):
    name = "Idle Agent"

    def _act(self, obs):
        return Action.NONE
