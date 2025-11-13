import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List

class FAABase(ABC):
    """
    Abstract Base Class for a Flexibility Asset Agent (FAA).
    """
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type

    @abstractmethod
    def get_baseline_operation(self, sim_state: Dict[str, Any]) -> np.ndarray:
        """
        Calculates the agent's self-determined optimal power consumption
        profile over the upcoming time horizon.
        """
        raise NotImplementedError

    @abstractmethod
    def get_flexibility_bids(self, sim_state: Dict[str, Any]) -> List[Dict]:
        """
        Generates a list of all possible flexibility bids this agent can offer.
        """
        raise NotImplementedError