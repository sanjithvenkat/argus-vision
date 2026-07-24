from abc import ABC, abstractmethod

class Investigation(ABC):
    """
    Base class for every investigation in Argus Vision.
    """

    @abstractmethod
    def run(self, model, dataset):
        pass