from abc import ABC, abstractmethod

class Workflow(ABC):
    @abstractmethod
    def run(self):
        """Execute the workflow."""
        pass

