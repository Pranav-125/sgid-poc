from abc import ABC,abstractmethod
class BaseFliter(ABC):
    @abstractmethod
    def fit(self,baseline_data):
        pass
    @abstractmethod
    def detect(self,value,timestamp=None):
        pass
    @abstractmethod
    def name(self):
        pass