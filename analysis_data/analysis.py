from abc import ABCMeta, abstractmethod

class Analysis(metaclass=ABCMeta):

    def __init__(self,data_name,do,fig_dir):
        self.data_name = data_name
        self.fig_name = data_name.replace(" ", "_")
        self.do = do
        self.fig_dir = fig_dir

    @abstractmethod
    def analyze(self, data):
        pass

    def update_name(self, new_name):
        self.data_name = new_name
        self.fig_name = new_name.replace(" ", "_")

