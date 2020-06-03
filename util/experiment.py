import pathlib
import os
from config import config

class Experiment():

    def __init__(self, run_n=0, **kwargs ):
        self.run_n = run_n
        self.setup( **kwargs )

    def setup(self, fig_dir=False, result_dir=False, tensorboard_dir=False, model_dir=False):

        if fig_dir:
            self.fig_dir = os.path.join(config['plot_dir'], 'run_' + str(self.run_n), 'analysis_image')
            pathlib.Path(self.fig_dir).mkdir(parents=True, exist_ok=True)

        if result_dir:
            self.result_dir = os.path.join(config['result_dir'], 'run_' + str(self.run_n))
            pathlib.Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        if tensorboard_dir:
            self.tensorboard_dir = os.path.join(config['tensorboard_dir'],'run_' + str(self.run_n))
            pathlib.Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)

        if model_dir:
            self.model_dir = os.path.join(config['model_dir'],'run_' + str(self.run_n))
            pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)
