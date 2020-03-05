import numpy as np
import math
from util_plotting import *


class Analysis( object ):

    def __init__(self, data_name ):
        self.data_name = data_name
        self.fig_name = data_name.replace(" ","_")
