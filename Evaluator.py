import numpy as np
# from Song import Song

class Evaluator:

    def __init__(self, mapping=[]):
        self.mapping = mapping
        
    def total_distance(self):
    	frets = [x[2] for x in self.mapping]
    	no_open = [x for x in frets if x != 0]
    	return sum(abs(np.diff(no_open)))
        



    