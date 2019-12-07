'''
change steering ratio based on speed.
used to make sim easier to drive
'''
import matplotlib.pyplot as plt
import numpy as np
import math

BASE = 8
MAX_MUL = 2.5

def variable_steer(vx):
    '''given speed in m/s, returns a steering ratio multiplier '''
    threshold = 10
    if vx<=threshold:
        #constant for low speed
        return 1
    #by this point, process speeds higher than threashold
    x = math.log(vx-(threshold-7),BASE)
    x = min(x, MAX_MUL)
    return x

if __name__=='__main__':
    speeds = np.arange(1,80,2)
    ratio_multipliers = []
    for s in speeds:
        ratio_multipliers.append(variable_steer(s))
    plt.plot(speeds,ratio_multipliers)
    plt.grid()
    plt.show()
