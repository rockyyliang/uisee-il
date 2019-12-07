import sys
import time
from timeloop import Timeloop
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pygame

from ctypes import c_uint64

import usimpy

from wheel import JSManager
from steering_ratio import variable_steer

SAVE_PATH = './recorded/'
IMG_PATH = SAVE_PATH + 'imgs/'

'''
make connection to sim and initialize variables.
this is not done in the init part of main_loop because
sim variables and helper functions need to have been
initialized before definition of periodic_job
'''
# connection
id = usimpy.UsimCreateApiConnection("127.0.0.1", 17771, 5000, 10000)
# start simulation
ret = usimpy.UsimStartSim(id, 10000)
print(ret)

# control
control = usimpy.UsimSpeedAdaptMode()
control.expected_speed = 0.0 # m/s
control.steering_angle = 0.0 # angle
control.handbrake_on = False

## action
states = usimpy.UsimVehicleState()
## collision
collision = usimpy.UsimCollision()
## image
image = usimpy.UsimCameraResponse()

'''helper functions to save pic and states'''
def get_image(id, image):
    '''
    gets one image from simulator
    image is usimpy image object, id is sim id returned by connecting api
    returns image as np array
    '''
    ret = usimpy.UsimGetOneCameraResponse(id, 0, image)
    #access image and convert to np array
    img = np.array(image.image[0:480*320*3],dtype=np.uint8)
    img = img.reshape((320, 480, 3))
    return img

def get_states(id, states):
    ret = usimpy.UsimGetVehicleState(id, states)
    return states

def save_data(id, img, states, control, count):
    '''
    saves data
    '''
    if states.forward_speed>0.1:
        print(count.value)
        plt.imsave(IMG_PATH+str(count.value)+'.png', img)
        with open(SAVE_PATH + 'action_pose.txt', 'a+') as f:
            f.write('%d %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' % (int(count.value), states.time_stamp, states.pose.position.x_val, \
                     states.pose.position.y_val, states.pose.position.z_val, states.pose.rotation.x_val, states.pose.rotation.y_val, \
                     states.pose.rotation.z_val, states.steering_angle, states.forward_speed, control.steering_angle, control.expected_speed))

        count.value += 1

#get how many datapoints there already are
try:
    dp_count = len(open(SAVE_PATH+'action_pose.txt').readlines())
except FileNotFoundError:
    dp_count = 0
#ctype because I need a mutable int for incrementing save count
count = c_uint64(dp_count)
'''
periodic_job
 is used to save data at a fixed freq
main loop runs at 100hz while this at 10hz
initialize tl just before main loop is called
'''
tl = Timeloop()
@tl.job(interval=timedelta(seconds=0.10))
def periodic_job(get_image=get_image, get_states=get_states, save_data=save_data, id=id, image=image, states=states, count=count):
    states = get_states(id, states)
    img = get_image(id, image)
    save_data(id, img, states, control, count)


def main_loop():
    try:
        start = time.time()
        index = 0
        dt = 0.01

        '''could try bounding vx request as well'''
        steer_bounds = (-30,30)

        #init joystick manager
        js = JSManager()


        run = True
        while run:
            # get vehicle post & action
            ret = usimpy.UsimGetVehicleState(id, states)

            #read from steering wheel
            axes = js.update_axes()
            buttons = js.update_buttons()
            for b_ind, button in enumerate(buttons):
                #loop thru all buttons
                if b_ind==23:
                    #exit game loop
                    if button==1:
                        run = False

            speed_request_delta = 0
            for a_ind, axis in enumerate(axes):
                #loop thru all axes
                if a_ind==0:
                    #handle wheel things
                    steer_request = max(min(-axis*100, steer_bounds[1]),steer_bounds[0])
                    multiplier = variable_steer(states.forward_speed)
                    steer_request = steer_request/multiplier
                    #print('mul: {:3f} request: {:3f}'.format(multiplier, steer_request))
                    control.steering_angle = steer_request
                if a_ind==2:
                    #gas
                    speed_request_delta = js.axis2percent(axis)*200
                if a_ind==3:
                    #brake
                    speed_request_delta -= js.axis2percent(axis)*200
            #control.expected_speed = states.forward_speed + speed_request_delta
            control.expected_speed = 20
            #('{:2f}'.format(states.steering_angle))

            #send command to car
            ret = usimpy.UsimSetVehicleControlsBySA(id, control)

            #enforce timing
            time.sleep(dt - (time.time()-start)%dt)

    except:
        print('error occured in main loop:', sys.exc_info()[0])
    finally:
        js.finish()
        ret = usimpy.UsimStopSim(id)
        print('exit loop')

if __name__=="__main__":
    try:
        tl.start(block=False)
        main_loop()
        #print('main')
    except:
        print('error occured:', sys.exc_info()[0])
    finally:
        #exit timeloop
        while True:
            try:
                time.sleep(1)
            except:
                tl.stop()
                break
        print('exit main')
        pygame.quit()
        sys.exit()
