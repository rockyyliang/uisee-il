import os
import sys
import numpy as np
import usimpy
import matplotlib.pyplot as plt
from lstm import Baseline2D_LSTM
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import torchvision
from collections import deque
import time
from ctypes import c_uint64
from timeloop import Timeloop
from datetime import timedelta

from wheel import JSManager
from steering_ratio import variable_steer

## DAGGER STORE PATH
SAVE_PATH = './dataset/'
IMG_PATH = SAVE_PATH + 'imgs/'


## Herper function to get data
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
    if states.forward_speed>0.1 and bool(record_flag.value):
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

'''
periodic_job is used to save data at a fixed freq
main loop runs at 100hz while this happens at 10hz
initialize tl just before main loop is called
'''
# connection
id = usimpy.UsimCreateApiConnection("127.0.0.1", 17771, 5000, 10000)

# start simulation
ret = usimpy.UsimStartSim(id, 10000)
print (ret)

## action
states = usimpy.UsimVehicleState()
## collision
collision = usimpy.UsimCollision()
## image
image = usimpy.UsimCameraResponse()

#control
control = usimpy.UsimSpeedAdaptMode()
control.handbrake_on = False
control.expected_speed = 50  # m/s
control.steering_angle = 0.0

#ctype because I need a mutable int for incrementing save count
count = c_uint64(dp_count)
tl = Timeloop()
@tl.job(interval=timedelta(seconds=0.10))
def periodic_job(get_image=get_image, get_states=get_states, save_data=save_data, id=id, image=image, states=states, count=count):
    states = get_states(id, states)
    img = get_image(id, image)
    save_data(id, img, states, control, count)

js = JSManager()
self_driving = c_uint64(1)
manual_driving = c_uint64(0)
gear_level = c_uint64(1)

record_flag = c_uint64(0)
@tl.job(interval=timedelta(seconds=0.01))
def wheel_job(js=js, flag=manual_driving):
    '''read steering wheel at 100hz'''
    #read from steering wheel
    axes = js.update_axes()
    buttons = js.update_buttons()
    for b_ind, button in enumerate(buttons):
        #loop thru all buttons
        if b_ind==1:
            #start recording
            if button==1:
                record_flag.value = 1
        if b_ind==2:
            #manual if button 2 is pressed
            if button==1:           # speed 20, with manual steer
                manual_driving.value = 1 #self_driving.value = 0
                gear_level.value = 0
                #print('manual mode')
            else:
                manual_driving.value = 0 #self_driving.value = 1
                gear_level.value = 1
        if b_ind==3:
            #triangle
            if button == 1:
                manual_driving.value = 1 # speed 40, with manual steer
                gear_level.value = 1
            else:
                manual_driving.value = 0
        if b_ind==23:
            #exit game loop
            if button==1:
                exit()
    speed_request_delta = 0
    for a_ind, axis in enumerate(axes):
        #loop thru all axes
        if a_ind==0:
            #handle wheel things
            steer_bounds = (-30,30)
            steer_request = max(min(-axis*100, steer_bounds[1]),steer_bounds[0])
            multiplier = variable_steer(states.forward_speed)
            steer_request = steer_request/multiplier
            #print('mul: {:3f} request: {:3f}'.format(multiplier, steer_request))
        if a_ind==2:
            #gas
            speed_request_delta = js.axis2percent(axis)*200
        if a_ind==3:
            #brake
            speed_request_delta -= js.axis2percent(axis)*200

    #send control to car
    if bool(manual_driving.value):# or not bool(gear_level.value):
        control.steering_angle = steer_request
        if bool(gear_level.value):
            control.expected_speed = 20
    elif not bool(gear_level.value):
        control.expected_speed = 50
        control.steering_angle = steer_request



### Load the model
model = Baseline2D_LSTM()
model.load_state_dict(torch.load('D:\\uisee\\models\\05-17-09\\epoch_4.pth.tar')['state_dict'])
model.cuda()
model.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor()
])

q = deque(maxlen=10)
v = deque(maxlen=10)

def main_loop():
    try:
        start = time.time()
        index = 0
        dt = 0.1




        while(1):



            # get vehicle post & action
            ret = usimpy.UsimGetVehicleState(id, states)
            # get image observation
            ret = usimpy.UsimGetOneCameraResponse(id, 0, image)
            # control vehicle via speed & steer
            ret = usimpy.UsimSetVehicleControlsBySA(id, control)
            # save image
            img = np.array(image.image[0:480*320*3])
            img = img.reshape((320, 480, 3))
            frame = np.uint8(img)[:, :, ::-1].copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            im = transforms(im)
            q.append(im)
            v.append(torch.tensor(states.forward_speed))


            if len(q) == 10:
                input = torch.stack(list(q), dim=0)
                input = input.unsqueeze(dim=0).cuda()
                input_speed = torch.stack(list(v), dim=0)
                input_speed = input_speed.unsqueeze(dim=0).cuda()
                with torch.no_grad():
                    outputs_a, outputs_s = model(input, input_speed)
                    outputs_a = outputs_a.cpu().item()
                    outputs_s = outputs_s.cpu().item()
                # outputs_s = outputs_s.cpu().item()
                # speed(mean, std): 129.64541911882017 100.99125912605909
                # steer(mean, std): 0.7267446733702028 3.7620375242243145
                    angle = outputs_a
                    # print(outputs_s)
                    #speed = outputs_s * (266.013+168.097) - 168.097
                #print("steer: {:5f}".format(angle))
                # control.expected_speed = speed  # m/s

                if int(manual_driving.value) == 0 and bool(gear_level.value): # network output

                    control.steering_angle = angle
                    if outputs_s > 0.5:
                        speed_request = 50
                    else:
                        speed_request = 20
                    #print('{:4f} {:4f}'.format(angle,speed_request))
                    control.expected_speed = speed_request
                    #print("network")
                #print(manual_driving.value, gear_level.value)





            time.sleep(dt - (time.time() - start) % dt)

    except:
        print('error occured in main loop:', sys.exc_info())
    finally:
        ret = usimpy.UsimStopSim(id)
        print('exit loop')


if __name__=="__main__":
    try:
        tl.start(block=False)
        main_loop()
        #print('main')
    except:
        print('error occured:', sys.exc_info())
    finally:
        #exit timeloop
        while True:
            try:
                time.sleep(1)
            except:
                tl.stop()
                break
        print('exit main')
        sys.exit()
