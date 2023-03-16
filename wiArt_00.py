
#!/usr/bin/python

"""
sense.py
"""

import sys
from scientisst import *
from scientisst import __version__
from threading import Timer
from threading import Event
from sense_src.arg_parser import ArgParser
from sense_src.custom_script import get_custom_script, CustomScript
from sense_src.device_picker import DevicePicker
from sense_src.file_writer import *
from matplotlib import pyplot as plt
import time
import random
import numpy as np
from threading import Thread
import os


fs=1000     #   Sensor sampling frequency (default is 1000)
gravityVector   =   [ 0 , 0 , 0 ]       #   Gravitational acceleration vector to be subtracted upon calibration
lastXhat        =   [ 1000 , 1000 , 1000 ]      #   Continuity values for filtering and calculations
lastXhatminus   =   [ 0 , 0 , 0 ]
lastP           =   [ 1000 , 1000 , 1000 ]
lastPminus      =   [ 0 , 0 , 0 ]
lastK           =   [ 0 , 0 , 0 ]
lastFiltered    =   [ 0 , 0 , 0 ]
avDim           =   50      #   Amount of samples used for averaging
storedFiltered  =   [np.zeros(avDim)]+[np.zeros(avDim)]+[np.zeros(avDim)]       #   Acceleration (x,y,z) values stored for averaging
weights         =   np.linspace(1,10, num=avDim)    #   Averaging weights for each element stored
weights         =   weights/np.linalg.norm(weights)     #   Normalize to avoid scaling
threshold       =   70      #   Instantaneous velocity threshold for stroke dash detection
time_window     =   100     #   Amount of samples used to derive intantaneous velocity
counter         =   0       #   Sample counter
dashSize        =   0       #   Relative size of each stroke
Pos             =   False



detectGravity   =   Event()
calibrationFlag =   False
calibrationSequence =   [ [] , [] , [] ]
accThreshold    =   0.5
raisedX         =   False




def run_scheduled_task(duration, stop_event):
    def stop(stop_event):
        stop_event.set()

    timer = Timer(duration, stop, [stop_event])
    timer.start()
    return timer


# refreshes the plot in order to have a seemingly continuous flux of data

def live_plotter(x_vec,y1_data,y2_data,y3_data,line1, line2, line3,identifier='Acceleration',pause_time=0.005):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-,',alpha=0.8) 
        #line2, = ax.plot(x_vec,y2_data,'-,',alpha=0.8) 
        #line3, = ax.plot(x_vec,y3_data,'-,',alpha=0.8)        
        #update plot label/title
        plt.ylabel('mV')
        plt.title('Title: {}'.format(identifier))
        plt.show()
    #print(y1_data[int(len(y1_data)/2)])
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_data(x_vec, y1_data)
    #line2.set_data(x_vec, y2_data)
    #line3.set_data(x_vec, y3_data)

    
    
    
    plt.ylim(-3,3)
    if np.min(x_vec)<=line1.axes.get_xlim()[0] or np.max(x_vec)>=line1.axes.get_xlim()[1]:
        plt.xlim([np.min(x_vec),np.max(x_vec)])
    
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1, line2, line3

def calibrate(sample):
    global calibrationSequence, gravityVector, calibrationFlag
    if not detectGravity.is_set():
        for i in range(3):
            calibrationSequence[i] += [sample[i]]
    elif not calibrationFlag:
        calibrationFlag =   True
        gravityVector   =   [ np.average( calibrationSequence[0][ int( 5 * fs / 3 ):] )] + [np.average( calibrationSequence[1][int(5*fs/3):] )] + [np.average( calibrationSequence[2][int(5*fs/3):] )]
        sys.stdout.write("\nCALIBRATION FINISHED\n")
    else:
        return


def kalman(sample, Q, R):
    
    
    correction_cooldown= Event()
    correction_cooldown.set()
    global lastXhat, lastXhatminus, lastP, lastPminus, lastK, gravityVector, calibrationFlag, calibrationSequence, raisedX, accThreshold
    
##  Data            X   Y   Z
    xhat       =  [ 0 , 0 , 0 ]
    xhatminus  =  [ 0 , 0 , 0 ]
    P          =  [ 0 , 0 , 0 ]
    Pminus     =  [ 0 , 0 , 0 ]
    K          =  [ 0 , 0 , 0 ]
        
    # time update
        
    Pminus      =   [z + Q for z in lastP]
    xhatminus   =   lastXhat
    K           =   [z / ( z + R ) for z in Pminus]
    for i in range( 3 ):
        xhat[i]        =   ( xhatminus[i] + K[i] * ( sample[i] - xhatminus[i] ) )
        P[i] =   ( 1 - K[i] ) * Pminus[i]


    lastXhat      =  xhat
    lastXhatminus =  xhatminus
    lastP         =  P
    lastPminus    =  Pminus
    lastK         =  K

    return xhat


def extract(array, Q, R, write):

    global lastFiltered, storedFiltered, weights, fs, calibrationSequence, threshold, time_window, counter, Pos, dashSize
    scale       =   1/fs
    filtered    =   [ [] , [] , [] ]
    integral    =   [ 0 , 0 , 0 ]

    for k in range(len(array[0])):
        counter+=1
        sample  =   [array[0][k]]+[array[1][k]]+[array[2][k]]
        calibrate(sample)
        for l in range(3):
            sample[l]-=gravityVector[l]
        for i in range(len(filtered)):
            storedFiltered[i]=np.append(storedFiltered[i][1:],[kalman(sample, Q, R)[i]])
            
            filtered[i] +=  [np.dot(storedFiltered[i], weights)]
            if calibrationFlag and counter > time_window and i==0:
                if filtered[i][k]  >   0.5:
                    Pos= True
                    integral[i] = np.sum(filtered[i][-1*time_window:])
                    
                    if dashSize< integral[i]//threshold:
                        
                        dashSize = integral[i]//threshold
            
                elif Pos:
                    os.write(write, str.encode(dashSize))
                    dashSize=0
                    Pos= False
                    
                    
                    
                    
        
    for i in range(3):
        
        lastFiltered[i] = filtered[i][-1]

    return filtered
    


def move(read, write):
    dash_size = os.read(read).decode()
    sys.stdout.write(dash_size)




def main():
    arg_parser = ArgParser()
    args = arg_parser.args
    global calibration_finished
    if args.version:
        sys.stdout.write("sense.py version {}\n".format(__version__))
        sys.exit(0)

    if args.address:
        address = args.address
    else:
        if args.mode == COM_MODE_BT:
            address = DevicePicker().select_device()
            if not address:
                arg_parser.error("No paired device found")
        else:
            arg_parser.error("No address provided")

    args.channels = sorted(map(int, args.channels.split(",")))

    scientisst = ScientISST(address, com_mode=args.mode, log=args.log)

    try:
        if args.output:
            firmware_version = scientisst.version_and_adc_chars(print=False)
            file_writer = FileWriter(
                args.output,
                address,
                args.fs,
                args.channels,
                args.convert,
                __version__,
                firmware_version,
            )
        if args.stream:
            from sense_src.stream_lsl import StreamLSL

            lsl = StreamLSL(
                args.channels,
                args.fs,
                address,
            )
        if args.script:
            script = get_custom_script(args.script)

        stop_event = Event()

        scientisst.start(args.fs, args.channels)
        sys.stdout.write("Start acquisition\n")

        if args.output:
            file_writer.start()
        if args.stream:
            lsl.start()
        if args.script:
            script.start()

        timer = None
        if args.duration > 0:
            timer = run_scheduled_task(args.duration, stop_event)
        try:
            if args.verbose:
                header = "\t".join(get_header(args.channels, args.convert)) + "\n"
                #sys.stdout.write(header)
            
            

            sizeWindow = 3000 #samples
            
            line_x=[]
            line_y=[]
            line_z=[]

            X_plotData=np.zeros(sizeWindow)
            
            Y_plotData=np.zeros(sizeWindow)

            Z_plotData=np.zeros(sizeWindow)

            X_max=2130
            X_min=1290
            Y_max=2127
            Y_min=1296
            Z_max=2250
            Z_min=1420

            axis= range(sizeWindow+1)[1:]

            run_scheduled_task(5, detectGravity)
            pipe_read,pipe_write = os.pipe()
            mouseMover = Thread(move, (pipe_read, pipe_write))
            mouseMover.start()
            n=-1
            while not stop_event.is_set() :
                rawX=[]
                rawY=[]
                rawZ=[]
                frames = scientisst.read(convert=args.convert, matrix=True)
                if args.output:
                    file_writer.put(frames)
                if args.stream:
                    lsl.put(frames)
                if args.script:
                    script.put(frames)
                #if args.verbose:
                for frame in frames:
                    n+=1
                    if n >= len(axis):
                        axis=np.append(axis[1:],[axis[-1]+1])
                    
                    x=frame[5]
                    y=frame[6]
                    z=frame[7]
                    
                    rawX= np.append(rawX, [(x-X_min)/(X_max-X_min)*2-1])
                    rawY= np.append(rawY, [(y-Y_min)/(Y_max-Y_min)*2-1])
                    rawZ= np.append(rawZ, [(z-Z_min)/(Z_max-Z_min)*2-1])
                
                
                output  =   extract([rawX]+[rawY]+[rawZ], 1e-4, 0.9, pipe_write)
                X_plotData  =   np.append(X_plotData[len(output[0]):], output[0])
                Y_plotData  =   np.append(Y_plotData[len(output[1]):], output[1])
                Z_plotData  =   np.append(Z_plotData[len(output[2]):], output[2])
                line_x,line_y, line_z=tuple(live_plotter(axis,X_plotData,Y_plotData,Z_plotData,line_x, line_y, line_z))
                

                    
        except KeyboardInterrupt:
            if args.duration and timer:
                timer.cancel()
            pass

        scientisst.stop()
        # let the acquisition stop before stoping other threads
        time.sleep(0.25)

        sys.stdout.write("Stop acquisition\n")
        if args.output:
            file_writer.stop()
        if args.stream:
            lsl.stop()
        if args.script:
            script.stop()

    finally:
        scientisst.disconnect()
    sys.exit(0)


if __name__ == "__main__":
    main()
