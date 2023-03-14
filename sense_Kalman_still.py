##CORRIGIR PLOT OUTRA VEZ PARA INCLUIR O Z, DESCOMENTAR LA EM BAIXO
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


def run_scheduled_task(duration, stop_event):
    def stop(stop_event):
        stop_event.set()

    timer = Timer(duration, stop, [stop_event])
    timer.start()
    return timer


# refreshes the plot in order to have a seemingly continuous flux of data

def live_plotter(x_vec,y1_data,y2_data,y3_data,line1, line2 ,line3,identifier='Acceleration',pause_time=0.005):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-,',alpha=0.8) 
        line2, = ax.plot(x_vec,y2_data,'-,',alpha=0.8) 
        line3, = ax.plot(x_vec,y3_data,'-,',alpha=0.8)        
        #update plot label/title
        plt.ylabel('mV')
        plt.title('Title: {}'.format(identifier))
        plt.show()
    #print(y1_data[int(len(y1_data)/2)])
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_data(x_vec, y1_data)
    line2.set_data(x_vec, y2_data)
    line3.set_data(x_vec, y3_data)

    
    
    
    plt.ylim(-3,3)
    if np.min(x_vec)<=line1.axes.get_xlim()[0] or np.max(x_vec)>=line1.axes.get_xlim()[1]:
        plt.xlim([np.min(x_vec),np.max(x_vec)])
    
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1, line2, line3

def Kalman(array, Q, R):
    fs=1000
    
    correction_cooldown= Event()
    correction_cooldown.set()
    global lastXhat, lastXhatminus, lastP, lastPminus, lastK, gravityVector, lastVelocity, lastOutput, lastDxhat, calibrationFlag, calibrationSequence, averageVec, averageDim
    
##  Data             X     Y     Z
    xhat       =  [ [0] , [0] , [0] ]
    xhatminus  =  [ [0] , [0] , [0] ]
    P          =  [ [0] , [0] , [0] ]
    Pminus     =  [ [0] , [0] , [0] ]
    K          =  [ [0] , [0] , [0] ]
    output     =  [ [0] , [0] , [0] ]
    dxhat      =  [ [0] , [0] , [0] ]
    velocity   =  [ [0] , [0] , [0] ]
## Iterate through X , Y and Z to set the array dimension
    for i in range(3):
        velocity[i]   =  np.append( [lastVelocity[i]] , np.zeros( len( array[0] ) - 1 ) )
        xhat[i]       =  np.append( [lastXhat[i]] , np.zeros( len( array[0] ) - 1 ) )
        xhatminus[i]  =  np.append( [lastXhatminus[i]] , np.zeros( len( array[0] ) - 1 ) )
        P[i]          =  np.append( [lastP[i]] , np.zeros( len( array[0] ) - 1 ) )
        Pminus[i]     =  np.append( [lastPminus[i]] , np.zeros( len( array[0] ) - 1 ) )
        K[i]          =  np.append( [lastK[i]] , np.zeros( len( array[0] ) - 1 ) ) 
        output[i]     =  np.append([lastOutput[i]],np.zeros(len(array[0])-1))
        dxhat[i]      =  np.append( [lastDxhat[i]] , np.zeros( len( array[0] ) - 1 ) )

    dxhat_copy  =  dxhat

## Iterate through sample index
    for k in range( 1 , len( array[0] ) ):
        
        for i in range( 3 ):

            # time update
            xhatminus[i][k] =   xhat[i][k-1]
            Pminus[i][k]    =   P[i][k-1] + Q

            # measurement update
            K[i][k] =   Pminus[i][k] / ( Pminus[i][k] + R )
            xhat[i][k]      =   ( xhatminus[i][k] + K[i][k] * ( array[i][k] - xhatminus[i][k] ) )
            
            P[i][k] =   ( 1 - K[i][k] ) * Pminus[i][k]
            
        for i in range( 3 ):


            if detectGravity.is_set():

                if not calibrationFlag:

                    gravityVector   =   [ np.average( calibrationSequence[0][ int( 5 * fs / 3 ):] )] + [np.average( calibrationSequence[1][int(5*fs/3):] )] + [np.average( calibrationSequence[2][int(5*fs/3):] )]
                    print(gravityVector)
                    print( np.linalg.norm( gravityVector ) )

                calibrationFlag     =   True
                output[i][k]        =   np.round( xhat[i][k] - gravityVector[i] , 5 )
                velocity[i][k]      =   output[i][k] / fs + velocity[i][k-1]

            else:

                calibrationSequence[i]  +=  [xhat[i][k]]
                output[i][k]        =   np.round( xhat[i][k] , 5 )


            #ISTO É A DERIVADA
            dxhat[i][k] =   ( output[i][k] - output[i][k-1] ) * fs / 10

            #ISTO É A MÉDIA DA DERIVADA
            if i == 0:

                averageVec  =   np.append( averageVec[1:] , dxhat[i][k] )
                dxhat_copy[i][k]    =   np.average( averageVec )
            
            
            
        


    for i in range(3):
        lastXhat[i]      =  xhat[i][-1]
        lastXhatminus[i] =  xhatminus[i][-1]
        lastP[i]         =  P[i][-1]
        lastPminus[i]    =  Pminus[i][-1]
        lastK[i]         =  K[i][-1]
        lastOutput[i]    =  output[i][-1]
        lastDxhat[i]      =  dxhat[i][-1]
        lastVelocity[i]  =  velocity[i][-1]
    return (output,dxhat_copy)
    #return output
    #return velocity
    

gravityVector   =  [ 0 , 0 , 0 ]
lastDxhat        =  [ [0] , [0] , [0] ]
lastXhat        =  [ [1000] , [1000] , [1000] ]
lastXhatminus   =  [ [0] , [0] , [0] ]
lastP           =  [ [1000] , [1000] , [1000] ]
lastPminus      =  [ [0] , [0] , [0] ]
lastK           =  [ [0] , [0] , [0] ]
lastVelocity    =  [ 0 , 0 , 0 ]
lastOutput      =  [ 0 , 0 , 0 ]
detectGravity   =  Event()
averageDim      =  100
averageVec      =  np.zeros(averageDim)
calibrationFlag =  False
calibrationSequence=[[],[],[]]

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

            counter=-1
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
                    counter+=1
                    if counter >= len(axis):
                        axis=np.append(axis[1:],[axis[-1]+1])
                    
                    x=frame[5]
                    y=frame[6]
                    z=frame[7]
                    
                    rawX= np.append(rawX, [(x-X_min)/(X_max-X_min)*2-1])
                    rawY= np.append(rawY, [(y-Y_min)/(Y_max-Y_min)*2-1])
                    rawZ= np.append(rawZ, [(z-Z_min)/(Z_max-Z_min)*2-1])
                
                acc, derivada= Kalman([rawX]+[rawY]+[rawZ], 1e-4, 0.3)
                X_plotData=np.append(X_plotData[len(acc[0]):], acc[0])
                Y_plotData=np.append(Y_plotData[len(acc[1]):], acc[1])
                Z_plotData=np.append(Z_plotData[len(acc[2]):], acc[2])
                line_x,line_y, line_z=tuple(live_plotter(axis,X_plotData,Y_plotData,Z_plotData,line_x, line_y, line_z))
                

                    #sys.stdout.write("{}\n".format(frames))
                    #print("Nseq: ",frame[0],"\tx: ", x,"\ty: ", y,"\tz: ", z)
                    ####print("Nseq: ",frame[0],"\tx: ", x)
                    #plt.scatter(counter,y)
                    #rand_val = np.random.randn(1)
                    #y_vec[-1] = rand_val
                    
                    #y_vec = np.append(y_vec[1:],0.0)
                """plt.plot(axis,xx)
                    plt.plot(axis,yy)
                    plt.plot(axis,zz)
                    if counter>=5000:
                        plt.axis([laststop, counter, 500 , 2500])
                        counter=0
                        plt.show()"""
                    #sys.stdout.write("{}\n".format(frames))
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
        """plt.plot(xx, label= "X")
        plt.plot(yy, label= "Y")
        plt.plot(zz, label= "Z")
        plt.legend()
        plt.show()"""
    sys.exit(0)


if __name__ == "__main__":
    main()
