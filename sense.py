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


def run_scheduled_task(duration, stop_event):
    def stop(stop_event):
        stop_event.set()

    timer = Timer(duration, stop, [stop_event])
    timer.start()
    return timer


# refreshes the plot in order to have a seemingly continuous flux of data

def live_plotter(x_vec,y1_data,line1,identifier='Acceleration',pause_time=0.005):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-,',alpha=0.8)        
        #update plot label/title
        plt.ylabel('mV')
        plt.title('Title: {}'.format(identifier))
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_data(x_vec, y1_data)

    
    
    
    plt.ylim(1000,3000)
    if np.min(x_vec)<=line1.axes.get_xlim()[0] or np.max(x_vec)>=line1.axes.get_xlim()[1]:
        plt.xlim([np.min(x_vec),np.max(x_vec)])
    
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1

calibration_finished = False

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
                sys.stdout.write(header)
            
            sizeWindow = 3000 #samples
            movingAverage_dim= 50 #samples
            line_x=[]
            X_plotData=np.zeros(sizeWindow)
            last_average_Xdata= np.zeros(movingAverage_dim)
            #Y_plotData=np.zeros(sizeWindow)
            #Z_plotData=np.zeros(sizeWindow)
            C_max=0
            C_min=5000
            calibration_time=10
            stop_calibration=Event()
            calibration = run_scheduled_task(calibration_time, stop_calibration)
            axis= range(sizeWindow+1)[1:]

            counter=-1
            while not stop_event.is_set() :
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
                    last_average_Xdata = np.append(last_average_Xdata[1:], [x])

                    #Se a calibração estiver a decorrer, queremos ir ajustando os C_max e C_min, com base nos valores raw
                    if not stop_calibration.is_set():
                        if C_max < x:
                            C_max=x
                        if C_min>x:
                            C_min=x
                        if counter >= len(last_average_Xdata):
                            X_plotData=np.append(X_plotData, [np.average(last_average_Xdata)])
                        else:
                            X_plotData= np.append(X_plotData, [x])
                    else:
                        X_plotData=np.append(X_plotData, [np.average(last_average_Xdata)])
                        if not calibration_finished:
                            print("Calibration has finished:\n Max= ", C_max , "\n min= ", C_min)
                            calibration_finished= True
                    X_plotData=X_plotData[1:]
                line_x=live_plotter(axis,X_plotData,line_x)
        



                    
                    
                        
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
