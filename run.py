import pyaudio
import wave

import os

import time
from datetime import datetime
import os
from argparse import ArgumentParser


def main():

    p = pyaudio.PyAudio()
    
    while True:
        audio = record_audio(args, p)

        

     


def record_audio(args, p):

    stream = p.open(format=format, channels=args.channels, rate=args.rate, input=True, frames_per_buffer=args.chunk)

    frames = []
    for _ in range(0,int(args.rate / args.chunk * args.seconds)):
        data = stream.read(args.chunk)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()




if __name__ == '__main__':
    
    parser = ArgumentParser()
    
    parser.add_argument('--chunk', type=int, default=1024, help='Set number of chunks')
    parser.add_argument('--format', type=str, default='Int16', help='Set the format of the audio track [Int8,Int16,Int32]')
    parser.add_argument('--channels', type=int, default=2, help='Set the number of channels')
    parser.add_argument('--seconds', type=int, default=4, help='Set the length of the recording (seconds)')
    parser.add_argument('--rate', type=int, default=44100, help='Set the rate')
    parser.add_argument('--name', type=str, default=None, help='Set the name of the audio track')
    
    args = parser.parse_args()

    main(args)

    