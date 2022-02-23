import pyaudio
import json
import io
import wave

import tensorflow as tf
import numpy as np

from datetime import datetime
from scipy.io import wavfile
from argparse import ArgumentParser

import logging
logging.getLogger().setLevel(logging.INFO)

from MQTT.DoSomething import DoSomething


def main(args):

    p = pyaudio.PyAudio()

    publisher = DoSomething("Publisher")
    publisher.run()

    while True:
        # record 4s (temporary)
        # to do: audio preprocessing
        tf_audio = record_audio(args, p)   
        
        # to do: convert number to label
        # prediction, probability = make_inference(tf_audio, args.tflite_path)

        # publish via MQTT
        # publish_outcome(publisher, prediction, probability, args.room)
        
        

def record_audio(args, p):

    logging.info('Start recoding...')

    stream = p.open(format=pyaudio.paInt16, channels=args.channels, rate=args.rate, input=True, frames_per_buffer=args.chunk)

    frames = []
    for _ in range(0,int(args.rate / args.chunk * args.seconds)):
        data = stream.read(args.chunk)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()

    wf = wave.open('filename.wav', 'wb')
    wf.setnchannels(args.channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(args.rate)
    wf.writeframes(b''.join(frames))    
    wf.close() 
    
    audio = tf.io.read_file('filename.wav')
    tf_audio, _ = tf.audio.decode_wav(audio) 

    logging.info('End recoding...')  


    return tf_audio


def preprocess_audio(audio):
    return audio


def make_inference(self, tf_audio, tflite_path):

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # give the input
    interpreter.set_tensor(input_details[0]["index"], tf_audio)
    interpreter.invoke()

    # predict and get the current ground truth
    curr_prediction_logits = interpreter.get_tensor(output_details[0]['index']).squeeze()
    curr_prediction = np.argmax(curr_prediction_logits)
    
    return curr_prediction, np.max(curr_prediction_logits)


def publish_outcome(publisher, prediction, probability, room):
    
    timestamp = int(datetime.now().timestamp())

    body = {
        'timestamp': timestamp,
        'class': prediction, 
        'confidence': probability
    }

    publisher.myMqttClient.myPublish("/{}/alerts".format(room), json.dumps(body))
    



if __name__ == '__main__':
    
    parser = ArgumentParser()
    
    parser.add_argument('--chunk', type=int, default=1024, help='Set number of chunks')
    parser.add_argument('--format', type=str, default='Int16', help='Set the format of the audio track [Int8,Int16,Int32]')
    parser.add_argument('--channels', type=int, default=2, help='Set the number of channels')
    parser.add_argument('--seconds', type=int, default=10, help='Set the length of the recording (seconds)')
    parser.add_argument('--rate', type=int, default=44100, help='Set the rate')
    parser.add_argument('--name', type=str, default=None, help='Set the name of the audio track')

    parser.add_argument('--room', type=int, default=1024, help='Room where the device is located. Useful with a set of different devices')
    parser.add_argument('--tflite_path', type=int, default=None, help='tflite_path')
    
    args = parser.parse_args()

    main(args)

    