import pyaudio
import json
import io
import wave

import tensorflow as tf
import numpy as np

from datetime import datetime
from scipy.io import wavfile
from array import array
from argparse import ArgumentParser
from io import BytesIO

import logging
logging.getLogger().setLevel(logging.INFO)

from MQTT.DoSomething import DoSomething


def main(args):

    p = pyaudio.PyAudio()

    print("\n\n")

    publisher = DoSomething("Publisher")
    publisher.run()

    logging.info("The mic is running...")

    while True:

        stream = p.open(format=pyaudio.paInt16, channels=1, rate=args.rate, input=True, frames_per_buffer=args.chunk)

        # wait for a trigger
        while(True):
            temp_data = stream.read(args.chunk)
            temp_chunk = array('h',temp_data)
            volume = max(temp_chunk)
        
            if volume >= 500:
                break

        # record the audio file & stop stream
        tf_audio = record_audio(args, p, stream)
        

        # get mfccs
        # tf_mfccs = get_mfccs(tf_audio)

        # print(tf.shape(tf_mfccs))

        break

        


        
        # to do: convert number to label
        # prediction, probability = make_inference(tf_audio, args.tflite_path)

        # publish via MQTT
        # publish_outcome(publisher, prediction, probability, args.room)
        
        

def record_audio(args, p, stream):

    logging.info('Start recoding...')

    chunks = int((args.rate / args.chunk) * args.seconds)

    frames = []

    stream.start_stream()
    for _ in range(chunks):
        data = stream.read(args.chunk)
        frames.append(data)
    stream.stop_stream()

    buffer = BytesIO()
    buffer.seek(0)

    wf = wave.open(buffer, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(args.rate)
    wf.writeframes(b''.join(frames))    
    wf.close() 
    buffer.seek(0)
    
    tf_audio, _ = tf.audio.decode_wav(buffer.read()) 
    tf_audio = tf.squeeze(tf_audio, 1)

    logging.info('End recoding...')  


    return tf_audio


def make_inference(tf_audio, tflite_path):

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
    
    parser.add_argument('--chunk', type=int, default=4410, help='Set number of chunks')
    parser.add_argument('--seconds', type=int, default=1, help='Set the length of the recording (seconds)')
    parser.add_argument('--rate', type=int, default=44100, help='Set the rate')

    parser.add_argument('--room', type=str, default='Entrance', help='Room where the device is located. Useful with a set of different devices')
    parser.add_argument('--tflite_path', type=str, default='models_tflite/model_test_tflite/model.tflite', help='tflite_path')
    
    args = parser.parse_args()

    main(args)

    