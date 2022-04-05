from MQTT.DoSomething import DoSomething

import os
import time
import json
import datetime

from picamera import PiCamera
from bot.botds import Bot

import subprocess


class Subscriber(DoSomething):

    def notify(self, topic, msg):

        input_json = json.loads(msg)
        
        timestamp = input_json['timestamp']
        label = input_json['class']
        confidence = input_json['confidence']

        time_str = time.strftime("%Y%m%d-%H%M%S")
        video_path = './assets/storage/{}.h264'.format(time_str)
        new_video_path = './assets/storage/{}.mp4'.format(time_str)

        

        """
        img_path = './assets/storage/img.jpeg'
        camera = PiCamera()
        camera.start_preview()    
        camera.capture(img_path)
        camera.stop_preview()
        camera.close()"""

        camera = PiCamera()
        camera.resolution = (480, 360)
        camera.rotation = 180
        camera.start_recording(video_path)
        camera.wait_recording(30)
        camera.stop_recording()
        camera.close()

        command = "ffmpeg -i {} -y -vf scale=-1:360 {}".format(video_path, new_video_path)
        subprocess.call([command], shell=True)

        self.bot.send_alarm(timestamp,new_video_path)

        os.remove(video_path)


if __name__ == "__main__":
    test = Subscriber("Bot")
    test.run()
    test.myMqttClient.mySubscribe("/R0001/alerts")

    while True:
        time.sleep(1)