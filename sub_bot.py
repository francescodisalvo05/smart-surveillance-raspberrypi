from MQTT.DoSomething import DoSomething

import os
import time
import json
import datetime
import time

from picamera import PiCamera
from bot.botds import Bot
from bot.botmessage import Bot

import subprocess


class Subscriber(DoSomething):

    def notify(self, topic, msg):

        input_json = json.loads(msg)

        timestamp = input_json['timestamp']
        label = input_json['class']

        if label == 'Bark' or label == 'Doorbell': # not an intrusion
            return 

        # keep a window of 5 minutes, if the bot has already sent an alarm
        # in the last 5 minutes, don't do anythings
        if time.time() - self.last_alarm > 10:  # to do : update to 300!

            self.last_alarm = time.time()

            if label == 'human': # send image with patch
                img_path = input_json['path']
                self.bot.send_alarm(timestamp,'img',label, img_path)


            else: # send video
                """
                time_str = time.strftime("%Y%m%d-%H%M%S")
                video_path = './assets/storage/{}.h264'.format(time_str)
                new_video_path = './assets/storage/{}.mp4'.format(time_str)

                camera = PiCamera()
                camera.resolution = (480, 360)
                camera.rotation = 180
                camera.start_recording(video_path)
                camera.wait_recording(30)
                camera.stop_recording()
                camera.close()

                command = "ffmpeg -i {} -y -vf scale=-1:360 {}".format(video_path, new_video_path)
                subprocess.call([command], shell=True)

                os.remove(video_path)
                """
                self.bot.send_alarm(timestamp,'img',label,'assets/storage/last_image.png')

            



if __name__ == "__main__":
    test = Subscriber("Bot")
    test.run()
    test.myMqttClient.mySubscribe("/R0001/alerts")

    while True:
        time.sleep(1)