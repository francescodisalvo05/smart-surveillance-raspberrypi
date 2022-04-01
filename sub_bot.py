from DoSomething import DoSomething
from bot.botds import Bot

import time
import json
import datetime

from picamera import PiCamera


class Subscriber(DoSomething):
    def __init__(self):
        self.bot = Bot(True)
        self.bot.run()

    def notify(self, topic, msg):
        input_json = json.loads(msg)
        
        timestamp = input_json['timestamp']
        label = input_json['class']
        confidence = input_json['confidence']

        # send message here
        # self.bot.send_message(timestamp,label,confidence)

        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        camera = PiCamera()
        camera.start_preview()
        
        time.sleep(2) # focus
        img_path = './assets/storage/{}.jpg'.format(now)
        
        camera.capture(img_path)
        camera.stop_preview()

        # start the streaming

        # self.bot.send_img(img_path)


        

if __name__ == "__main__":
    test = Subscriber("Bot")
    test.run()
    test.myMqttClient.mySubscribe("/R0001/alerts")

    while True:
        time.sleep(1)