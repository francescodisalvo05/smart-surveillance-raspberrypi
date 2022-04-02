from MQTT.DoSomething import DoSomething


import time
import json
import datetime

from picamera import PiCamera
from bot.botds import Bot


class Subscriber(DoSomething):

    def notify(self, topic, msg):

        input_json = json.loads(msg)
        
        timestamp = input_json['timestamp']
        label = input_json['class']
        confidence = input_json['confidence']

        img_path = './assets/storage/{}.jpg'.format('test')

        camera = PiCamera()
        camera.start_preview()    
        camera.capture(img_path)
        camera.stop_preview()
        camera.close()

        # start streaming? 

        self.bot.send_alarm(timestamp,img_path)


        

if __name__ == "__main__":
    test = Subscriber("Bot")
    test.run()
    test.myMqttClient.mySubscribe("/R0001/alerts")

    while True:
        time.sleep(1)