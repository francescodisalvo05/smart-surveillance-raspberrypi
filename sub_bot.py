from MQTT.DoSomething import DoSomething


import time
import json
import datetime

from picamera import PiCamera


class Subscriber(DoSomething):

    def notify(self, topic, msg):
        input_json = json.loads(msg)
        
        timestamp = input_json['timestamp']
        # use them?
        label = input_json['class']
        confidence = input_json['confidence']

        # send message here
        camera = PiCamera()
        camera.start_preview()
        
        time.sleep(2) # focus
        img_path = './assets/storage/{}.jpg'.format('test') # use a clear version of timestamp
        
        camera.capture(img_path)
        camera.stop_preview()

        # start streaming? 

        self.bot.send_alarm(timestamp,img_path)


        

if __name__ == "__main__":
    test = Subscriber("Bot")
    test.run()
    test.myMqttClient.mySubscribe("/R0001/alerts")

    while True:
        time.sleep(1)