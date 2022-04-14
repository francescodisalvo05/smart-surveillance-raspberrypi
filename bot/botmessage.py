from urllib.parse import uses_params
from telegram.ext.callbackcontext import CallbackContext
from datetime import datetime
import logging
import pandas as pd
import time
import requests
import telegram
from telegram.ext import Updater, CommandHandler
import csv
import os.path
import os
import numpy as np
import math
from bot_settings import * 


class Bot:

    def __init__(self,danger) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.TOKEN = TOKEN

        self.logger = logging.getLogger("LOG")
        self.logger.info("Starting BOT.")
        self.updater = Updater(self.TOKEN)
        self.dispatcher = self.updater.dispatcher
        self.danger = danger
        
        
        enable_handler = CommandHandler("enable", self.send_enable)
        self.dispatcher.add_handler(enable_handler)

        enable_handler = CommandHandler("disable", self.send_disable)
        self.dispatcher.add_handler(enable_handler)

        start_handler = CommandHandler("start", self.send_start)
        self.dispatcher.add_handler(start_handler)

        help_handler = CommandHandler("help", self.send_help)
        self.dispatcher.add_handler(help_handler)

        enable_handler = CommandHandler("live_video", self.live_video)
        self.dispatcher.add_handler(enable_handler)
         
        enable_handler = CommandHandler("report", self.report)
        self.dispatcher.add_handler(enable_handler)


        # force_handler = CommandHandler("force", self.force)
        # self.dispatcher.add_handler(force_handler)

        # daily_handler = CommandHandler("daily", self.send_daily)
        # self.dispatcher.add_handler(daily_handler)

    # message to send when the bot is started
    def send_start(self, chatbot, update) -> None:
        welcome_message =  "*Hello, I am the bot that will keep your home safe* \n\n"
        welcome_message += 'Welcome to the notification centre'
        chatbot.message.reply_text(welcome_message)
    
   
    # start the bot
    def send_alarm(self,timestampm,input_type, label, path) -> int:

        reports = []
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        db = self._read_db()
        db.set_index('ids',inplace = True)

        for chat_id in CHAT_IDS:
            if db.at[chat_id,"status"] == True:
                # message = "⚠️ *Intrusion Alert* ⚠️\n" + "🕚 " + current_time
                # self._send_text(message,99706017)

                if input_type == 'video':
                    self._send_video(open(path, 'rb'),chat_id)
                elif input_type == 'img':
                    self._send_img(open(path, 'rb'),label,chat_id)
        
        reports.append(current_time)
        arr = np.genfromtxt('bot/reports.txt',dtype='str')
        reports = np.append(arr,str(current_time))
        np.savetxt('reports.txt', reports, delimiter=" ", fmt="%s")
    


    def _send_text(self,bot_message,chat_id):
        send_text = 'https://api.telegram.org/bot' + TOKEN_MSG + '/sendMessage?chat_id=' + str(chat_id) + '&parse_mode=Markdown&text=' + bot_message
        response = requests.get(send_text)
        return response.json()


    def _send_video(self,file_opened,bot_chatID):
        caption = "⚠️ Intrusion Alert ⚠️\n" + "🕚 " + datetime.now().strftime("%H:%M:%S")

        method = "sendVideo"
        params = {'chat_id': bot_chatID, 'caption': caption}
        files = {'video': file_opened}
        url = 'https://api.telegram.org/bot' + TOKEN_MSG + "/"
        response = requests.post(url + method, params, files=files)
        return response.json()


    def _send_img(self,file_opened,label,bot_chatID):

        if label == 'Human':
            emoticon_label = "📷"
        else:
            emoticon_label = "🔈"

        timestamp = datetime.now().strftime("%H:%M:%S")

        caption = f"⚠️ *Intrusion Alert* ⚠️\n\n🕚 {timestamp} \n{emoticon_label} {label}"
        
        method = "sendPhoto"
        params = {'chat_id': bot_chatID, 'caption':caption, 'parse_mode':'Markdown'}
        files = {'photo': file_opened}
        url = 'https://api.telegram.org/bot' + TOKEN_MSG + "/"
        response = requests.post(url + method, params, files=files)
        return response.json()


    def _read_db(self):      
        df = pd.read_csv("bot/users.csv")
        return df


    def _update_db(self,df,index,column,value):
        df.at[index, column] = value
        df.to_csv("bot/users.csv")