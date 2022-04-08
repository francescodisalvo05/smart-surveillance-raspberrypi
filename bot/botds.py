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


class Bot:

    def __init__(self,danger) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.TOKEN = '5165021744:AAGqhFc_5heY5EXaBulsRy_HGeC67diZFGs'

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
        welcome_message += '🗞 Type: /enable in order to enable the notifications!\n\n'
        welcome_message += '❌ Type: /disable in order to disable the notifications!\n\n'
        welcome_message += '📓 Type: /report in order to see the full report of alarms!\n\n'
        welcome_message += '🎥 Type: /live_video in order to receive a live video of your house!.\n\n'
        welcome_message += '🆘 Type: /help in order to contact the authors!\n\n'
        chatbot.message.reply_text(welcome_message)
    
   
    def send_enable(self, chatbot, update) -> None:
        # write the chat id in the database
        chat_id = chatbot.message.chat_id
        db = self._read_db()
        db.set_index('ids',inplace = True)

        if chat_id in db.index:
            if db.at[chat_id,'status'] == True:
                enable_message = '✅ The notifications are already enabled'
                chatbot.message.reply_text(enable_message)
            else:
                enable_message = "✅ You'll receive all the notifications"
                self._update_db(db,chat_id,'status',True)
                chatbot.message.reply_text(enable_message)
        else:
            enable_message = "You don't have the permission for this service, contact the authors"
            chatbot.message.reply_text(enable_message)

    def send_disable(self, chatbot, update) -> None:
        # write the chat id in the database
        chat_id = chatbot.message.chat_id
        db = self._read_db()
        db.set_index('ids',inplace = True)

        if chat_id in db.index:
            if db.at[chat_id,'status'] == True:
                enable_message = "❌ You won't receive other notifications"
                chatbot.message.reply_text(enable_message)
                self._update_db(db,chat_id,'status',False)

            else:
                enable_message = "You are not subscribed to the notification service"
                chatbot.message.reply_text(enable_message)

        else:
            enable_message = "You don't have the permission for this service, contact the authors"
            chatbot.message.reply_text(enable_message)                    

    # message to send when /help is received
    def send_help(self, chatbot, update) -> None:
        help_message =  'Authors: @GianlucaLM  @francescodis  @leomaggio \n'
        help_message += 'Feel free to write us!\n'
        chatbot.message.reply_text(help_message, parse_mode = telegram.ParseMode.MARKDOWN)
       
    # message to send when /enable is received
    def live_video(self, chatbot, update) -> None:
        chat_id = chatbot.message.chat_id
        db = self._read_db()
        db.set_index('ids',inplace = True)
        
        if chat_id in db.index:
            if chat_id == 99706017:
                enable_message = 'Hey Gianluca, here\'s the link: \n"http://raspberrypi.local:8000/index.html'
                chatbot.message.reply_text(enable_message)
            if chat_id == 129347830:
                enable_message = 'Hey Francesco, here\'s the link: \n"http://raspberrypi.local:8000/index.html'
                chatbot.message.reply_text(enable_message)
            if chat_id == 171207972:
                enable_message = 'Hey Leonardo, here\'s the link: \n"http://raspberrypi.local:8000/index.html'
                chatbot.message.reply_text(enable_message)
                   
        else: 
            chatbot.message.reply_text("Access denied ❌")
    
    def report(self, chatbot, update) -> None:
        string = ""
        chat_id = chatbot.message.chat_id
        db = self._read_db()
        db.set_index('ids',inplace = True)

        if chat_id in db.index:
            if not os.path.exists("bot/reports.txt"):
                chatbot.message.reply_text("There are no events")
            

            enable_message = 'Hey "User", this is the full report'
            reports = np.genfromtxt('bot/reports.txt',dtype='str')
            for report in reports:
                string += "•" + report + "\n"
            chatbot.message.reply_text(enable_message)
            chatbot.message.reply_text(string)

        else: 
            chatbot.message.reply_text("Access denied negato❌")

    # start the bot
    def send_alarm(self,timestampm,input_type, path) -> int:

        reports = []
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        db = self._read_db()
        db.set_index('ids',inplace = True)
            
        if db.at[99706017,"status"] == True:
            message = "⚠️ *Intrusion Alert* ⚠️\n" + "🕚 " + current_time
            self._send_text(message,99706017)

            if input_type == 'video':
                self._send_video(open(path, 'rb'),99706017)
            elif input_type == 'img':
                self._send_img(open(path, 'rb'),99706017)
        
    
        if db.at[129347830,"status"] == True:
            message = "⚠️ *Intrusion Alert* ⚠️\n" + "🕚 " + current_time
            self._send_text(message,129347830)
            if input_type == 'video':
                self._send_video(open(path, 'rb'),129347830)
            elif input_type == 'img':
                self._send_img(open(path, 'rb'),129347830)
            
        

        if db.at[171207972,"status"] == True:
            message = "⚠️ *Intrusion Alert* ⚠️\n" + "🕚 " + current_time
            self._send_text(message,171207972)
            if input_type == 'video':
                self._send_video(open(path, 'rb'),171207972)
            elif input_type == 'img':
                self._send_img(open(path, 'rb'),171207972)
        
        reports.append(current_time)
        arr = np.genfromtxt('bot/reports.txt',dtype='str')
        reports = np.append(arr,str(current_time))
        np.savetxt('reports.txt', reports, delimiter=" ", fmt="%s")
    
        # self.logger.info("Polling BOT.")
        # self.updater.start_polling()
        # Run the BOT until you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the BOT gracefully.
        # self.updater.idle()


    def _send_text(self,bot_message,chat_id):
        send_text = 'https://api.telegram.org/bot' + self.TOKEN + '/sendMessage?chat_id=' + str(chat_id) + '&parse_mode=Markdown&text=' + bot_message
        response = requests.get(send_text)
        return response.json()


    def _send_video(self,file_opened,bot_chatID):
        method = "sendVideo"
        params = {'chat_id': bot_chatID}
        files = {'video': file_opened}
        url = 'https://api.telegram.org/bot' + self.TOKEN + "/"
        response = requests.post(url + method, params, files=files)
        return response.json()


    def _send_img(self,file_opened,bot_chatID):
        method = "sendPhoto"
        params = {'chat_id': bot_chatID}
        files = {'photo': file_opened}
        url = 'https://api.telegram.org/bot' + self.TOKEN + "/"
        response = requests.post(url + method, params, files=files)
        return response.json()


    def _read_db(self):      
        df = pd.read_csv("bot/users.csv")
        return df


    def _update_db(self,df,index,column,value):
        df.at[index, column] = value
        df.to_csv("bot/users.csv")