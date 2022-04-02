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
        welcome_message =  "*Ciao, sono il bot che controlla la tua casa* \n\n"
        welcome_message += '🗞 Digita: /enable per ricevere gli avvisi!\n\n'
        welcome_message += '❌ Digita: /disable per disattivare le notifiche!\n\n'
        welcome_message += '📓 Digita: /report per visualizzare lo storico degli avvisi!\n\n'
        welcome_message += '🎥 Digita: /live_video per guardare cosa sta succedendo a casa.\n\n'
        welcome_message += '🆘 Digita: /help per contattare gli autori.\n\n'
        chatbot.message.reply_text(welcome_message)
    
   
    def send_enable(self, chatbot, update) -> None:
        # write the chat id in the database
        chat_id = chatbot.message.chat_id
        db = self._read_db()
        db.set_index('ids',inplace = True)

        if chat_id in db.index:
            if db.at[chat_id,'status'] == True:
                enable_message = '✅ Sei già iscritto al servizio di notifiche'
                chatbot.message.reply_text(enable_message)
            else:
                enable_message = '✅ Riceverai tutti gli aggiornamenti sulla tua casa'
                self._update_db(db,chat_id,'status',True)
                chatbot.message.reply_text(enable_message)
        else:
            enable_message = 'Non sei abilitato a questo servizio, contattare gli amministatori'
            chatbot.message.reply_text(enable_message)

    def send_disable(self, chatbot, update) -> None:
        # write the chat id in the database
        chat_id = chatbot.message.chat_id
        db = self._read_db()
        db.set_index('ids',inplace = True)

        if chat_id in db.index:
            if db.at[chat_id,'status'] == True:
                enable_message = '❌ Non riceverai più alcuna notifica'
                chatbot.message.reply_text(enable_message)
                self._update_db(db,chat_id,'status',False)

            else:
                enable_message = 'Non sei iscritto al servizio di notifiche'
                chatbot.message.reply_text(enable_message)

        else:
            enable_message = 'Non sei abilitato a questo servizio, contattare gli amministatori'
            chatbot.message.reply_text(enable_message)                    

    # message to send when /help is received
    def send_help(self, chatbot, update) -> None:
        help_message =  'Authors: @GianlucaLM  @francescodis  @leomaggio \n'
        help_message += 'Scrivici per qualsiasi problema\n'
        chatbot.message.reply_text(help_message, parse_mode = telegram.ParseMode.MARKDOWN)
       
    # message to send when /enable is received
    def live_video(self, chatbot, update) -> None:
        chat_id = chatbot.message.chat_id
        db = self._read_db()
        db.set_index('ids',inplace = True)
        
        if chat_id in db.index:
            if chat_id == 99706017:
                enable_message = 'Ciao Gianluca, ecco il link:\n"http://raspberrypi.local:8000/index.html'
                chatbot.message.reply_text(enable_message)
            if chat_id == 129347830:
                enable_message = 'Ciao Francesco, ecco il link:\n"http://raspberrypi.local:8000/index.html'
                chatbot.message.reply_text(enable_message)
            if chat_id == 171207972:
                enable_message = 'Ciao Leonardo, ecco il link:\n"http://raspberrypi.local:8000/index.html'
                chatbot.message.reply_text(enable_message)
                   
        else: 
            chatbot.message.reply_text("Accesso negato❌")
    
    def report(self, chatbot, update) -> None:
        string = ""
        chat_id = chatbot.message.chat_id
        db = self._read_db()
        db.set_index('ids',inplace = True)

        if chat_id in db.index:
            if not os.path.exists("bot/reports.txt"):
                chatbot.message.reply_text("Non ci sono avvertimenti")
            

            enable_message = 'Ciao "User" ecco lo storico degli avvertimenti'
            reports = np.genfromtxt('bot/reports.txt',dtype='str')
            for report in reports:
                string += "•" + report + "\n"
            chatbot.message.reply_text(enable_message)
            chatbot.message.reply_text(string)

        else: 
            chatbot.message.reply_text("Accesso negato❌")

    # start the bot
    def send_alarm(self,timestampm,img_path) -> int:

        reports = []
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        db = self._read_db()
        db.set_index('ids',inplace = True)
            
        if db.at[99706017,"status"] == True:
            message = "⚠️ *Allarme Intrusione* ⚠️\n" + "🕚 Orario: " + current_time + "\n\nDai un'occhiata a cosa sta succedendo:\nhttp://raspberrypi.local:8000/index.html"
            self._send_text(message,99706017)
            self._send_photo(open(img_path, 'rb'),99706017)
        
    
        if db.at[129347830,"status"] == True:
            message = "⚠️ *Allarme Intrusione* ⚠️\n" + "🕚 Orario: " + current_time + "\n\nDai un'occhiata a cosa sta succedendo: \n\nhttp://raspberrypi.local:8000/index.html"
            self._send_text(message,129347830)
            self._send_photo(open(img_path, 'rb'),129347830)
        

        if db.at[171207972,"status"] == True:
            message = "⚠️ *Allarme Intrusione* ⚠️\n" + "🕚 Orario: " + current_time + "\n\nDai un'occhiata a cosa sta succedendo: \n\nhttp://raspberrypi.local:8000/index.html"
            self._send_text(message,171207972)
            self._send_photo(open(img_path, 'rb'),171207972)
        
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

    def _send_photo(self,file_opened,bot_chatID):
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