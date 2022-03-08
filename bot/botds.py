from urllib.parse import uses_params
from telegram.ext.callbackcontext import CallbackContext
from datetime import datetime
import logging
import time
import requests
import telegram
from telegram.ext import Updater, CommandHandler
import csv
import os.path
import os
import numpy as np


TOKEN = '5165021744:AAGqhFc_5heY5EXaBulsRy_HGeC67diZFGs'
def send_text(bot_message):
        
    send_text = 'https://api.telegram.org/bot' + TOKEN + '/sendMessage?chat_id=' + '99706017' + '&parse_mode=Markdown&text=' + bot_message
    response = requests.get(send_text)
    return response.json()
        

class Bot:

    def __init__(self,danger) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger("LOG")
        self.logger.info("Starting BOT.")
        self.updater = Updater(TOKEN)
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
        welcome_message += '📓 Digita: /recap per visualizzare lo storico degli avvisi!\n\n'
        welcome_message += '🎥 Digita: /live_video per guardare cosa sta succedendo a casa.\n\n'
        welcome_message += '🆘 Digita: /help per contattare gli autori.\n\n'
        chatbot.message.reply_text(welcome_message)
    
   
    def send_enable(self, chatbot, update) -> None:
        # write the chat id in the database
        chat_id = chatbot.message.chat_id
        users = []
        if not os.path.exists("users.csv"):
                    users.append(chat_id)
                    np.savetxt("users.csv",users,delimiter =", ", fmt ='% s')
                    # send the confermation message
                    enable_message = '✅ Riceverai tutti gli aggiornamenti sulla tua casa'
                    chatbot.message.reply_text(enable_message)
                
        else:
            with open('users.csv') as csv_file:
                csv_reader = csv.reader(csv_file)
                for line in csv_reader:
                    
                    users.append(str(line).replace("[","").replace("]","").replace('\'',""))
                    print(users)

            if str(chat_id) not in users:
                users.append(chat_id)
                with open("users.csv", 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(users)
                    # send the confermation message
                    enable_message = '✅ Riceverai tutti gli aggiornamenti sulla tua casa'
                    chatbot.message.reply_text(enable_message)
                
            else:
                chatbot.message.reply_text("Sei già iscritto al servizio di notifiche")

    def send_disable(self, chatbot, update) -> None:
        # write the chat id in the database
        chat_id = chatbot.message.chat_id
        users = []
        if not os.path.exists("users.csv"):
                    enable_message = 'Non sei iscritto al servizio di notifiche'
                    chatbot.message.reply_text(enable_message)
                
        else:
            with open('users.csv') as csv_file:
                csv_reader = csv.reader(csv_file)
                for line in csv_reader:
                    users.append(str(line).replace("[","").replace("]","").replace('\'',""))
                
                if str(chat_id) in  users:
                    users.remove(str(chat_id))
                    enable_message = '❌ Non riceverai più alcuna notifica'
                    chatbot.message.reply_text(enable_message)
                    os.remove("users.csv")
                    
                    np.savetxt("users.csv",users,delimiter =", ", fmt ='% s')
                        
                else:
                    enable_message = 'Non sei iscritto al servizio di notifiche'
                    chatbot.message.reply_text(enable_message)

                    

    # message to send when /help is received
    def send_help(self, chatbot, update) -> None:
        help_message =  'Author: @GianlucaLM\n'
        help_message += 'Scrivimi per qualsiasi problema\n'
        chatbot.message.reply_text(help_message, parse_mode = telegram.ParseMode.MARKDOWN)
       
    # message to send when /enable is received
    def live_video(self, chatbot, update) -> None:
        
        if chatbot.message.chat_id == 99706017:
            # send the confermation message
            enable_message = 'Ciao "User" ecco il link'
            chatbot.message.reply_text(enable_message)
            
        else: 
            chatbot.message.reply_text("Accesso negato❌")
    
    def report(self, chatbot, update) -> None:
        strings = ""
        if chatbot.message.chat_id == 99706017:
            if not os.path.exists("reports.csv"):
                chatbot.message.reply_text("Non ci sono avvertimenti")
            

            enable_message = 'Ciao "User" ecco lo storico degli avvertimenti'
            with open('reports.csv') as csv_file:
                csv_reader = csv.reader(csv_file)
                for line in csv_reader:
                    print(line)
                    strings += "•" + str(line).replace("[","").replace("]","").replace('\'',"") + "\n"
            
            chatbot.message.reply_text(enable_message)
            chatbot.message.reply_text(strings)

        else: 
            chatbot.message.reply_text("Accesso negato❌")

    # start the bot
    def run(self) -> int:
        reports = []
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        users = []
        if os.path.exists("users.csv"):
            with open('users.csv') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    for line in csv_reader:
                        users.append(str(line))
        print(users)
        if self.danger and '[\'99706017\']' in users:
            message = "⚠️ *Allarme Intrusione* ⚠️\n" + "Orario🕚 :" + current_time
            send_text(message)
            reports.append(current_time)
            if not os.path.exists("reports.csv"):
                np.savetxt("reports.csv",reports,delimiter =", ", fmt ='% s')
            else:
                with open("reports.csv", 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(reports)
        
        self.logger.info("Polling BOT.")
        self.updater.start_polling()
        # Run the BOT until you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the BOT gracefully.
        self.updater.idle()
        


if __name__ == "__main__":

    BOT = Bot(True)
    BOT.run()
    
