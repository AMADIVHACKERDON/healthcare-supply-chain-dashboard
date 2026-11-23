from constance import API_KEY

import telebot
from telebot import types

bot = telebot.TeleBot(API_KEY, parse_mode = None)

@bot.message_handler(commands = ["help", "hello"])
def send_help_message(message):
    bot.reply_to(message, "Hello! This is the test bot")

@bot.message_handler(commands= ["close"])
def send_content_message(message):
    markup = types.ReplyKeyboardRemove(selective=False)
    bot.send_message(chat_id=message.chat.id, text="Transaction Ended", reply_markup=markup)

@bot.message_handler(commands=["send"])
def send_game_message(message):
    markup =  types.ReplyKeyboardMarkup(row_width=2)
    btn = types.KeyboardButton("Payment Integration")
    markup.add(btn) 
    bot.send_message(chat_id=message.chat.id, text="What do you want?", reply_markup=markup)


@bot.message_handler(commands= ['pay'])
def send_welcome(message):
    button1 = types.InlineKeyboardButton(text = "Payment Integration" ,callback_data="go_next")
    markup = types.InlineKeyboardMarkup().add(button1)
    bot.send_message(message.chat.id,'Choose an option', reply_markup=markup)

@bot.callback_query_handler(func= lambda call:call.data == "go_next")
def show_second_buttons(call):
    mar = types.InlineKeyboardButton("Option A", callback_data= "opt_a")
    mark = types.InlineKeyboardButton("Option B", callback_data= "opt_b")
    markup = types.InlineKeyboardMarkup().add(mar, mark)
    bot.edit_message_text("Choose an option:", call.message.chat.id,call.message.message_id, reply_markup=markup)

@bot.callback_query_handler(func= lambda call:call.data in ["opt_a","opt_b"])
def ask_for_secret_key(call):
    choice = call.data
    bot.send_message(call.message.chat.id, f"You selected {choice}. Now enter your secret key")     

bot.polling()