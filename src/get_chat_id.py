import json
import telebot

with open(r"C:\Users\LENOVO\Desktop\AI-Public-Safety-Monitor\telegram_config.json", "r") as f:

    cfg = json.load(f)

bot = telebot.TeleBot(cfg["bot_token"])

@bot.message_handler(func=lambda message: True)
def get_id(message):
    print("Your chat_id:", message.chat.id)
    bot.reply_to(message, f"Your chat_id is: {message.chat.id}")

print("Send ANY message to your bot now…")
bot.infinity_polling()
