# -*- coding: UTF-8 -*-
#pre train model https://drive.google.com/file/d/1Vwx7Q3lofAop-VkIYfTZ8ceLIL3m-Sag/view?usp=sharing

#pip install tkinter
#pip install torch
from tkinter import *
import torch

from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer
)

# settings
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
model = AutoModelWithLMHead.from_pretrained('./model/wizard_wiki')

#UI 設定
#————————————————————————————————————————————————————————————————————————
BG_GRAY = '#ABB2B9'
BG_COLOR = '#17202A'
TEXT_COLOR = '#EAECEE'

FONT = 'Helvetica 14'
FONT_BOLD = 'Helvetica 13 bold'
Bot_name = 'Bot'

class ChatApplication:
    
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
    
    def run(self):
        self.window.mainloop()
        
    def _setup_main_window(self):
        self.window.title('ChatBot')
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)
        
        # head lable
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text='Welcome', font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)
        
        #tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)
        
        #text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor='arrow', state=DISABLED)
        
        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)
        
        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        
        #message entry box
        self.msg_entry = Entry(bottom_label, bg='#2C3E50', fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind('<Return>', self._on_enter_pressed) # press enter to send msg
        
        # send button
        send_botton = Button(bottom_label, text='Send', font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_pressed(None))
        send_botton.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
        
        
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, sender='User')
        
    #——————————————————————————————————————————————————————————————————————————————  

    def _insert_message(self, msg, sender='User'):
        # declare global
        global chat_history_ids
        
        if not msg:
            return
        
        # create msg
        self.msg_entry.delete(0, END)
        # user: msg1
        msg1 = f'{sender}: {msg}\n\n'
        new_user_input_ids = tokenizer.encode(msg + tokenizer.eos_token,
                                              return_tensors='pt')
        # print(new_user_input_ids)

        # append the new user input tokens to the chat history
        if 'chat_history_ids' in globals():
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        
        chat_history_ids = model.generate(
            bot_input_ids, max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.92, top_k = 50
            )
        
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
        
        # bot response
        response = "{}".format(tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True))
        msg2 = f'{Bot_name}: {response}\n\n'
        
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)
        
        self.text_widget.see(END)

if __name__ == '__main__':
    app = ChatApplication()
    app.run()
