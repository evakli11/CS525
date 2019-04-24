"""
Created on Tue Apr 23

@author: Bernie Ye
"""
from tkinter import *
import search_training as st
import tkinter.ttk as ttk
import pandas as pd
import final
from PIL import ImageTk, Image

def insert_point():
    keywords = kw_input.get()
    sentiments = sentiment.get()
    index_set,sentiments= search.search_indexes(keywords,sentiments)
    data = pd.read_csv('processed_lyrics.csv')
    training_data = data[(data['index'].isin(list(index_set))) & (data['sentiment'].isin(sentiments))]
    if len(training_data) > 500:
        training_data = training_data.sample(n=500, axis=0)
    training_data['lyrics'].to_csv('training_lyrics.txt', index=False)
    #-----train and generating process----#
    n = final.neuralRapper("kanye_west",'neural_rap.txt')
    n.main("kanye_west", 'neural_rap.txt', "training_lyrics.txt")
    #self.lyric.insert('insert', sentiments)
    lyric.config(state='normal')
    lyric.delete("1.0", "end")
    with open('neural_rap.txt','r') as f:
        lyrics = f.read()
        print(lyrics)
    lyric.insert('insert', lyrics)
    lyric.config(state='disabled')

def play():
    import speech
        
root = Tk()
root.title('Rap Maker')
root.geometry('500x700')
canv = Canvas(root, width=500, height=700, bg='white')
canv.place(x=0,y=0)
img = ImageTk.PhotoImage(Image.open("bgfullsmall.jpg"))  # PIL solution
canv.create_image(250, 350, anchor='center', image=img)

#label1 = Label(root, text='Insert Keywords:', font=('Helvetica', 14), width=15, height=1)
#label1.place(x=80,y=100)

kw_input = Entry(root)
kw_input.place(x=200,y=103)

#label2 = Label(root, text='Choose Sentiment:', font=('Helvetica', 14), width=15, height=1).place(x=80,y=150)##

sentiment = StringVar()
sent = ttk.Combobox(root, textvariable=sentiment,width=12,state='readonly')
sent["values"] = ("Positive","Negative", "All")
sent.current(0)
sent.place(x=220,y=153)

extend1 = Button(root, text="Generate", width=15, height=2, command=insert_point)
extend1.place(x=170,y=240)

#label3 = Label(root, text='Generated Lyrics:', font=('Helvetica', 14), width=15, height=2).place(x=170,y=280)


lyric = Text(root,width=70, height=18, bg = '#FEF5E7',state='disabled')
lyric.place(x=0,y=355)

extend2 = Button(root, text="Play MP3", width=15, height=2, command = play)
extend2.place(x=170, y=650)

search = st.search()

mainloop()
