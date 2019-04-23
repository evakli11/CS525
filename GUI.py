import tkinter as tk
import search_training as st
import tkinter.ttk as ttk
import pandas as pd


class App:
    def __init__(self,master):
        self.label1 = tk.Label(window, text='Insert Keywords:', font=('Helvetica', 14), width=15, height=2).place(x=80,y=100)

        self.kw_input = tk.Entry(window)
        self.kw_input.place(x=200,y=103)

        self.label2 = tk.Label(window, text='Choose Sentiment:', font=('Helvetica', 14), width=15, height=2).place(x=80,y=150)

        self.sentiment = tk.StringVar()
        self.sent = ttk.Combobox(window, textvariable=self.sentiment,width=12,state='readonly')
        self.sent["values"] = ("Positive","Negative", "All")
        self.sent.current(0)
        self.sent.place(x=220,y=153)

        self.extend = tk.Button(window, text="Generate", width=15, height=2, command=self.insert_point)
        self.extend.place(x=170,y=240)

        self.label3 = tk.Label(window, text='Generated Lyrics:', font=('Helvetica', 14), width=15, height=2).place(x=170,y=280)


        self.lyric = tk.Text(window,width=70, height=30, bg = '#FEF5E7',state='disabled')
        self.lyric.place(x=0,y=320)

        self.search = st.search()


    def insert_point(self):
        keywords = self.kw_input.get()
        sentiments = self.sentiment.get()
        index_set,sentiments= self.search.search_indexes(keywords,sentiments)
        data = pd.read_csv('processed_lyrics.csv')
        training_data = data[(data['index'].isin(list(index_set))) & (data['sentiment'].isin(sentiments))]
        if len(training_data) > 500:
            training_data = training_data.sample(n=500, axis=0)
        training_data['lyrics'].to_csv('training_lyrics.txt', index=False)
        #-----train and generating process----#


        self.lyric.config(state='normal')
        self.lyric.insert('insert', sentiments)
        self.lyric.config(state='disabled')


window = tk.Tk()
window.title('DS595 Rapper Maker')
window.geometry('500x700')  # window size
display = App(window)
window.mainloop()