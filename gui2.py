#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:27:27 2019

@author: pengyiye
"""

import tkinter as tk

class App:
    def __init__(self, master):
        tk.Label(window,text='Insert Keyword',font=('Arial', 14), width=15, height=2).pack()
        
        self.kw_input = tk.Entry(window)
        self.kw_input.pack()
        
        tk.Radiobutton(window, text='Positive',value='A').pack()
        tk.Radiobutton(window, text='Negative',value='B').pack()
        
        self.extend = tk.Button(window,text="Generate",width=15,height=2,command=self.insert_point)
        self.extend.pack()
        
        self.lyric = tk.Text(window,height=2)
        self.lyric.pack()
        
    def insert_point(self):
        var = self.kw_input.get()
        self.lyric.insert('insert',var)

window = tk.Tk()
window.title('DS595_RAPPER')
window.geometry('300x400')#window size
display = App(window)
window.mainloop()