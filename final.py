#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 20:12:04 2019

@author: pengyiye
"""

import pronouncing
import markovify
import re
import random
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import LSTM 
from keras.layers.core import Dense
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class neuralRapper:
    
    def __init__(self, artist, o_filename):
        self.artist = None
        self.output = None
        self.depth = 4
        self.maxsyllables = 8
        self.train_mode = False
        
    def create_network(self,depth):
        model = Sequential()
        model.add(LSTM(4, input_shape=(2, 2), return_sequences=True))
        for i in range(depth):
            model.add(LSTM(8, return_sequences=True))
        model.add(LSTM(2, return_sequences=True))
        model.summary()
        model.compile(optimizer='rmsprop',loss='mse')
        if (self.artist + '.rap') in os.listdir(".") and self.train_mode == False:
            model.load_weights(self.artist+'.rap')
            print("loading saved network:" + str(self.artist)+".rap")
        return model
    
    def markov(self,text_file):
        read = open(text_file, 'r').read()
        text_model = markovify.NewlineText(read)
        
        return text_model
    
    def syllables(self,line):
        count = 0
        #print(line)
        tokenList = [x for x in line.split(" ") if x.isalpha()]
        for word in tokenList:
            vowels = 'aeiouy'
            word = word.lower().strip(".:;?!")
            if word[0] in vowels:
                count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index - 1] not in vowels:
                    count += 1
            if word.endswith('e'):
                count -= 1
            if word.endswith('le'):
                count += 1
            if count == 0:
                count += 1
           

        return count / self.maxsyllables
    
    # writes a rhyme list to a rhymes file that allows for use when
# building the dataset, and composing the rap
    def rhymeindex(self,lyrics):
        if str(self.artist) + ".rhymes" in os.listdir(".") and self.train_mode == False:
            print("loading saved rhymes from " + str(self.artist) + ".rhymes")
            return open(str(self.artist) + ".rhymes", "r").read().split("\n")
        else:
            rhyme_master_list = []
            print("Alright, building the list of all the rhymes")
            for i in lyrics:
                # grabs the last word in each bar
                word = re.sub(r"\W+", '', i.split(" ")[-1]).lower()
                # pronouncing.rhymes gives us a word that rhymes with the word being passed in
                rhymeslist = pronouncing.rhymes(word)
                # need to convert the unicode rhyme words to UTF8
                rhymeslist = [x.encode('UTF8') for x in rhymeslist]
                # rhymeslistends contains the last two characters for each word
                # that could potentially rhyme with our word
                rhymeslistends = []
                for i in rhymeslist:
                    rhymeslistends.append(i[-2:])
                try:
                    # rhymescheme gets all the unique two letter endings and then
                    # finds the one that occurs the most
                    rhymescheme = max(set(rhymeslistends), key=rhymeslistends.count)
                except Exception:
                    rhymescheme = word[-2:]
                rhyme_master_list.append(rhymescheme)
            # rhyme_master_list is a list of the two letters endings that appear
            # the most in the rhyme list for the word
            rhyme_master_list = list(set(rhyme_master_list))
    
            reverselist = [ x[::-1] for x in rhyme_master_list]
            reverselist = [ x.decode("utf-8") for x in reverselist if type(x)== bytes ]
            reverselist = sorted(reverselist)
            # rhymelist is a list of the two letter endings (reversed)
            # the reason the letters are reversed and sorted is so
            # if the network messes up a little bit and doesn't return quite
            # the right values, it can often lead to picking the rhyme ending next to the
            # expected one in the list. But now the endings will be sorted and close together
            # so if the network messes up, that's alright and as long as it's just close to the
            # correct rhymes
            rhymelist = [x[::-1] for x in reverselist]
    
            f = open(str(self.artist) + ".rhymes", "w")
            f.write("\n".join(rhymelist))
            f.close()
            print(rhymelist)
            return rhymelist
        
    # converts the index of the most common rhyme ending
    # into a float
    def rhyme(self,line, rhyme_list):
        word = re.sub(r"\W+", '', line.split(" ")[-1]).lower()
        rhymeslist = pronouncing.rhymes(word)
        rhymeslist = [x.encode('UTF8') for x in rhymeslist]
        rhymeslistends = []
        for i in rhymeslist:
            rhymeslistends.append(i[-2:])
        try:
            rhymescheme = max(set(rhymeslistends), key=rhymeslistends.count)
        except Exception:
            rhymescheme = word[-2:]
        try:
            float_rhyme = rhyme_list.index(rhymescheme)
            float_rhyme = float_rhyme / float(len(rhyme_list))
            return float_rhyme
        except Exception:
            return 0.
        
    # grabs each line of the lyrics file and puts them
    # in their own index of a list, and then removes any empty lines
    # from the lyrics file and returns the list as bars
    def split_lyrics_file(self,text_file):
        text = open(text_file).read()
        text = text.split("\n")
        while "" in text:
            text.remove("")
        return text
    
    # only ran when not training
    def generate_lyrics(self,lyrics_file):
        bars = []
        last_words = []
        lyriclength = len(open(lyrics_file).read().split("\n"))
        count = 0
        markov_model = self.markov(lyrics_file)
    
        while len(bars) < lyriclength / 9 and count < lyriclength * 2:
            # By default, the make_sentence method tries, a maximum of 10 times per invocation,
            # to make a sentence that doesn't overlap too much with the original text.
            # If it is successful, the method returns the sentence as a string.
            # If not, it returns None. (https://github.com/jsvine/markovify)
            bar = markov_model.make_sentence()
    
            # make sure the bar isn't 'None' and that the amount of
            # syllables is under the max syllables
            if type(bar) != type(None) and self.syllables(bar) < 1:
    
                # function to get the last word of the bar
                def get_last_word(bar):
                    last_word = bar.split(" ")[-1]
                    # if the last word is punctuation, get the word before it
                    if last_word[-1] in "!.?,":
                        last_word = last_word[:-1]
                    return last_word
    
                last_word = get_last_word(bar)
                # only use the bar if it is unique and the last_word
                # has only been seen less than 3 times
                if bar not in bars and last_words.count(last_word) < 3:
                    bars.append(bar)
                    last_words.append(last_word)
                    count += 1
    
        return bars
    
    # used to construct the 2x2 inputs for the LSTMs
    # the lyrics being passed in are lyrics (original lyrics if being trained,
    # or ours if it's already trained)
    def build_dataset(self,lyrics, rhyme_list):
        dataset = []
        line_list = []
        # line_list becomes a list of the line from the lyrics, the syllables for that line (either 0 or 1 since
        # syllables uses integer division by maxsyllables (16)), and then rhyme returns the most common word
        # endings of the words that could rhyme with the last word of line
        for line in lyrics:
            line_list = [line, self.syllables(line), self.rhyme(line, rhyme_list)]
            dataset.append(line_list)
    
        x_data = []
        y_data = []
    
        # using range(len(dataset)) - 3 because of the way the indices are accessed to
        # get the lines
        for i in range(len(dataset) - 3):
            line1 = dataset[i][1:]
            line2 = dataset[i + 1][1:]
            line3 = dataset[i + 2][1:]
            line4 = dataset[i + 3][1:]
    
            # populate the training data
            # grabs the syllables and rhyme index here
            x = [line1[0], line1[1], line2[0], line2[1]]
            x = np.array(x)
            # the data is shaped as a 2x2 array where each row is a
            # [syllable, rhyme_index] pair
            x = x.reshape(2, 2)
            x_data.append(x)
    
            # populate the target data
            y = [line3[0], line3[1], line4[0], line4[1]]
            y = np.array(y)
            y = y.reshape(2, 2)
            y_data.append(y)
    
        # returns the 2x2 arrays as datasets
        x_data = np.array(x_data)
        y_data = np.array(y_data)
    
        # print "x shape " + str(x_data.shape)
        # print "y shape " + str(y_data.shape)
        return x_data, y_data

    # only used when not training
    def compose_rap(self,lines, rhyme_list, lyrics_file, model):
        rap_vectors = []
        human_lyrics = self.split_lyrics_file(lyrics_file)
    
        # choose a random line to start in from given lyrics
        initial_index = random.choice(range(len(human_lyrics) - 1))
        # create an initial_lines list consisting of 2 lines
        initial_lines = human_lyrics[initial_index:initial_index + 8]
    
        starting_input = []
        for line in initial_lines:
            # appends a [syllable, rhyme_index] pair to starting_input
            starting_input.append([self.syllables(line), self.rhyme(line, rhyme_list)])
    
        # predict generates output predictions for the given samples
        # it's reshaped as a (1, 2, 2) so that the model can predict each
        # 2x2 matrix of [syllable, rhyme_index] pairs
        starting_vectors = model.predict(np.array([starting_input]).flatten().reshape(4, 2, 2))
        rap_vectors.append(starting_vectors)
    
        for i in range(49):
            rap_vectors.append(model.predict(np.array([rap_vectors[-1]]).flatten().reshape(4, 2, 2)))
    
        return rap_vectors
    
    def vectors_into_song(self,vectors, generated_lyrics, rhyme_list):
        print('\n\n')
        print("About to write rap (this could take a moment)...")
        print("\n\n")
        def last_word_compare(rap, line2):
            penalty = 0
            for line1 in rap:
                word1 = line1.split(" ")[-1]
                word2 = line2.split(" ")[-1]
                if word1.isalpha() * word2.isalpha():
                    while word1[-1] in "?!,. ":
                        word1 = word1[:-1]
                    while word2[-1] in "?!,. ":
                        word2 = word2[:-1]
                    if word1 == word2:
                        penalty += 0.2
                else:
                    pass
                return penalty
    
        def calculate_score(vector_half, syllables, rhyme, penalty):
            desired_syllables = vector_half[0]
            desired_rhyme = vector_half[1]
            desired_syllables = desired_syllables * self.maxsyllables
            desired_rhyme = desired_rhyme * len(rhyme_list)
            score = 1.0 - (abs((float(desired_syllables) - float(syllables))) + abs((float(desired_rhyme) - float(rhyme)))) - penalty
            
            return score
        
        dataset = []
        
        for line in generated_lyrics:
            line_list = [line, self.syllables(line), self.rhyme(line, rhyme_list)]
            dataset.append(line_list)
            
        rap = []
        vector_halves = []
        for vector in vectors:
            vector_halves.append(list(vector[0][0])) 
            vector_halves.append(list(vector[0][1]))
        
        for vector in vector_halves:
            scorelist = []
            for item in dataset:
                line = item[0]
                if len(rap) != 0:
                    penalty = last_word_compare(rap, line)
                else:
                    penalty = 0
                total_score = calculate_score(vector, item[1], item[2], penalty)
                score_entry = [line, total_score]
                scorelist.append(score_entry)
                
            fixed_score_list = []
            for score in scorelist:
                fixed_score_list.append(float(score[1]))
            max_score = max(fixed_score_list)
            for item in scorelist:
                if item[1] == max_score:
                    rap.append(item[0])
                    #print(str(item[0]))
                    for i in dataset:
                        if item[0] == i[0]:
                            dataset.remove(i)
                            break
                    break
        return rap
    
    def train(self,x_data, y_data, model):
        # fit is used to train the model for 5 'epochs' (iterations) where
        # the x_data is the training data, and the y_data is the target data
        # x is the training and y is the target data
        # batch_size is a subset of the training data (2 in this case)
        # verbose simply shows a progress bar
        model.fit(np.array(x_data), np.array(y_data),
                  batch_size=2,
                  epochs=5,
                  verbose=1)
        # save_weights saves the best weights from training to a hdf5 file
        model.save_weights(self.artist + ".rap")
        
    def main(self, artist, outputFile, trainfile):
        self.artist = artist
        self.output = outputFile
        model = self.create_network(self.depth)
        # change the lyrics file to the file with the lyrics you want to be trained on
        text_file = trainfile

        if self.train_mode == True:
            bars = self.split_lyrics_file(text_file)
    
        if self.train_mode == False:
            bars = self.generate_lyrics(text_file)
    
        rhyme_list = self.rhymeindex(bars)
        if self.train_mode == True:
            x_data, y_data = self.build_dataset(bars, rhyme_list)
            self.train(x_data, y_data, model)
    
        if self.train_mode == False:
            vectors = self.compose_rap(bars, rhyme_list, text_file, model)
            rap = self.vectors_into_song(vectors, bars, rhyme_list)
            f = open(self.output, "w")
            for bar in rap:
                f.write(bar)
                f.write("\n")
        

 # depth of the network. changing will require a retrain

#artist = "kanye_west" # used when saving the trained model
#rap_file = "neural_rap.txt" # where the rap is written to
#n = neuralRapper("kanye_west",'neural_rap.txt')
#n.main("kanye_west", 'neural_rap.txt',"training_lyrics.txt")