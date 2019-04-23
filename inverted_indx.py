from collections import defaultdict
import pickle
import pandas as pd

def inverted_index(list_dict):
    inv_indx = defaultdict(list)
    for idx, text in enumerate(list_dict):
        for word in text:
            inv_indx[word].append(idx)
    # number = len(self.inv_indx)
    return inv_indx

def strtolist(doc):
    return set(doc.split(' '))

data = pd.read_csv('processed_lyrics.csv')
del data['Unnamed: 0']
list_dict = list(data['keywords'].map(strtolist))
inv_indx = inverted_index(list_dict)
print(len(inv_indx))

output = open('inverted_index.pkl', 'wb')
pickle.dump(inv_indx, output)
output.close()


