#%%
import lda2vec.Lda2vec
import pandas as pd
import lda2vec.nlppipe
import spacy
import numpy as np
import re
import lda2vec.utils

#Depuis \Ruben
path = 'Avis_Id_TextProcessed'


#%%
data_final = pd.read_csv('data_final.csv')
data_final.drop(['Unnamed: 0'],axis = 1)
data_final.drop(['Unnamed: 0.1'],axis = 1)

#%%
themes=list(np.unique(re.sub("[\(\[].*?[\)\]]", "",
    re.sub(","," ",
        re.sub(";"," ",' '.join(np.unique(data_final.theme.values))))).split(' ')))
themes.remove('ET'),themes.remove('')

#%%
dfByTheme = {}
for theme in themes:
    dfByTheme[theme] = []
#%%
processorsByTheme = {}

data_processor = lda2vec.nlppipe.Preprocessor(df = data_final[data_final['len'] < 6000], textcol = 'txt_AAE')
data_processor.preprocess()
data_processor.save_data('Avis_Id_TextProcessed.csv')

#%%

data_processed = lda2vec.utils.load_preprocessed_data(path)

#%%
len(np.unique(data_processed[3]))
#%%
model = lda2vec.Lda2vec.Lda2vec(num_unique_documents = data_processor.num_docs, vocab_size = data_processor.vocab_size, num_topics = 10)

# %%
