#%%
import lda2vec.Lda2vec
import pandas as pd
import lda2vec.nlppipe
import spacy
import numpy as np
import re
import lda2vec.utils

# %%

data = pd.read_csv('Data\Avis_Id_Text.csv',index_col='index')

#%%
data_theme = pd.read_csv('Data\etudes_avis_dep_theme.csv', delimiter = ';')
data_theme.replace('nan',np.nan, inplace = True)
data_theme.dropna(axis = 0, subset = ['url_avis'], inplace = True)
#%%
def extract_idAAE(val):
    idAAE = ''
    approx = val[58:]
    for car in approx:
        if car.isdigit()==True:
            idAAE += car
        else:
            return(int(idAAE))

        
data_theme['id_AAE']=data_theme['url_avis'].apply(extract_idAAE)

#%%

data['len'] = data.txt_AAE.str.len()
too_smol = data[data['len'] < 6000].index
data.drop(too_smol,inplace = True)

#%%
data_final = data.merge(data_theme, on = ['id_AAE'], how = 'inner')

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
path = 'Avis_Id_TextProcessed.csv'
data_processed = lda2vec.utils.load_preprocessed_data(path)

#%%
len(np.unique(data_processed[3]))
#%%
model = lda2vec.Lda2vec.Lda2vec(num_unique_documents = data_processor.num_docs, vocab_size = data_processor.vocab_size, num_topics = 10)

# %%
