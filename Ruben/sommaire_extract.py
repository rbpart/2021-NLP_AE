#%%
from bs4 import BeautifulSoup
import os
import unicodedata
import string
import numpy as np
# %%
dirpath = '.\Data\TreatedFEIHTML'
doc_path = os.listdir(dirpath)
# %%

def toc_extractor(filepath):
    file = unicodedata.normalize('NFKC',open(filepath,'rb').read().decode('utf-8'))
    soup = BeautifulSoup(file, features = "html")
    balises = soup.find_all('a',text = True)
    return [line.get_text().replace(u'\xa0',u'') for line in balises]

def txtBalise_extractor(filepath):
    file = unicodedata.normalize('NFKC',open(filepath,'rb').read().decode('utf-8'))
    soup = BeautifulSoup(file, features = "html")
    balises = soup.find_all()
    return balises

# %%
def id_extractor(filepath):
    id = ''
    for car in filepath:
        if car in '1234567890':
            id += car
    return(id)

# %%
# Pour chaque ligne, on extrait les features suivantes : font
def features_extractor(b4string):
    try:
        parent = b4string.parent
        features = [parent.name]
        try:
            for attr in parent.attrs:
                features.append(attr)
                try:
                    features.append(parent.attrs[attr][0])
                except:
                    features.append(parent.attrs[attr])
        except:
            pass
    except:
        return []
    

    try:
        parent_cont = parent.contents.remove(b4string)
    except:
        parent_cont = parent.contents
    try:
        for child in parent_cont:
            features.append(child.name)
            attrs_child = [attr for attr in child.attrs]
            for attr in attrs_child:
                try :
                    features.append(attr)
                    try:
                        features.append(child.attrs[attr][0])
                    except:
                        features.append(child.attrs[attr])
                    
                except:
                    pass
    except:
        pass

    gd_parent = parent.parent
    try:
        features.append(gd_parent.name)   
        try:
            for attr in gd_parent.attrs:
                features.append(attr)
                try:
                    features.append(gd_parent.attrs[attr][0])
                except:
                    features.append(gd_parent.attrs[attr])
        except:
            pass
    except:
        pass         
    while None in features:
        features.remove(None)      

    for car in b4string.get_text():
            if car in '0123456789':
                features.append('hasNumber')
                break

    return(features)




#%%
def features(doc):
    features = []
    for chain in doc:
        feats = features_extractor(chain)
        for feat in feats:
            if type(feat) == list:
                feats.replace(feat,feat[0])
        features.append(feats)
    return(features)

def features_collection(doc_collection):
    feature_collection = []
    for doc in doc_collection:
        feature_collection.append(features(doc))
    return(feature_collection)

def build_features(ftlists):
    diff_feat = []
    for entry in ftlists:
        for feats in entry:
            for feat in feats:
                diff_feat.append(feat)
    diff_feat = np.unique(diff_feat)
    
    k = 0
    feat_dico = {}
    for feat in diff_feat:
        feat_dico[feat] = k
        k += 1
    return(feat_dico)

#%%

def encode_matrix(doc_feat,features_names):
    matrix = np.zeros((len(doc_feat),len(features_names)))
    k = 0
    for rowfeat in doc_feat:
        for feat in rowfeat:
            ind = features_names[feat]
            matrix[k,ind] = 1
        k += 1
    return(matrix)

def encode_collection(docs_feature_collection,feature_names):
    matrix = encode_matrix(docs_feature_collection[0],feature_names)
    i = 0
    for doc_feat in docs_feature_collection[1:]:
        child = encode_matrix(doc_feat,feature_names)
        matrix = np.concatenate((matrix,child),axis = 0)
    return(matrix)
# %%

def Title_clean(resultsTitle):
    resultsTitleClean = []
    for result in resultsTitle:
        if 'www' in result:
            pass
        else:
            try:
                int(result)
            except:
                for car in result:
                    if car.isnumeric()==True:
                        resultsTitleClean.append(result)
                        break
    return(resultsTitleClean)
                


# %%

def Encode(results,resultsTitle):
    vector = []
    for res in results:
        clean = res.get_text()
        if clean in resultsTitle:
            vector.append(1)
        else:
            vector.append(0)
    return(vector)

def Encode_vector(doc_collection,Title_collection):
    vector = []
    for doc,titles in zip(doc_collection,Title_collection):
        vector += Encode(doc,titles)
    return(vector)

# %%

doc_collection = []
for doc in doc_path:
    doc_collection.append(txtBalise_extractor(dirpath+'\\'+doc))


# %%

feats_collection = features_collection(doc_collection)

#%%

features_name = build_features(feats_collection)

# %%
Title_collection = []

for doc in doc_path:
    Title_collection.append(Title_clean(toc_extractor(dirpath+'\\'+doc)))

#%%

ft_matrix = encode_collection(feats_collection,features_name)

# %%

target_vector = Encode_vector(doc_collection,Title_collection)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(ft_matrix,
                                                target_vector,
                                                test_size=0.1,
                                                random_state=10)

classif = RandomForestClassifier(n_estimators=100,random_state=0)
classif.fit(X_train,y_train)
# %%
classif.score(X_test,y_test)
# %%

test_mat = encode_matrix(features(doc_collection[0]),features_name)

# %%
classif.predict(test_mat)

# %%
