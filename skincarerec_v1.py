from os import remove
import numpy as np
import pandas as pd
import scipy as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from IPython.display import display

#remove 'stop words' from a pandas dataframe given the col name
def remove_stopwords(data, col_name):
    for i in range(len(data[col_name])):
        data[col_name].iloc[i] = str(data[col_name].iloc[i]).replace('[', '').replace(']', '').replace("'", '').replace('"', '')
    return data[col_name]

def main():
    # import the data and remove stopwords on the ingredients 
    # and run remove stop words on the ingredients col
    data = pd.read_csv('skincare_2020.csv')
    data['ingredients'] = remove_stopwords(data, 'ingredients')

    # vectorize df 
    # and make sparse matrix 
    # min_df = 0 bc we only list each ingredient once so the default of 1 would make our strings -
    tvec = TfidfVectorizer(min_df=0)
    tvec.fit(data['ingredients'])
    df2 = tvec.transform(data['ingredients'])
    sparse_df = sp.sparse.csr_matrix(df2)

    # calc distance
    # bewteen vectors from sparse matrix using cosine similiaritity 
    distances = pairwise_distances(sparse_df,metric='cosine')

    # make unique label data 
    data['unique'] = data['product_name'] + data['product_id'].astype(str)
  

    # labeled df serves as our rec
    recommender_df = pd.DataFrame(distances, 
                                columns=data['unique'], 
                                index=data['unique'])
    
    # add back product_name feature 
    df3=pd.concat([data['product_name'].reset_index(),recommender_df.reset_index()],axis=1)


    rec_input = df3[['CeraVe Facial Moisturising Lotion SPF 25 52ml2','unique','product_name']].sort_values(by='CeraVe Facial Moisturising Lotion SPF 25 52ml2').head(6)[1:]
    print(rec_input)
    
if __name__ == "__main__":
    main()