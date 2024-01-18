import pandas as pd 
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def get_data():
    df = pd.read_csv('./coffee_analysis.csv')


    #loc_country, origin_1, origin_2, roast, roaster
    to_binary_column_names = ['loc_country', 'origin_1', 'origin_2', 'roast', 'roaster']

    for col in to_binary_column_names:
        possible_values = df[col].dropna().explode().unique()#.str.split(',').apply(lambda x: [i.strip() for i in x]).explode().unique()
        microdf = pd.DataFrame()
        microdf_binary_values = [df[col].str.contains(re.escape(val)).fillna(0).astype(int) for val in possible_values]    
        possible_values = [f'{x}_{col}' for x in possible_values]

        microdf = pd.DataFrame(dict(zip(possible_values, microdf_binary_values)))
        df = pd.concat([df, microdf], axis=1)

    df = df.drop(to_binary_column_names, axis=1)

    #drop names and review date
    df = df.drop(['name', 'review_date'], axis=1)


    #td_idf_vectorizing descriptions

    descriptions = df[['desc_1', 'desc_2', 'desc_3']].apply(lambda x: '-'.join(x.astype(str)), axis=1).to_numpy()
    # print(descriptions)
    df = df.drop(['desc_1', 'desc_2', 'desc_3'], axis=1)
    tfidf = TfidfVectorizer(
        ngram_range=(1,3),
        stop_words = 'english',
        max_features=5000
    )
    tfidf_matrix = tfidf.fit_transform(descriptions)
    tfidf_features = tfidf.get_feature_names_out()
    df_tfidf_descriptions = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_features)
    df = pd.concat([df, df_tfidf_descriptions], axis=1)
    print(f'df_shape: {df.shape}')
    return df
