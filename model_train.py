import json
import pandas as pd
import numpy as np
import nltk
import spacy
from datetime import datetime
import gensim
import gensim.corpora as corpora
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile, datapath
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")

def main():
    raw_df = pd.read_pickle('./data.pkl')
    def clean_data(raw_df):
        '''
        Input: raw dataframe in .pkl format
        Output: clean datafram
        '''
        def convert_to_dt(s):
            try:
                return datetime.strptime(s[5:-6], '%d %b %Y %H:%M:%S')
            except Exception:
                return np.nan
        df = raw_df.filter(["Article", "Summary","keywords"])                      # Extracting only useful columns into another df   
        df.columns =["title", "body", "keywords"]                 # Renaming columns
        df.drop_duplicates(subset=list(df.columns), keep='first', inplace=True)       # dropping dupliacte rows
        df.drop_duplicates(subset=["body"], keep='first', inplace=True)               # There were repeated values in the body column, so dropped all except the one published first 
        return df
    df = clean_data(raw_df)
    l=len(df)
    News=df["body"].values
    print(df.head())  

    # Helper Functions
    def article_to_words(articles):
        #tokenizes articles; yeilds list of words for each article
        for article in articles:
            yield(gensim.utils.simple_preprocess(article, deacc=True))  # deacc=True removes punctuations
            
    def remove_stopwords(texts):
        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'say', 'also', 'would', 'may'])
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        #Applies the bigram model(defined below) on each document
        return [bigram_mod[doc] for doc in texts]

    nlp = spacy.load('en', disable=['parser', 'ner'])
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    
    def preprocess(df):
        #Input: clean dataframe
        #Output: list of articles after preprocess, bigram model  
        articles = list(df["body"])                                          # Corpus of content of all articles
        data_words = list(article_to_words(articles))                        # Each article is a list of all it's words after "simple_preprocess"
        
        common_terms = ["of", "with", "without", "and", "or", "the", "a"]
        bigram = gensim.models.Phrases(data_words, 
                                    min_count=4, 
                                    threshold=30, 
                                    common_terms=common_terms)            # higher threshold fewer phrases.
        bigram_mod = gensim.models.phrases.Phraser(bigram)                   # wrapper
        
        return data_words, bigram_mod

    data_words, bigram_mod = preprocess(df)

    data_words_lemmatized = lemmatization(data_words)
    data_words_nostops = remove_stopwords(data_words_lemmatized)
    data_final = make_bigrams(data_words_nostops)
    word_dict = corpora.Dictionary(data_final)

    # Bag of words (Document Term Frequency)
    # list of tuples for each doc- (id,frequency) of word in that document
    corpus = [word_dict.doc2bow(text) for text in data_final]
    print("data words:\n",data_words)
    print("\nbigram_mod:",bigram_mod)
    print("\ndata final:",data_final)
    print("\nword_dict:",word_dict)
    print("\n Corpus:",corpus)
    def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
        
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models`
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            print("The model training",num_topics)
            model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=word_dict,
                                            num_topics=num_topics,
                                            random_state=100,
                                            passes=20,
                                            per_word_topics=True,
                                            workers=3)
            model_list.append(model)
            coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    start=1; limit=l+1; step=1
    model_list, coherence_values = compute_coherence_values(dictionary=word_dict, corpus=corpus, texts=data_final, start=start, limit=limit, step=step)
    prob=[]
    for value in coherence_values:
        P = (0.5 * (1 + (value/2)))*100
        prob.append(P)
    print(News,"\n\t\t****The probability is***\t\t",prob)    
if __name__=='__main__':
    main()
