# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 20:45:10 2016

@author: DIP
"""

from .contractions import CONTRACTION_MAP
from re import compile, IGNORECASE, DOTALL, sub, escape
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopword_list = stopwords.words('english')  # nltk.download()
wnl = WordNetLemmatizer()


def tokenize_text(text):
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def expand_contractions(text, contraction_mapping):
    
    contractions_pattern = compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=IGNORECASE | DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = sub("'", "", expanded_text)
    return expanded_text


# Annotate text tokens with POS tags
def pos_tag_text(text):
    
    def penn_to_wn_tags(postag):
        if postag.startswith('J'):
            return wn.ADJ
        elif postag.startswith('V'):
            return wn.VERB
        elif postag.startswith('N'):
            return wn.NOUN
        elif postag.startswith('R'):
            return wn.ADV
        else:
            return None
    
    tagged_text = pos_tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(postag))
                         for word, postag in
                         tagged_text]
    return tagged_lower_text


# lemmatize text based on POS tags    
def lemmatize_text(text):

    text = word_tokenize(text)
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wnl.lemmatize(word, postag) if postag
                         else word
                         for word, postag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text
    

def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = compile('[{}]'.format(escape(punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
    
    
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def normalize_corpus(corpus, tokenize=False):
    
    normalized_corpus = []    
    for text in corpus:
        text = expand_contractions(text, CONTRACTION_MAP)
        text = lemmatize_text(text)
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)

    return normalized_corpus
