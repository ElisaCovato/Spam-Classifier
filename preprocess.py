"""Preprocess emails dataset"""
import re
from html import unescape
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer





class CleanEmails(BaseEstimator, TransformerMixin):
    def __init__(self, to_lowercase=True, url_to_word=True, num_to_word=True,
                 no_punc=True, stemming=True):
        self.to_lowercase = to_lowercase
        self.url_to_word = url_to_word
        self.num_to_word = num_to_word
        self.no_punc = no_punc
        self.stemming = stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_process = []
        for email in X:
            text = email_to_text(email) or ""

            if self.to_lowercase:
                text = text.lower()
            if self.url_to_word:
                text = replace_urls(text)
            if self.num_to_word:
                text = replace_numbers(text)
            if self.no_punc:
                text = remove_punctuation(text)

            # word_counts = Counter(text.split())
            # if self.stemming and stemmer is not None:
            #     stemmed_word_counts = Counter()
            #     for word, count in word_counts.items():
            #         stemmed_word = stemmer.stem(word)
            #         stemmed_word_counts[stemmed_word] += count
            #     word_counts = stemmed_word_counts
            X_process.append(text)
        return (X_process)





def html_to_text(email):
    # convert html formatted emails into plain text
    email = re.sub('<head.*?>.*?</head>', '', email, flags=re.M | re.S | re.I)
    email = re.sub('<a\s.*?>', ' HYPERLINK ', email, flags=re.M | re.S | re.I)
    email = re.sub('<.*?>', '', email, flags=re.M | re.S)
    email = re.sub(r'(\s*\n)+', '\n', email, flags=re.M | re.S)
    return unescape(email)

def email_to_text(email):
    # convert any email to text
    html = None
    for entity in email.walk():
        # some emails have multiparts, this unsure that each part is treated seprately
        entity_type = entity.get_content_type()
        if not entity_type in ("text/plain", "text/html"):
            continue
        try:
            content = entity.get_content()
        except: # sometimes this is impossible because of encoding problems
            content = str(entity.get_payload())
        if entity_type == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_text(html)

def remove_punctuation(email):
    email = re.sub(r'\W+', ' ', email, flags=re.M)
    return email

def replace_numbers(email):
    # convert all numbers into the word NUM
    email = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', email)
    return email

def replace_urls(email):
    # convert all urls into the word URL
    email = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
                       "[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", 'URL', email)
    return email



process_emails = Pipeline([
            ('clean_email', CleanEmails()),
            ('bag_of_words', CountVectorizer())
    ])