import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from textstat import flesch_reading_ease, gunning_fog, syllable_count
import pandas as pd
import re


def clean_text(text):

    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text)).lower()
    return cleaned_text

df = pd.read_csv('S:\Blackcoffer\Final_Submission_Shikhar_Goyal\intake.csv')
results = []
for URL in df["URL"]:
    
    response = requests.get(URL)
    html = response.content.decode('utf-8')

    html = clean_text(html)
    
    tokens = word_tokenize(html)

    word_count = len(tokens)
    complex_word_count = 0
    for word in tokens:
        if syllable_count(word) > 2:
            complex_word_count += 1

    sentences = sent_tokenize(html)
    sentence_count = len(sentences)
    total_sentence_length = sum([len(sentence) for sentence in sentences])
    avg_sentence_length = total_sentence_length / sentence_count

    percentage_complex_words = (complex_word_count / word_count) * 100
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

    personal_pronouns = sum([1 for word in tokens if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
    avg_word_length = sum([len(word) for word in tokens]) / word_count

    senIa = SentimentIntensityAnalyzer()
    sentiment = senIa.polarity_scores(html)
    positive_score = sentiment["pos"]
    negative_score = sentiment["neg"]
    polarity_score = sentiment["compound"]
    subjectivity_score = 1 - polarity_score

    results.append({
        "URL": URL,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "polarity_score": polarity_score,
        "subjectivity_score": subjectivity_score,
        "avg_sentence_length": avg_sentence_length,
        "percentage_complex_words": percentage_complex_words,
        "fog_index": fog_index,
        "avg_words_per_sentence": word_count / sentence_count,
        "complex_word_count": complex_word_count,
        "word_count": word_count,
        "syllables_per_word": sum([syllable_count(word) for word in tokens]) / word_count,
        "personal_pronouns": personal_pronouns,
        "avg_word_length": avg_word_length
    })

results_df = pd.DataFrame(results)

results_df.to_csv("Output Data Structure.csv", index=False)
