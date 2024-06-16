import os
import re
import unicodedata
import nltk
import spacy
import torch
import joblib
import streamlit as st
import numpy as np
import subliminal
import chardet
import subprocess
import sys
import gc
import logging

from transformers import LongformerTokenizer, LongformerModel
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from subliminal import download_best_subtitles, save_subtitles, Video
from babelfish import Language
from subliminal.cache import region
from pydantic import BaseModel, Field
from typing import List, Optional
from opensubtitlescom import OpenSubtitles


class TokenPatternString(BaseModel):
    REGEX: Optional[str] = Field(None, alias='regex')
    INCLUDED_VALUES: Optional[List[str]] = Field(None, alias='included_values')
    NOT_IN: Optional[List[str]] = Field(None, alias='not_in')

    class Config:
        populate_by_name = True

# Установите устройство (GPU или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@st.cache_resource
def load_nltk_resources():
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    return stop_words

stop_words = load_nltk_resources()

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

@st.cache_resource
def load_longformer_model():
    model_name = 'allenai/longformer-base-4096'
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    model = LongformerModel.from_pretrained(model_name).to(device)
    return tokenizer, model

tokenizer, longformer_model = load_longformer_model()

@st.cache_resource
def load_model_and_thresholds():
    model = joblib.load('trained_voting_classifier.pkl')
    thresholds = np.load('best_thresholds.npy')
    return model, thresholds

voting_clf, best_thresholds = load_model_and_thresholds()

# Функции для очистки и обработки текста
def is_word_in_wordnet(word):
    return bool(wordnet.synsets(word))

def clean_text(input_text):
    clean_text = re.sub('<[^<]+?>|http\S+', '', input_text)
    clean_text = re.sub('\s+', ' ', input_text.lower())
    clean_text = unicodedata.normalize('NFKD', clean_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    clean_text = re.sub('\d', '', clean_text)
    tokens = [token for token in word_tokenize(clean_text) if token not in stop_words and is_word_in_wordnet(token)]
    doc = nlp(' '.join(tokens))
    clean_text = ' '.join([token.lemma_ for token in doc if not token.is_punct])
    return clean_text

def get_longformer_embeddings(text, max_length=4096):
    encoded_input = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    with torch.no_grad():
        outputs = longformer_model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_state = outputs.last_hidden_state
    mean_hidden_state = torch.mean(last_hidden_state, dim=1)
    return mean_hidden_state.cpu().numpy()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USERNAME = "YOUR_USER_NAME"
PASSWORD = "YOUR_PASSWORD"
API_KEY = "YOUR_API"

def download_subtitles(movie_name):
    logging.info(f"Поиск субтитров для: {movie_name} с использованием Subliminal")

    video = Video.fromname(movie_name)
    subtitles = download_best_subtitles([video], {Language('eng')}, providers=['podnapisi'])

    if subtitles[video]:
        subtitle = subtitles[video][0]
        subtitle_path = f'{movie_name}.srt'
        save_subtitles(video, [subtitle], single=True)
        logging.info(f'Субтитры найдены с помощью Subliminal для: {movie_name}')

        with open(subtitle_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        with open(subtitle_path, 'r', encoding=encoding) as f:
            subtitles_text = f.read()

        os.remove(subtitle_path)
        return subtitles_text
    else:
        logging.info(f'Субтитры не найдены с помощью Subliminal для: {movie_name}. Поиск с помощью OpenSubtitles.')
        return search_with_opensubtitles(movie_name)

def search_with_opensubtitles(movie_name):
    # Инициализация клиента OpenSubtitles
    subtitles_client = OpenSubtitles('Streamlit/1.35.0', API_KEY)

    # Логин (получение токена аутентификации)
    subtitles_client.login(USERNAME, PASSWORD)

    # Поиск субтитров
    response = subtitles_client.search(query=movie_name, languages="en")

    if response.data:
        # Получение субтитров из первого ответа
        srt = subtitles_client.download_and_parse(response.data[0])
        
        # Определение пути к субтитрам
        subtitle_path = f'{movie_name}.srt'
        
        with open(subtitle_path, 'w', encoding='utf-8') as f:
            for subtitle in srt:
                f.write(str(subtitle))
        
        # Определение кодировки
        with open(subtitle_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        # Чтение субтитров с правильной кодировкой
        with open(subtitle_path, 'r', encoding=encoding) as f:
            subtitles_text = f.read()
        
        # Удаление временного файла с субтитрами
        os.remove(subtitle_path)
        
        logging.info(f"Субтитры найдены с помощью OpenSubtitles для: {movie_name}")
        return subtitles_text
    else:
        logging.error(f"Субтитры не найдены с помощью OpenSubtitles для: {movie_name}")
        return None
st.title('Оптимальные фильмы для вашего уровня английского')

st.markdown("""
<style>
    .main {
        background-color: #121212;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #333333;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stAlert {
        background-color: #333333;
        color: white;
    }
    .stCodeBlock {
        background-color: #333333;
    }
</style>
""", unsafe_allow_html=True)

movie_name = st.text_input('Введите официальное англоязычное название фильма (скопируйте с Кинопоиска или IMDb):')

if movie_name:
    subtitles = download_subtitles(movie_name)
    if subtitles:
        st.success(f'Субтитры загружены для: {movie_name}')
        cleaned_subtitles = clean_text(subtitles)
        embeddings = get_longformer_embeddings(cleaned_subtitles)
        embeddings = embeddings.reshape(1, -1)
        y_proba = voting_clf.predict_proba(embeddings)
        num_classes = y_proba.shape[1]

        y_pred_final = np.zeros_like(y_proba)
        for i in range(num_classes):
            y_pred_final[:, i] = y_proba[:, i] >= best_thresholds[i]
        y_pred_class = np.argmax(y_pred_final, axis=1)

        level_mapping = {0: 'A2', 1: 'B1', 2: 'B2', 3: 'C1'}
        recommended_level = level_mapping.get(y_pred_class[0], 'Unknown')

        st.write(f"Рекомендуется к просмотру с уровнем знания английского языка: {recommended_level}")

    else:
        st.error(f'Субтитры не найдены для: {movie_name},проверьте название, попробуйте другой фильм или зайдите позже - превышен лиимит')

clear_memory()
