import gc
import cv2
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import RetinaFace
import timm
import torch.nn as nn
import logging

# Установка уровня логирования для отладки
logging.basicConfig(level=logging.DEBUG)


# Функция для очистки памяти
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@st.cache_resource
def load_age_model():
    class AgeModel(nn.Module):
        def __init__(self):
            super(AgeModel, self).__init__()
            model_path = './AgePrediction/age_model.pth'  # Путь к модели предсказания возраста
            self.model = timm.create_model('hf_hub/tiny_vit_11m_224.dist_in22k_ft_in1k', pretrained=True)
            self.model.reset_classifier(num_classes=0)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(self.model.num_features * 7 * 7, 1)
            self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        def forward(self, x):
            x = self.model.forward_features(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = AgeModel()
    model.eval()
    return model


model = load_age_model()

# Трансформации изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Функция для предсказания возраста
def predict_age(image, model):
    image = transform(image).unsqueeze(0)
    logging.debug(f'Transformed image shape: {image.shape}')
    with torch.no_grad():
        age = model(image).item()
        # Увеличить диапазон предсказания для крайних возрастных групп
        if age < 20:
            lower_bound, upper_bound = int(age) - 3, int(age) + 3
        elif age > 50:
            lower_bound, upper_bound = int(age) - 5, int(age) + 5
        else:
            lower_bound, upper_bound = int(age) - 2, int(age) + 2
        age_str = f'{lower_bound}-{upper_bound}'
        logging.debug(f'Predicted age: {age_str}')
    return age_str


# Функция для предсказания возрастов для нескольких лиц
def predict_ages(faces_pil, model):
    face_tensors = [transform(face).unsqueeze(0) for face in faces_pil]
    batch_tensor = torch.cat(face_tensors, dim=0)
    logging.debug(f'Batch tensor shape: {batch_tensor.shape}')
    with torch.no_grad():
        ages = model(batch_tensor).view(-1).tolist()
        age_ranges = []
        for age in ages:
            if age < 20:
                lower_bound, upper_bound = int(age) - 3, int(age) + 3
            elif age > 50:
                lower_bound, upper_bound = int(age) - 5, int(age) + 5
            else:
                lower_bound, upper_bound = int(age) - 2, int(age) + 2
            age_ranges.append((lower_bound, upper_bound))
        logging.debug(f'Predicted age ranges: {age_ranges}')
    return age_ranges


# Заголовок и описание приложения
st.title('Определение возраста по фото')
st.write('Загрузите изображение, и нейросеть предскажет примерный возраст.')

# Загрузка изображения
uploaded_file = st.file_uploader('Выберите изображение...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write('')
    st.write('Поиск людей и предсказание возраста...')

    # Преобразование изображения в формат numpy
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Обнаружение лиц
    faces = RetinaFace.detect_faces(image_cv)
    logging.debug(f'Detected faces: {faces}')

    if faces:
        st.write(f'Обнаружено {len(faces)} человек(а).')

        # Список лиц в формате PIL
        faces_pil = [Image.fromarray(image_np[face['facial_area'][1]:face['facial_area'][3],
                                     face['facial_area'][0]:face['facial_area'][2]]) for face in faces.values()]

        # Предсказать возраст для всех лиц
        age_ranges = predict_ages(faces_pil, model)

        # Обработать каждое лицо и аннотировать изображение
        for i, (key, face) in enumerate(faces.items()):
            facial_area = face['facial_area']
            age_range = age_ranges[i]
            text = f'{age_range[0]}-{age_range[1]}'
            logging.debug(f'Facial area: {facial_area}')

            # Нарисовать прямоугольник вокруг лица
            draw = ImageDraw.Draw(image)
            draw.rectangle([facial_area[0], facial_area[1], facial_area[2], facial_area[3]], outline="red", width=2)

            # Добавить текст с предсказанным возрастом над лицом
            font = ImageFont.truetype("arial.ttf", size=15)
            text_width, text_height = draw.textsize(text, font=font)
            text_position = (facial_area[0] + (facial_area[2] - facial_area[0]) // 2 - text_width // 2,
                             max(facial_area[1] - text_height - 5, 0))
            draw.text(text_position, text, fill="red", font=font)

        st.image(image, caption='Обнаруженные люди с возрастом', use_column_width=True)
    else:
        st.write('Люди на фото не обнаружены.')

clear_memory()
