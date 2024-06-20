import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from retinaface import RetinaFace
import timm
import torch.nn as nn
import logging
import gc  # Добавлен импорт gc для управления памятью

# Установка уровня логирования для отладки
logging.basicConfig(level=logging.DEBUG)


# Декоратор для кэширования загрузки модели
@st.cache_resource
def load_age_model():
    model_path = './AgePrediction/age_model.pth'  # Путь к модели предсказания возраста
    model = AgeModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# Определение модели для предсказания возраста
class AgeModel(nn.Module):
    def __init__(self):
        super(AgeModel, self).__init__()
        self.model = timm.create_model('hf_hub/tiny_vit_11m_224.dist_in22k_ft_in1k', pretrained=True)
        self.model.reset_classifier(num_classes=0)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.model.num_features * 7 * 7, 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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
        lower_bound = int(age) - 2
        upper_bound = int(age) + 2
        age_str = f'{lower_bound}-{upper_bound}'
        logging.debug(f'Predicted age: {age_str}')

    # Освободить неиспользуемую память после предсказания возраста
    gc.collect()

    return age_str


# Заголовок и описание приложения
st.title('Определение возраста по фото')
st.write('Загрузите изображение, и нейросеть предскажет примерный возраст.')

# Загрузка изображения
uploaded_file = st.file_uploader('Выберите изображение...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write('')
    st.write('Поиск людей и предсказание возраста...')

    # Преобразовать изображение в формат OpenCV
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Обнаружение лиц
    faces = RetinaFace.detect_faces(image_cv)
    logging.debug(f'Detected faces: {faces}')

    if faces:
        st.write(f'Обнаружено {len(faces)} человек(а).')

        # Загрузка модели предсказания возраста с кэшированием
        model = load_age_model()

        # Обработать каждое лицо и аннотировать изображение
        for key, face in faces.items():
            facial_area = face['facial_area']
            logging.debug(f'Facial area: {facial_area}')
            face_image = image_cv[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
            face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

            # Предсказать возраст
            age = predict_age(face_pil, model)

            # Нарисовать прямоугольник вокруг лица
            cv2.rectangle(image_cv, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (0, 0, 255), 2)

            # Определить размер текста
            text = f'{age}'
            font_scale = 2.0  # Исходный масштаб шрифта
            thickness = 5  # Толщина текста
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Адаптировать масштаб шрифта к размеру лица
            max_text_width = facial_area[2] - facial_area[0]  # Максимальная ширина текста, равная ширине лица
            font_scale = min(font_scale, max_text_width / text_size[0])  # Адаптировать масштаб шрифта

            # Определить положение текста
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_position = (facial_area[0] + (facial_area[2] - facial_area[0]) // 2 - text_size[0] // 2,
                             max(facial_area[1] - 10, facial_area[1] + text_size[1] + 5))  # Адаптировать высоту текста

            # Добавить текст с предсказанным возрастом над лицом
            cv2.putText(image_cv, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness,
                        cv2.LINE_AA)

            # Освободить неиспользуемую память после обработки каждого лица
            gc.collect()

        # Преобразовать обратно в формат PIL для отображения в Streamlit
        image_with_annotations = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        st.image(image_with_annotations, caption='Обнаруженные люди с возрастом', use_column_width=True)
    else:
        st.write('Люди на фото не обнаружены.')
