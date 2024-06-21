import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from retinaface import RetinaFace
import timm
import cv2
import torch.nn as nn
import logging
import gc

logging.basicConfig(level=logging.DEBUG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@st.cache_resource
def load_age_model():
    class AgeModel(nn.Module):
        def __init__(self):
            super(AgeModel, self).__init__()
            model_path = './AgePrediction/age_model.pth'
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

    model = AgeModel().to(device)
    model.eval()
    return model

model = load_age_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_age(image, model):
    image = transform(image).unsqueeze(0).to(device)
    logging.debug(f'Transformed image shape: {image.shape}')
    with torch.no_grad():
        age = model(image).item()
        if 20 <= age <= 60:
            lower_bound = int(age) - 2
            upper_bound = int(age) + 2
        else:
            lower_bound = int(age) - 3
            upper_bound = int(age) + 3
        age_str = f'{lower_bound}-{upper_bound}'
        logging.debug(f'Predicted age: {age_str}')
    return age_str

st.title('Определение возраста по фото')
st.write('Загрузите изображение, и нейросеть предскажет примерный возраст.')

st.markdown("""
<style>
.small-font {
    font-size: 12px;
}
</style>
<div class="small-font">
* Точность предсказания зависит от качества загруженного изображения и четкости лица на фотографии.<br>
* Из-за особенностей работы модели, результаты могут быть менее точными для детей и пожилых людей.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader('Выберите изображение...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write('')
    st.write('Поиск людей и предсказание возраста...')

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    faces = RetinaFace.detect_faces(image_cv)
    logging.debug(f'Detected faces: {faces}')

    if faces:
        st.write(f'Обнаружено {len(faces)} человек(а).')

        for key, face in faces.items():
            facial_area = face['facial_area']
            logging.debug(f'Facial area: {facial_area}')
            face_image = image_cv[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
            face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

            age = predict_age(face_pil, model)

            cv2.rectangle(image_cv, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (0, 0, 255), 2)

            text = f'{age}'
            face_width = facial_area[2] - facial_area[0]
            face_height = facial_area[3] - facial_area[1]
            font_scale = min(face_width / 200, face_height / 60)
            font_scale = max(font_scale, 0.5)
            thickness = max(int(font_scale), 1)
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_position = (facial_area[0] + (face_width - text_size[0]) // 2,
                             facial_area[1] - 10 if facial_area[1] - 10 > text_size[1] else facial_area[3] + text_size[1] + 10)

            text_background_position = (facial_area[0], text_position[1] - text_size[1] - 5)
            text_background_end = (facial_area[2], text_position[1] + 5)
            if text_position[1] - text_size[1] - 5 > 0:
                cv2.rectangle(image_cv, text_background_position, text_background_end, (255, 255, 255), cv2.FILLED)
            else:
                text_background_position = (facial_area[0], facial_area[3] + 5)
                text_background_end = (facial_area[2], facial_area[3] + 5 + text_size[1] + 5)
                cv2.rectangle(image_cv, text_background_position, text_background_end, (255, 255, 255), cv2.FILLED)
                text_position = (facial_area[0] + (face_width - text_size[0]) // 2, facial_area[3] + text_size[1] + 10)

            cv2.putText(image_cv, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        image_with_annotations = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        st.image(image_with_annotations, caption='Обнаруженные люди с возрастом', use_column_width=True)
    else:
        st.write('Люди на фото не обнаружены.')

clear_memory()
