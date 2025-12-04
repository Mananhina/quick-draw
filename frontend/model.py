import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.v2 as v2
from PIL import Image
from io import BytesIO


def my_efficientnet(num_classes):
    """Загрузка предобученной модели,
       заморозка признаковых слоев, обучение только классификатора"""

    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    # Изначально замораживаем все
    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[6].parameters():  # 6-й блок
        param.requires_grad = True

    for param in model.features[7].parameters():  # 7-й блок
        param.requires_grad = True

    # Заменяем классификатор
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    # Размораживаем классификатор
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


class MyEfficientnetModel():
    def __init__(self):
        
        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # все изображения будут масштабированы к размеру 112x112 px
        self.RESCALE_SIZE = 112
        # параметры нормировки изображений по трем каналам перед подачей в модель
        self.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
        self.NORMALIZE_STD = [0.229, 0.224, 0.225]

        self.classes = []
        with open('categories_rus.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                self.classes.append(line.strip(',\n'))

        self.classes = np.array(self.classes)

        # число классов
        self.NUM_CLASSES = len(self.classes)

        # Загружаем предобученную efficientnet
        self.model = my_efficientnet(self.NUM_CLASSES)
        self.model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
        self.model.to(self.DEVICE)


    def predict_image_top5(self, image_bytes):
        """
        Загружаем .jpg, применяем те же преобразования, что для датаcета,
        выводим top-5 предсказаний с вероятностями.
        """

        # Загружаем картинку с байт-кода
        img = Image.open(BytesIO(image_bytes)).convert("L")  # grayscale
        img = img.resize((self.RESCALE_SIZE, self.RESCALE_SIZE), Image.LANCZOS)

        # Обработка цветов как в трен. датасете
        arr = 1 - (np.array(img).astype(np.float32) / 255.0)

        kernel = np.ones((2, 2), np.uint8)
        arr = cv2.dilate(arr, kernel, iterations=1)

        # сглаживание
        arr = cv2.GaussianBlur(arr, (3, 3), 0.7)

        # нормализация диапазона
        arr = np.clip(arr, 0.0, 1.0)

        # дублируем три канала цветов
        arr = np.stack([arr, arr, arr], axis=0)
        tensor = torch.from_numpy(arr)

        # трансформ как на трен. датасете
        transform = v2.Compose([
            v2.Resize(self.RESCALE_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(self.NORMALIZE_MEAN, self.NORMALIZE_STD),
        ])

        tensor = transform(tensor)

        # добавлем размерность от батча
        tensor = tensor.unsqueeze(0).to(self.DEVICE)

        # инференс модели
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        # получаем топ-5 классов
        top5_prob, top5_idx = torch.topk(probs, k=5)

        # обработка типов
        top5_prob = top5_prob.cpu().numpy()
        top5_prob = np.round(top5_prob*100, 2).astype(int)

        top5_idx = top5_idx.cpu().numpy().tolist()
        top5_classes = self.classes[top5_idx]

        # форматируем выход
        top5 = dict(zip(top5_classes.tolist(), top5_prob.tolist()))

        return top5
