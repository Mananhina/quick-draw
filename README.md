# quick-draw
Веб-сервис для распознавания скетчей, вдохновленный Quick, Draw!

![qd_ex](https://github.com/user-attachments/assets/27f44de9-f0d8-48d6-bd61-de160be5ea44)

В качестве модели классификации использована EfficientNet-B0 с дообучением последних слоев. Метрика F1-macro на тестовой выборке составила 92%. В качестве данных для обучения использовн избранный набор 60 классов из открытого датасета скетчей [Quick, Draw! The Data](https://quickdraw.withgoogle.com/data). Backend реализован на FastAPI, интерфейс — на HTML/JS с Canvas API.

## Запуск проекта
### 1. Через pip
```
# Клонируем репозиторий
git clone https://github.com/Mananhina/quick-draw.git
cd quick-draw

# Устанавливаем зависимости
pip install -r requirements.txt

# Запускаем сервер FastAPI
uvicorn main:app --port 8000
```
После запуска сервер доступен по адресу http://localhost:8000

### 2. Через Docker со сборкой образа
```
# Клонируем репозиторий
git clone https://github.com/Mananhina/quick-draw.git
cd quick-draw

# Собираем образ
docker build -t quick-draw-app .

# Запускаем контейнер
docker run -d -p 8000:8000 --name quick-draw-container quick-draw-app
```

После запуска сервер доступен по адресу http://localhost:8000

### 3. Через Docker Hub
```
# Скачиваем образ
docker pull liza0407/quick-draw-app-cpu:v1

# Запускаем контейнер
docker run -d -p 8000:8000 --name quick-draw-container liza0407/quick-draw-app-cpu:v1
```

После запуска сервер доступен по адресу http://localhost:8000

## Технологии
- Python 3.10
- FastAPI
- PyTorch
- Docker
- HTML/JS Canvas
