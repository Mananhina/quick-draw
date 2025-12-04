# quick-draw
Веб-сервис для распознавания скетчей, вдохновленный Quick, Draw!

![qd_ex](https://github.com/user-attachments/assets/27f44de9-f0d8-48d6-bd61-de160be5ea44)

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
