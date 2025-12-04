from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import base64

from model_utils.model import MyEfficientnetModel

app = FastAPI()
model = MyEfficientnetModel()


# Класс для данных с запроса
class ImageData(BaseModel):
    image_data: str  


@app.get("/")
async def main():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/process/")
async def process_image(data: ImageData) -> JSONResponse:
    """
    Обрабатываем изображение и возвращаем предсказания
    """
    try:
        # декодируем base64 строку
        image_bytes = base64.b64decode(data.image_data)
        
        # вызываем модель для предсказания
        predictions = model.predict_image_top5(image_bytes)
        
        # формируем ответ
        response_data = {
            "predictions": predictions,  # Это должен быть словарь {1: "label1", 2: "label2", ...}
            "url": None
        }
        return JSONResponse(content=response_data)
        
    except HTTPException as he:
        raise he
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки изображения: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")
