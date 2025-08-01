from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI(title="Water Quality Prediction API")
templates = Jinja2Templates(directory="templates")


from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")


# Load models
model_ph = joblib.load("models/xgboost_ph.pkl")
model_tur = joblib.load("models/xgboost_tur.pkl")
model_cond = joblib.load("models/xgboost_cond.pkl")


@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/predict/", response_class=HTMLResponse)
def predict(
    request: Request,
    pH: float = Form(...),
    Tur: float = Form(...),
    Cond: float = Form(...),
):
    input_data = np.array([[pH, Tur, Cond]])
    ph_pred = round(model_ph.predict(input_data)[0], 4)
    tur_pred = round(model_tur.predict(input_data)[0], 4)
    cond_pred = round(model_cond.predict(input_data)[0], 6)

    return templates.TemplateResponse(
        "form.html",
        {
            "request": request,
            "pH": pH,
            "Tur": Tur,
            "Cond": Cond,
            "ph_pred": ph_pred,
            "tur_pred": tur_pred,
            "cond_pred": cond_pred,
        },
    )
