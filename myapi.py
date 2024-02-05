from fastapi import  FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

class model(BaseModel):
    param1:float
    param2:float
    param3:float

with open('mymodel.joblib','rb') as file:
    mymodel = joblib.load(file)

@app.post('/')
async def fuction(item:model):
    df = pd.DataFrame([item.dict()])
    print(df)
    pred = mymodel.predict(df)
    return {"prediction":pred.tolist()}