from fastapi import FastAPI
import mlflow

mlflow.set_tracking_uri("http://192.168.1.4:5000")
app = FastAPI()
model =  mlflow.xgboost.load_model("models:/xgboost/3")
print("loaded")
@app.get("/")
def read_root():
    return {"root":"reached"}

@app.get("/model")
def get_item():
    return model.get_booster().save_config()