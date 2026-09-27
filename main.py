import mlflow
from sklearn.linear_model import LogisticRegression

mlflow.set_tracking_uri("http://192.168.1.4:5000")
mlflow.set_experiment("exp test")
def transform(x):
    return x/10
class Model(mlflow.pyfunc.PythonModel):
    def predict(self,x):
        return transform(x)

# with mlflow.start_run() as run:
#     model = Model()
#     model_uri = mlflow.pyfunc.log_model(python_model=model,artifact_path="test_model")
#     mlflow.register_model(model_uri.model_uri,"test")
model = mlflow.pyfunc.load_model("models:/test/1")
print(model.predict(2))