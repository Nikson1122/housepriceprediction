from flask import Flask, render_template, request
import numpy as np
import os
import joblib 

app = Flask(__name__)

# Load your trained model
MODEL_PATH = os.path.join("notebooks", "house_price_model.pkl")
with open(MODEL_PATH, "rb") as f:
  model = joblib.load(MODEL_PATH) 

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        try:
            # Collect values from form
            features = [
                float(request.form["crim"]),
                float(request.form["zn"]),
                float(request.form["indus"]),
                int(request.form["chas"]),
                float(request.form["nox"]),
                float(request.form["rm"]),
                float(request.form["age"]),
                float(request.form["dis"]),
                int(request.form["rad"]),
                float(request.form["tax"]),
                float(request.form["ptratio"]),
                float(request.form["b"]),
                float(request.form["lstat"]),
            ]
            features = np.array([features])
            prediction = model.predict(features)[0]
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("predict.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
