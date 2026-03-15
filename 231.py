from flask import Flask, request, render_template_string
import pandas as pd
import pickle

app = Flask(__name__)

MODEL_PATH = "models/lightGBM_model.pkl"

FEATURES = [
    "EXT_SOURCE_MEAN","EXT_SOURCE_3","EXT_SOURCE_2","EXT_SOURCE_1",
    "YEARS_EMPLOYED","ANNUITY LENGTH","CREDIT_TO_GOODS_RATIO",
    "EMPLOYMENT_STABILITY","AGE","LOG_AMT_ANNUITY","LOG_AMT_GOODS_PRICE",
    "CONVERTED_DAYS_ID_PUBLISH","CONVERTED_DAYS_LAST_PHONE_CHANGE",
    "LOG_AMT_CREDIT","ANNUITY_BY_INCOME","CONVERTED_DAYS_REGISTRATION",
    "CREDIT_BY_INCOME","GOODS_PRICE_BY_INCOME",
    "REGION_POPULATION_RELATIVE","INCOME_PER_PERSON",
]

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>AI Credit Scoring</title>

<style>
body{
font-family: Arial;
background:#f2f4f8;
margin:0;
}

.container{
max-width:900px;
margin:auto;
padding:30px;
}

.card{
background:white;
padding:30px;
border-radius:10px;
box-shadow:0 4px 20px rgba(0,0,0,0.1);
}

h1{
text-align:center;
color:#333;
}

.grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:15px;
}

input{
padding:10px;
border-radius:6px;
border:1px solid #ccc;
width:100%;
}

button{
margin-top:20px;
padding:12px;
width:100%;
background:#4CAF50;
border:none;
color:white;
font-size:16px;
border-radius:6px;
cursor:pointer;
}

button:hover{
background:#45a049;
}

.result{
margin-top:20px;
padding:15px;
border-radius:6px;
font-size:18px;
text-align:center;
}

.approve{
background:#e6f7e6;
color:#2d7a2d;
}

.reject{
background:#ffe6e6;
color:#a30000;
}
</style>
</head>

<body>

<div class="container">

<div class="card">

<h1>AI Credit Scoring</h1>

<form method="POST">

<div class="grid">

{% for f in features %}
<div>
<label>{{f}}</label>
<input type="number" step="any" name="{{f}}" required>
</div>
{% endfor %}

</div>

<button type="submit">Predict Risk</button>

</form>

{% if result %}
<div class="result {{cls}}">
<b>{{result}}</b><br>
Probability: {{prob}}
</div>
{% endif %}

</div>
</div>

</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():

    if request.method == "POST":

        data = []

        for f in FEATURES:
            data.append(float(request.form[f]))

        X = pd.DataFrame([data], columns=FEATURES)

        prob = model.predict_proba(X)[0][1]

        decision = "Reject" if prob >= 0.5 else "Approve"

        tier = "Low" if prob < 0.3 else ("Medium" if prob < 0.6 else "High")

        result = f"{decision} (Risk: {tier})"

        cls = "reject" if decision=="Reject" else "approve"

        return render_template_string(
            HTML,
            features=FEATURES,
            result=result,
            prob=f"{prob:.2%}",
            cls=cls
        )

    return render_template_string(HTML,features=FEATURES)

if __name__ == "__main__":
    app.run(debug=True)