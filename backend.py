# backend.py
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import plotly
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def make_regression_plot(df, x_col, y_col, degree=1):
    df = df[[x_col, y_col]].dropna()
    X = df[x_col].values.flatten()
    y = df[y_col].values.flatten()

    # Polynomial regression
    X_poly = np.vander(X.flatten(), N=degree+1, increasing=True)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)

    X_list = [float(val) for val in X]
    y_list = [float(val) for val in y]
    y_pred_list = [float(val) for val in y_pred]
    # Build Plotly figure
    '''
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode="markers", name="Data", marker=dict(color="blue", size=8)))
    fig.add_trace(go.Scatter(x=X, y=y_pred, mode="lines", name=f"Degree {degree} fit", line=dict(color="red")))
    fig.update_layout(title=f"Regression of {y_col} vs {x_col}",
                      xaxis_title=x_col,
                      yaxis_title=y_col,
                      template="plotly_white")
    '''
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_list, y=y_list, mode="markers", name="Data", marker=dict(color="blue", size=8)))
    fig.add_trace(go.Scatter(x=X_list, y=y_pred_list, mode="lines", name=f"Degree {degree} fit", line=dict(color="red")))
    fig.update_layout(title=f"Regression of {y_col} vs {x_col}",
                  xaxis_title=x_col,
                  yaxis_title=y_col,
                  template="plotly_white")
    # Debug prints
    print("Columns in file:", df.columns.tolist())
    print("User selected:", x_col, y_col)
    print("Data head:\n", df.head())
    #
    
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # Convert figure to JSON (so frontend can render it)
   # return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder), model.coef_.tolist(), model.intercept_
    return fig_json, model.coef_.tolist(), model.intercept_

# backend.py
"""
@app.route("/regression", methods=["POST"])
def regression():
    file = request.files["file"]
    df = pd.read_excel(file)

    x_col = request.form.get("x_col")
    y_col = request.form.get("y_col")

    X = df[[x_col]].values
    y = df[y_col].values

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    print("➡️ Params from frontend:", x_col, y_col)
    print("📊 Columns in dataframe:", df.columns.tolist())

    # Convert to lists for JSON
    response = {
        "x": X.flatten().tolist(),
        "y": y.tolist(), 
        "slope": slope, 
        "intercept": intercept,
    }
    return jsonify(response)
"""
@app.route("/regression", methods=["POST"])
def regression():
    print("✅ Received request at /regression")

    if "file" not in request.files:
        print("❌ No file in request.files")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    print("📂 File received:", file.filename)

    x_col = request.form.get("x_col")
    y_col = request.form.get("y_col")
    degree = int(request.form.get("degree", 1))
    print("➡️ Params:", x_col, y_col, degree)

    # Load into pandas
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file)
        print("✅ CSV loaded, shape:", df.shape)
    else:
        df = pd.read_excel(file)
        print("✅ Excel loaded, shape:", df.shape)

    # Now call regression function
    fig_json, coeffs, intercept = make_regression_plot(df, x_col, y_col, degree)
    print("✅ make_regression_plot executed successfully")

    return jsonify({
        "figure": json.loads(fig_json),
        "coefficients": coeffs,
        "intercept": intercept
    })


if __name__ == "__main__":
    app.run(debug=True)

#python backend.py
#python -m http.server 8000
