# backend.py
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import plotly
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import json
from flask_cors import CORS
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)
CORS(app)


def load_df(file):
    name = (file.filename or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    raise ValueError("Unsupported file type. Upload .csv or .xlsx/.xls")


def make_regression_plot(df, x_col, y_col, degree=1):
    df = df[[x_col, y_col]].dropna()
    X = df[x_col].values.flatten()
    y = df[y_col].values.flatten()

    # Polynomial regression
    X_poly = np.vander(X.flatten(), N=degree+1, increasing=True)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)

    X_list = [float(val) for val in X]
    X_list = sorted(X_list)
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
    fig.update_layout(
        title=f"Regression of {y_col} vs {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white",
        xaxis=dict(range=[0, max(X_list)]),
        yaxis=dict(range=[0, max(y_list)])
    )

    fig.add_trace(go.Scatter(x=X_list, y=y_list, mode="markers", name="Data", marker=dict(color="blue", size=5)))
    fig.add_trace(go.Scatter(x=X_list, y=y_pred_list, mode="lines", name=f"Degree {degree} fit", line=dict(color="red")))

    # Debug prints
    print("Columns in file:", df.columns.tolist())
    print("User selected:", x_col, y_col)
    print("Data head:\n", df.head())
    #
    
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # Convert figure to JSON (so frontend can render it)
   # return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder), model.coef_.tolist(), model.intercept_
    return fig_json, model.coef_.tolist(), model.intercept_
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
    print(" Params from frontend:", x_col, y_col)
    print(" Columns in dataframe:", df.columns.tolist())

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
    print(" Received request at /regression")

    if "file" not in request.files:
        print(" No file in request.files")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    print(" File received:", file.filename)

    x_col = request.form.get("x_col")
    y_col = request.form.get("y_col")
    degree = int(request.form.get("degree", 1))
    if not x_col or not y_col:
        return jsonify({"error": "Missing x_col or y_col"}), 400
    if degree < 1:
        return jsonify({"error": "Degree must be >= 1"}), 400
    print(" Params:", x_col, y_col, degree)

    df=load_df(file)

    sub = df[[x_col, y_col]].dropna().copy()
    x = pd.to_numeric(sub[x_col], errors="coerce")
    y = pd.to_numeric(sub[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask].to_numpy(dtype=float)
    y = y[mask].to_numpy(dtype=float)

    if x.size < degree + 1:
        return jsonify({"error": f"Not enough data points for degree {degree}"}), 400

    # Now call regression function
    X = x.reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)

    # include_bias=True already adds the constant 1 column -> no intercept
    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)

    # Dense x-grid for smooth curve
    x_curve = np.linspace(x.min(), x.max(), 600).reshape(-1, 1)
    x_curve_poly = poly.transform(x_curve)
    y_curve = model.predict(x_curve_poly)

    return jsonify({
        "x_data": x.tolist(),
        "y_data": y.tolist(),
        "x_curve": x_curve.squeeze().tolist(),
        "y_curve": y_curve.tolist(),
        "coefficients": model.coef_.tolist(),   # [a0, a1, a2, ...] for 1, x, x^2, ...
        "degree": degree
    })


if __name__ == "__main__":
    app.run(debug=True)

#python backend.py
#python -m http.server 8000
