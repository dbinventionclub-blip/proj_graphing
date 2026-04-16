# backend.py
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import plotly
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import json
from flask_cors import CORS
from sklearn.preprocessing import PolynomialFeatures
import os

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def root():
    return send_from_directory(os.path.dirname(__file__), "index.html")

@app.route("/styles.css", methods=["GET"])
def styles():
    return send_from_directory(os.path.dirname(__file__), "styles.css")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


def load_df(file):
    name = (file.filename or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    raise ValueError("Unsupported file type. Upload .csv or .xlsx/.xls")


def require_uploaded_dataframe():
    if "file" not in request.files:
        raise ValueError("No file uploaded")

    file = request.files["file"]
    if not file or not file.filename:
        raise ValueError("No file uploaded")
    return load_df(file)


def extract_numeric_xy(df, x_col, y_col):
    missing_columns = [col for col in (x_col, y_col) if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Column not found: {', '.join(missing_columns)}")

    sub = df[[x_col, y_col]].dropna().copy()
    x = pd.to_numeric(sub[x_col], errors="coerce")
    y = pd.to_numeric(sub[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask].to_numpy(dtype=float)
    y = y[mask].to_numpy(dtype=float)

    if x.size == 0:
        raise ValueError("No numeric rows were found for the selected columns")

    return x, y


def transform_identity(x, y):
    return x, y


def transform_x_power(x, y, power):
    return np.power(x, power), y


def transform_inverse_x(x, y, power):
    if np.any(np.isclose(x, 0.0)):
        raise ValueError("This transform requires x != 0 for every point")
    return np.power(x, -power), y


def transform_log_y(x, y):
    if np.any(y <= 0):
        raise ValueError("This transform requires y > 0 for every point")
    return x, np.log(y)


def transform_log_log(x, y):
    if np.any(x <= 0):
        raise ValueError("This transform requires x > 0 for every point")
    if np.any(y <= 0):
        raise ValueError("This transform requires y > 0 for every point")
    return np.log(x), np.log(y)


LINEARIZATION_TRANSFORMS = {
    "y_vs_x": {
        "label": "y vs x",
        "x_axis_label": lambda x_col, y_col: x_col,
        "y_axis_label": lambda x_col, y_col: y_col,
        "transform": transform_identity,
    },
    "y_vs_x2": {
        "label": "y vs x^2",
        "x_axis_label": lambda x_col, y_col: f"{x_col}^2",
        "y_axis_label": lambda x_col, y_col: y_col,
        "transform": lambda x, y: transform_x_power(x, y, 2),
    },
    "y_vs_x3": {
        "label": "y vs x^3",
        "x_axis_label": lambda x_col, y_col: f"{x_col}^3",
        "y_axis_label": lambda x_col, y_col: y_col,
        "transform": lambda x, y: transform_x_power(x, y, 3),
    },
    "y_vs_1_over_x": {
        "label": "y vs 1/x",
        "x_axis_label": lambda x_col, y_col: f"1/{x_col}",
        "y_axis_label": lambda x_col, y_col: y_col,
        "transform": lambda x, y: transform_inverse_x(x, y, 1),
    },
    "y_vs_1_over_x2": {
        "label": "y vs 1/x^2",
        "x_axis_label": lambda x_col, y_col: f"1/({x_col}^2)",
        "y_axis_label": lambda x_col, y_col: y_col,
        "transform": lambda x, y: transform_inverse_x(x, y, 2),
    },
    "ln_y_vs_x": {
        "label": "ln(y) vs x",
        "x_axis_label": lambda x_col, y_col: x_col,
        "y_axis_label": lambda x_col, y_col: f"ln({y_col})",
        "transform": transform_log_y,
    },
    "ln_y_vs_ln_x": {
        "label": "ln(y) vs ln(x)",
        "x_axis_label": lambda x_col, y_col: f"ln({x_col})",
        "y_axis_label": lambda x_col, y_col: f"ln({y_col})",
        "transform": transform_log_log,
    },
}


def format_linear_equation(y_label, x_label, slope, intercept):
    slope_str = f"{slope:.6g}"
    intercept_abs_str = f"{abs(intercept):.6g}"
    if np.isclose(intercept, 0.0):
        return f"{y_label} = {slope_str} * ({x_label})"

    sign = "+" if intercept >= 0 else "-"
    return f"{y_label} = {slope_str} * ({x_label}) {sign} {intercept_abs_str}"


def build_linearization_response(x, y, x_col, y_col, transform_key, selection_mode):
    config = LINEARIZATION_TRANSFORMS[transform_key]
    x_transformed, y_transformed = config["transform"](x, y)

    if x_transformed.size < 2:
        raise ValueError("At least two valid points are required to linearize the data")

    X_model = x_transformed.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_model, y_transformed)

    r_squared = float(model.score(X_model, y_transformed))
    x_curve = np.linspace(x_transformed.min(), x_transformed.max(), 600)
    y_curve = model.predict(x_curve.reshape(-1, 1))

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    x_axis_label = config["x_axis_label"](x_col, y_col)
    y_axis_label = config["y_axis_label"](x_col, y_col)

    return {
        "selection_mode": selection_mode,
        "transform_key": transform_key,
        "transform_label": config["label"],
        "x_axis_label": x_axis_label,
        "y_axis_label": y_axis_label,
        "x_data": x_transformed.tolist(),
        "y_data": y_transformed.tolist(),
        "x_curve": x_curve.tolist(),
        "y_curve": y_curve.tolist(),
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "equation_text": format_linear_equation(y_axis_label, x_axis_label, slope, intercept),
    }


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
        title=f"{y_col} vs {x_col}",
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

    x_col = request.form.get("x_col")
    y_col = request.form.get("y_col")
    degree = int(request.form.get("degree", 1))
    model_type = request.form.get("model_type", "polynomial").strip().lower()
    allowed_models = {"polynomial", "inverse", "exponential"}
    if not x_col or not y_col:
        return jsonify({"error": "Missing x_col or y_col"}), 400
    if degree < 1:
        return jsonify({"error": "Degree must be >= 1"}), 400
    if model_type not in allowed_models:
        return jsonify({"error": "model_type must be polynomial, inverse, or exponential"}), 400
    print(" Params:", x_col, y_col, degree, model_type)

    try:
        df = require_uploaded_dataframe()
        x, y = extract_numeric_xy(df, x_col, y_col)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if model_type == "polynomial":
        if x.size < degree + 1:
            return jsonify({"error": f"Not enough data points for degree {degree}"}), 400

        X = x.reshape(-1, 1)
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(X)

        model = LinearRegression(fit_intercept=False)
        model.fit(X_poly, y)

        x_curve = np.linspace(x.min(), x.max(), 600).reshape(-1, 1)
        x_curve_poly = poly.transform(x_curve)
        y_curve = model.predict(x_curve_poly)

        return jsonify({
            "x_data": x.tolist(),
            "y_data": y.tolist(),
            "x_curve": x_curve.squeeze().tolist(),
            "y_curve": y_curve.tolist(),
            "coefficients": model.coef_.tolist(),   # [a0, a1, a2, ...] for 1, x, x^2, ...
            "degree": degree,
            "model_type": model_type
        })

    if model_type == "inverse":
        if x.size < degree + 1:
            return jsonify({"error": f"Not enough data points for n = {degree}"}), 400
        if np.any(np.isclose(x, 0.0)):
            return jsonify({"error": "Inverse model requires x != 0 for all points"}), 400

        X_inv = np.column_stack([np.ones_like(x)] + [x ** (-k) for k in range(1, degree + 1)])
        model = LinearRegression(fit_intercept=False)
        model.fit(X_inv, y)

        x_curve = np.linspace(x.min(), x.max(), 600)
        x_curve = x_curve[~np.isclose(x_curve, 0.0)]
        X_curve_inv = np.column_stack(
            [np.ones_like(x_curve)] + [x_curve ** (-k) for k in range(1, degree + 1)]
        )
        y_curve = model.predict(X_curve_inv)

        return jsonify({
            "x_data": x.tolist(),
            "y_data": y.tolist(),
            "x_curve": x_curve.tolist(),
            "y_curve": y_curve.tolist(),
            "coefficients": model.coef_.tolist(),   # [a0, a1, ...] for 1, x^-1, x^-2, ...
            "degree": degree,
            "model_type": model_type
        })

    # model_type == "exponential"
    if x.size < 2:
        return jsonify({"error": "Not enough data points for exponential fit"}), 400
    if np.any(y <= 0):
        return jsonify({"error": "Exponential model requires y > 0 for all points"}), 400

    X = x.reshape(-1, 1)
    log_y = np.log(y)
    model = LinearRegression()
    model.fit(X, log_y)

    b = float(model.coef_[0])
    a = float(np.exp(model.intercept_))

    x_curve = np.linspace(x.min(), x.max(), 600)
    y_curve = a * np.exp(b * x_curve)

    return jsonify({
        "x_data": x.tolist(),
        "y_data": y.tolist(),
        "x_curve": x_curve.tolist(),
        "y_curve": y_curve.tolist(),
        "model_type": model_type,
        "exp_params": {"a": a, "b": b}
    })


@app.route("/linearize", methods=["POST"])
def linearize():
    x_col = request.form.get("x_col")
    y_col = request.form.get("y_col")
    selection_mode = request.form.get("linearization_mode", "manual").strip().lower()
    transform_key = request.form.get("linearization_transform", "y_vs_x").strip().lower()

    if not x_col or not y_col:
        return jsonify({"error": "Missing x_col or y_col"}), 400
    if selection_mode not in {"manual", "automatic"}:
        return jsonify({"error": "linearization_mode must be manual or automatic"}), 400
    if transform_key not in LINEARIZATION_TRANSFORMS:
        return jsonify({"error": "Invalid linearization transform"}), 400

    try:
        df = require_uploaded_dataframe()
        x, y = extract_numeric_xy(df, x_col, y_col)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if x.size < 2:
        return jsonify({"error": "At least two data points are required to linearize the data"}), 400

    if selection_mode == "manual":
        try:
            result = build_linearization_response(x, y, x_col, y_col, transform_key, selection_mode)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    best_result = None
    valid_options = []
    for candidate_key in LINEARIZATION_TRANSFORMS:
        try:
            candidate = build_linearization_response(x, y, x_col, y_col, candidate_key, selection_mode)
        except ValueError:
            continue

        valid_options.append(candidate_key)
        if best_result is None or candidate["r_squared"] > best_result["r_squared"]:
            best_result = candidate

    if best_result is None:
        return jsonify({"error": "No valid automatic linearization was found for this dataset"}), 400

    best_result["evaluated_transforms"] = valid_options
    return jsonify(best_result)


if __name__ == "__main__":
    app.run(debug=True)

#python backend.py
#python -m http.server 8000
