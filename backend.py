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

PLOT_SAMPLE_COUNT = 600
INTERCEPT_TOL = 1e-9

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


def clean_number(value, tol=INTERCEPT_TOL):
    value = float(value)
    return 0.0 if np.isclose(value, 0.0, atol=tol, rtol=0.0) else value


def format_point_value(value):
    return f"{clean_number(value):.6g}"


def unique_sorted(values, tol=1e-7):
    cleaned = []
    for value in sorted(float(v) for v in values if np.isfinite(v)):
        if not cleaned or not np.isclose(value, cleaned[-1], atol=tol, rtol=0.0):
            cleaned.append(clean_number(value))
    return cleaned


def extract_real_roots(coefficients_desc, tol=1e-7):
    coeffs = np.asarray(coefficients_desc, dtype=float)
    if coeffs.size <= 1 or np.all(np.isclose(coeffs, 0.0, atol=tol, rtol=0.0)):
        return []

    roots = np.roots(coeffs)
    real_roots = [
        float(root.real)
        for root in roots
        if np.isfinite(root.real) and np.isclose(root.imag, 0.0, atol=tol, rtol=0.0)
    ]
    return unique_sorted(real_roots, tol=tol)


def make_intercept_points(axis, values):
    points = []
    values = unique_sorted(values)
    total = len(values)
    prefix = "X-intercept" if axis == "x" else "Y-intercept"

    for index, value in enumerate(values, start=1):
        x_val = clean_number(value if axis == "x" else 0.0)
        y_val = clean_number(0.0 if axis == "x" else value)
        name = f"{prefix} {index}" if total > 1 else prefix
        points.append({
            "axis": axis,
            "name": name,
            "x": x_val,
            "y": y_val,
            "label": f"{name} ({format_point_value(x_val)}, {format_point_value(y_val)})",
        })

    return points


def build_plot_x_bounds(x_values, extra_x_values=None):
    candidates = [float(v) for v in x_values if np.isfinite(v)]
    if extra_x_values:
        candidates.extend(float(v) for v in extra_x_values if np.isfinite(v))

    if not candidates:
        return -1.0, 1.0

    x_min = min(candidates)
    x_max = max(candidates)
    if np.isclose(x_min, x_max, atol=INTERCEPT_TOL, rtol=0.0):
        padding = max(1.0, abs(x_min) * 0.2, 0.25)
    else:
        padding = max((x_max - x_min) * 0.08, 0.1)

    return x_min - padding, x_max + padding


def build_linear_curve_points(slope, intercept, x_values, extra_x_values=None, num_points=PLOT_SAMPLE_COUNT):
    x_min, x_max = build_plot_x_bounds(x_values, extra_x_values)
    x_curve = np.linspace(x_min, x_max, num_points)
    y_curve = slope * x_curve + intercept
    return x_curve.tolist(), y_curve.tolist()


def evaluate_inverse_curve(x_values, coeffs):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.zeros_like(x_values, dtype=float)
    for power, coeff in enumerate(coeffs):
        if power == 0:
            y_values += coeff
        else:
            y_values += coeff * np.power(x_values, -power)
    return y_values


def build_inverse_curve_points(coeffs, x_values, extra_x_values=None, num_points=PLOT_SAMPLE_COUNT):
    x_min, x_max = build_plot_x_bounds(x_values, extra_x_values)

    if x_min < 0 < x_max:
        scale = max(abs(x_min), abs(x_max), 1.0)
        zero_gap = max(scale * 1e-3, 1e-6)
        left_count = max(2, num_points // 2)
        right_count = max(2, num_points - left_count)

        left_x = np.linspace(x_min, -zero_gap, left_count)
        right_x = np.linspace(zero_gap, x_max, right_count)
        left_y = evaluate_inverse_curve(left_x, coeffs)
        right_y = evaluate_inverse_curve(right_x, coeffs)

        return (
            left_x.tolist() + [None] + right_x.tolist(),
            left_y.tolist() + [None] + right_y.tolist(),
        )

    x_curve = np.linspace(x_min, x_max, num_points)
    x_curve = x_curve[~np.isclose(x_curve, 0.0, atol=INTERCEPT_TOL, rtol=0.0)]
    y_curve = evaluate_inverse_curve(x_curve, coeffs)
    return x_curve.tolist(), y_curve.tolist()


def build_linear_intercept_payload(slope, intercept):
    intercept_points = make_intercept_points("y", [intercept])
    intercept_notes = []
    extra_x_values = [0.0]

    if np.isclose(slope, 0.0, atol=INTERCEPT_TOL, rtol=0.0):
        if np.isclose(intercept, 0.0, atol=INTERCEPT_TOL, rtol=0.0):
            intercept_points.extend(make_intercept_points("x", [0.0]))
        else:
            intercept_notes.append("No finite x-intercept for this fitted line.")
    else:
        x_intercept = -intercept / slope
        intercept_points.extend(make_intercept_points("x", [x_intercept]))
        extra_x_values.append(x_intercept)

    return intercept_points, intercept_notes, extra_x_values


def build_polynomial_intercept_payload(coeffs):
    y_intercept = coeffs[0] if coeffs else 0.0
    x_intercepts = extract_real_roots(list(reversed(coeffs)))
    intercept_points = make_intercept_points("y", [y_intercept]) + make_intercept_points("x", x_intercepts)
    intercept_notes = []
    if not x_intercepts:
        intercept_notes.append("No finite x-intercept for this fitted curve.")
    return intercept_points, intercept_notes, [0.0] + x_intercepts


def build_inverse_intercept_payload(coeffs):
    intercept_points = []
    intercept_notes = []

    x_intercepts = [
        root for root in extract_real_roots(coeffs)
        if not np.isclose(root, 0.0, atol=INTERCEPT_TOL, rtol=0.0)
    ]
    intercept_points.extend(make_intercept_points("x", x_intercepts))
    if not x_intercepts:
        intercept_notes.append("No finite x-intercept for this inverse fit.")

    if np.all(np.isclose(coeffs[1:], 0.0, atol=INTERCEPT_TOL, rtol=0.0)):
        intercept_points.extend(make_intercept_points("y", [coeffs[0]]))
        extra_x_values = [0.0] + x_intercepts
    else:
        intercept_notes.append("No finite y-intercept for this inverse fit.")
        extra_x_values = x_intercepts

    return intercept_points, intercept_notes, extra_x_values


def build_exponential_intercept_payload(a_value):
    intercept_points = make_intercept_points("y", [a_value])
    intercept_notes = ["No finite x-intercept for this exponential fit."]
    return intercept_points, intercept_notes, [0.0]


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
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    x_axis_label = config["x_axis_label"](x_col, y_col)
    y_axis_label = config["y_axis_label"](x_col, y_col)
    intercept_points, intercept_notes, extra_x_values = build_linear_intercept_payload(slope, intercept)
    x_curve, y_curve = build_linear_curve_points(slope, intercept, x_transformed, extra_x_values)

    return {
        "selection_mode": selection_mode,
        "transform_key": transform_key,
        "transform_label": config["label"],
        "x_axis_label": x_axis_label,
        "y_axis_label": y_axis_label,
        "x_data": x_transformed.tolist(),
        "y_data": y_transformed.tolist(),
        "x_curve": x_curve,
        "y_curve": y_curve,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "equation_text": format_linear_equation(y_axis_label, x_axis_label, slope, intercept),
        "intercept_points": intercept_points,
        "intercept_notes": intercept_notes,
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

        coefficients = model.coef_.tolist()
        intercept_points, intercept_notes, extra_x_values = build_polynomial_intercept_payload(coefficients)
        x_curve_min, x_curve_max = build_plot_x_bounds(x, extra_x_values)
        x_curve = np.linspace(x_curve_min, x_curve_max, PLOT_SAMPLE_COUNT).reshape(-1, 1)
        x_curve_poly = poly.transform(x_curve)
        y_curve = model.predict(x_curve_poly)

        return jsonify({
            "x_data": x.tolist(),
            "y_data": y.tolist(),
            "x_curve": x_curve.squeeze().tolist(),
            "y_curve": y_curve.tolist(),
            "coefficients": coefficients,   # [a0, a1, a2, ...] for 1, x, x^2, ...
            "degree": degree,
            "model_type": model_type,
            "intercept_points": intercept_points,
            "intercept_notes": intercept_notes,
        })

    if model_type == "inverse":
        if x.size < degree + 1:
            return jsonify({"error": f"Not enough data points for n = {degree}"}), 400
        if np.any(np.isclose(x, 0.0)):
            return jsonify({"error": "Inverse model requires x != 0 for all points"}), 400

        X_inv = np.column_stack([np.ones_like(x)] + [x ** (-k) for k in range(1, degree + 1)])
        model = LinearRegression(fit_intercept=False)
        model.fit(X_inv, y)
        coefficients = model.coef_.tolist()
        intercept_points, intercept_notes, extra_x_values = build_inverse_intercept_payload(coefficients)
        x_curve, y_curve = build_inverse_curve_points(coefficients, x, extra_x_values)

        return jsonify({
            "x_data": x.tolist(),
            "y_data": y.tolist(),
            "x_curve": x_curve,
            "y_curve": y_curve,
            "coefficients": coefficients,   # [a0, a1, ...] for 1, x^-1, x^-2, ...
            "degree": degree,
            "model_type": model_type,
            "intercept_points": intercept_points,
            "intercept_notes": intercept_notes,
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
    intercept_points, intercept_notes, extra_x_values = build_exponential_intercept_payload(a)
    x_curve_min, x_curve_max = build_plot_x_bounds(x, extra_x_values)
    x_curve = np.linspace(x_curve_min, x_curve_max, PLOT_SAMPLE_COUNT)
    y_curve = a * np.exp(b * x_curve)

    return jsonify({
        "x_data": x.tolist(),
        "y_data": y.tolist(),
        "x_curve": x_curve.tolist(),
        "y_curve": y_curve.tolist(),
        "model_type": model_type,
        "exp_params": {"a": a, "b": b},
        "intercept_points": intercept_points,
        "intercept_notes": intercept_notes,
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
