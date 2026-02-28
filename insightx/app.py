from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import json
import random

app = Flask(__name__)

# ─────────────────────────────────────────
# Synthetic Dataset Generator
# ─────────────────────────────────────────
def generate_dataset(n=1000):
    np.random.seed(42)
    household_size = np.random.randint(1, 7, n)
    appliance_count = np.random.randint(2, 15, n)
    avg_temperature = np.random.uniform(15, 42, n)
    working_hours = np.random.uniform(4, 16, n)
    solar_usage = np.random.choice([0, 1], n, p=[0.6, 0.4])
    electricity_tariff = np.random.uniform(3.5, 9.5, n)
    day_type = np.random.choice(['weekday', 'weekend'], n)
    previous_consumption = np.random.uniform(10, 80, n)

    day_type_enc = np.where(np.array(day_type) == 'weekend', 1, 0)

    energy_consumption = (
        household_size * 4.2 +
        appliance_count * 1.8 +
        avg_temperature * 0.5 +
        working_hours * 2.1 -
        solar_usage * 8.5 +
        electricity_tariff * 0.3 +
        day_type_enc * 5.0 +
        previous_consumption * 0.4 +
        np.random.normal(0, 3, n)
    )

    high_usage = (energy_consumption > energy_consumption.mean()).astype(int)

    efficiency_score = (
        (1 - solar_usage) * 30 +
        (appliance_count / 15) * 40 +
        np.random.uniform(0, 30, n)
    )
    efficiency_category = pd.cut(efficiency_score, bins=3, labels=['High', 'Medium', 'Low'])

    df = pd.DataFrame({
        'household_size': household_size,
        'appliance_count': appliance_count,
        'avg_temperature': avg_temperature.round(2),
        'working_hours': working_hours.round(2),
        'solar_usage': solar_usage,
        'electricity_tariff': electricity_tariff.round(2),
        'day_type': day_type,
        'previous_consumption': previous_consumption.round(2),
        'energy_consumption': energy_consumption.round(2),
        'high_usage': high_usage,
        'efficiency_category': efficiency_category
    })
    return df

# Train models once at startup
df = generate_dataset(1000)
le = LabelEncoder()
df['day_type_enc'] = le.fit_transform(df['day_type'])
df['efficiency_enc'] = LabelEncoder().fit_transform(df['efficiency_category'])

features = ['household_size', 'appliance_count', 'avg_temperature',
            'working_hours', 'solar_usage', 'electricity_tariff',
            'day_type_enc', 'previous_consumption']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Linear Regression
X_train, X_test, y_train, y_test = train_test_split(X, df['energy_consumption'], test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_metrics = {
    'r2': round(r2_score(y_test, lr_preds), 4),
    'rmse': round(np.sqrt(mean_squared_error(y_test, lr_preds)), 4)
}

# Decision Tree
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, df['high_usage'], test_size=0.2, random_state=42)
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train2, y_train2)
dt_preds = dt_model.predict(X_test2)
dt_metrics = {'accuracy': round(accuracy_score(y_test2, dt_preds), 4)}

# KNN
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_scaled, df['efficiency_enc'], test_size=0.2, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train3, y_train3)
knn_preds = knn_model.predict(X_test3)
knn_metrics = {'accuracy': round(accuracy_score(y_test3, knn_preds), 4)}

# K-Means
kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans_model.fit_predict(X_scaled)
df['cluster'] = kmeans_labels

cluster_names = {0: 'Eco Savers', 1: 'High Consumers', 2: 'Moderate Users', 3: 'Solar Champions'}

# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/visualize')
def visualize():
    return render_template('visualize.html')

@app.route('/api/dataset-stats')
def dataset_stats():
    stats = {
        'total_records': len(df),
        'avg_consumption': round(df['energy_consumption'].mean(), 2),
        'max_consumption': round(df['energy_consumption'].max(), 2),
        'min_consumption': round(df['energy_consumption'].min(), 2),
        'solar_usage_pct': round(df['solar_usage'].mean() * 100, 1),
        'high_usage_pct': round(df['high_usage'].mean() * 100, 1),
        'lr_r2': lr_metrics['r2'],
        'lr_rmse': lr_metrics['rmse'],
        'dt_accuracy': dt_metrics['accuracy'],
        'knn_accuracy': knn_metrics['accuracy'],
    }
    return jsonify(stats)

@app.route('/api/chart-data')
def chart_data():
    # Consumption distribution
    bins = pd.cut(df['energy_consumption'], bins=10)
    dist = bins.value_counts().sort_index()
    consumption_dist = {
        'labels': [str(b) for b in dist.index],
        'values': dist.values.tolist()
    }

    # Cluster distribution
    cluster_dist = df['cluster'].value_counts().sort_index()
    cluster_data = {
        'labels': [cluster_names[i] for i in cluster_dist.index],
        'values': cluster_dist.values.tolist()
    }

    # Scatter: temperature vs consumption (sample 200)
    sample = df.sample(200, random_state=1)
    scatter = {
        'x': sample['avg_temperature'].tolist(),
        'y': sample['energy_consumption'].tolist(),
        'cluster': sample['cluster'].tolist()
    }

    # Time series mock (monthly avg)
    monthly = [round(random.uniform(35, 65), 2) for _ in range(12)]
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    # Efficiency categories
    eff = df['efficiency_category'].value_counts()
    efficiency_data = {
        'labels': eff.index.tolist(),
        'values': eff.values.tolist()
    }

    return jsonify({
        'consumption_dist': consumption_dist,
        'cluster_data': cluster_data,
        'scatter': scatter,
        'monthly': {'labels': months, 'values': monthly},
        'efficiency': efficiency_data,
        'feature_importance': {
            'features': features,
            'values': [abs(c) for c in lr_model.coef_]
        }
    })

@app.route('/api/generate-dataset')
def api_generate():
    new_df = generate_dataset(1000)
    sample = new_df.head(10).to_dict(orient='records')
    return jsonify({'status': 'success', 'records': 1000, 'sample': sample})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    try:
        household_size = float(data['household_size'])
        appliance_count = float(data['appliance_count'])
        avg_temperature = float(data['avg_temperature'])
        working_hours = float(data['working_hours'])
        solar_usage = float(data['solar_usage'])
        electricity_tariff = float(data['electricity_tariff'])
        day_type = 1 if data['day_type'] == 'weekend' else 0
        previous_consumption = float(data['previous_consumption'])

        input_arr = np.array([[household_size, appliance_count, avg_temperature,
                                working_hours, solar_usage, electricity_tariff,
                                day_type, previous_consumption]])
        input_scaled = scaler.transform(input_arr)

        # Linear Regression prediction
        lr_pred = float(lr_model.predict(input_arr)[0])

        # Decision Tree prediction
        dt_pred = int(dt_model.predict(input_arr)[0])
        dt_label = 'High Usage' if dt_pred == 1 else 'Low Usage'

        # KNN prediction
        knn_pred = int(knn_model.predict(input_scaled)[0])
        knn_labels = ['High Efficiency', 'Medium Efficiency', 'Low Efficiency']
        knn_label = knn_labels[knn_pred] if knn_pred < 3 else 'Medium Efficiency'

        # K-Means cluster
        km_cluster = int(kmeans_model.predict(input_scaled)[0])
        km_label = cluster_names[km_cluster]

        # Recommendations
        recs = []
        if solar_usage == 0:
            recs.append("🌞 Install solar panels to reduce consumption by ~15%")
        if appliance_count > 8:
            recs.append("🔌 Reduce active appliances during peak hours")
        if avg_temperature > 30:
            recs.append("❄️ Use energy-efficient cooling systems")
        if working_hours > 10:
            recs.append("⏰ Stagger high-power tasks to off-peak hours")

        return jsonify({
            'linear_regression': {'prediction': round(lr_pred, 2), 'unit': 'kWh'},
            'decision_tree': {'prediction': dt_label, 'raw': dt_pred},
            'knn': {'prediction': knn_label, 'raw': knn_pred},
            'kmeans': {'cluster': km_cluster, 'label': km_label},
            'recommendations': recs,
            'metrics': {'lr_r2': lr_metrics['r2'], 'dt_accuracy': dt_metrics['accuracy'], 'knn_accuracy': knn_metrics['accuracy']}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/bulk')
def bulk_page():
    return render_template('bulk.html')

@app.route('/api/bulk-generate-predict', methods=['POST'])
def bulk_generate_predict():
    data = request.json
    n_rows = max(1500, min(5000, int(data.get('rows', 1500))))
    seed = int(data.get('seed', 42))
    if seed == -1:
        seed = random.randint(0, 9999)
    noise_map = {'low': 1.5, 'medium': 3.0, 'high': 6.0}
    noise_level = noise_map.get(data.get('noise', 'medium'), 3.0)

    np.random.seed(seed)

    # ── Generate synthetic dataset ──
    household_size = np.random.randint(1, 7, n_rows)
    appliance_count = np.random.randint(2, 15, n_rows)
    avg_temperature = np.random.uniform(15, 42, n_rows).round(2)
    working_hours = np.random.uniform(4, 16, n_rows).round(2)
    solar_usage = np.random.choice([0, 1], n_rows, p=[0.6, 0.4])
    electricity_tariff = np.random.uniform(3.5, 9.5, n_rows).round(2)
    day_type = np.random.choice(['weekday', 'weekend'], n_rows)
    previous_consumption = np.random.uniform(10, 80, n_rows).round(2)
    day_type_enc = np.where(day_type == 'weekend', 1, 0)

    new_X = np.column_stack([
        household_size, appliance_count, avg_temperature, working_hours,
        solar_usage, electricity_tariff, day_type_enc, previous_consumption
    ])

    # ── Scale for KNN / KMeans ──
    new_X_scaled = scaler.transform(new_X)

    # ── Predict: Linear Regression ──
    lr_preds_bulk = lr_model.predict(new_X).round(2)

    # ── Predict: Decision Tree ──
    dt_preds_bulk = dt_model.predict(new_X)
    dt_labels_bulk = ['High Usage' if v == 1 else 'Low Usage' for v in dt_preds_bulk]

    # ── Predict: KNN ──
    knn_preds_bulk = knn_model.predict(new_X_scaled)
    knn_label_map = {0: 'High Efficiency', 1: 'Medium Efficiency', 2: 'Low Efficiency'}
    knn_labels_bulk = [knn_label_map.get(v, 'Medium Efficiency') for v in knn_preds_bulk]

    # ── Predict: KMeans ──
    km_preds_bulk = kmeans_model.predict(new_X_scaled)
    km_labels_bulk = [cluster_names[v] for v in km_preds_bulk]

    # ── Build rows ──
    rows_out = []
    for i in range(n_rows):
        rows_out.append({
            'household_size': int(household_size[i]),
            'appliance_count': int(appliance_count[i]),
            'avg_temperature': float(avg_temperature[i]),
            'working_hours': float(working_hours[i]),
            'solar_usage': int(solar_usage[i]),
            'electricity_tariff': float(electricity_tariff[i]),
            'day_type': day_type[i],
            'previous_consumption': float(previous_consumption[i]),
            'pred_kwh': float(lr_preds_bulk[i]),
            'pred_high_usage': dt_labels_bulk[i],
            'pred_efficiency': knn_labels_bulk[i],
            'pred_cluster': int(km_preds_bulk[i]),
            'pred_cluster_label': km_labels_bulk[i]
        })

    # ── Summary stats ──
    high_usage_count = sum(1 for r in dt_labels_bulk if r == 'High Usage')
    high_eff_count = sum(1 for r in knn_labels_bulk if 'High' in r)
    solar_count = int(solar_usage.sum())
    cluster_counts = [int((km_preds_bulk == c).sum()) for c in range(4)]

    # kWh distribution (10 bins)
    kwh_arr = lr_preds_bulk
    bins = np.linspace(kwh_arr.min(), kwh_arr.max(), 11)
    kwh_hist, _ = np.histogram(kwh_arr, bins=bins)
    kwh_bin_labels = [f'B{i+1}' for i in range(10)]

    # ── CSV Content ──
    csv_header = 'household_size,appliance_count,avg_temperature,working_hours,solar_usage,electricity_tariff,day_type,previous_consumption,pred_kwh,pred_high_usage,pred_efficiency,pred_cluster,pred_cluster_label\n'
    csv_rows = '\n'.join([
        f"{r['household_size']},{r['appliance_count']},{r['avg_temperature']},{r['working_hours']},"
        f"{r['solar_usage']},{r['electricity_tariff']},{r['day_type']},{r['previous_consumption']},"
        f"{r['pred_kwh']},{r['pred_high_usage']},{r['pred_efficiency']},{r['pred_cluster']},{r['pred_cluster_label']}"
        for r in rows_out
    ])

    return jsonify({
        'total': n_rows,
        'n_clusters': 4,
        'rows': rows_out,
        'csv_content': csv_header + csv_rows,
        'summary': {
            'avg_kwh': round(float(kwh_arr.mean()), 2),
            'high_usage_pct': round(high_usage_count / n_rows * 100, 1),
            'high_eff_pct': round(high_eff_count / n_rows * 100, 1),
            'solar_pct': round(solar_count / n_rows * 100, 1)
        },
        'chart_data': {
            'kwh_dist': {'labels': kwh_bin_labels, 'values': kwh_hist.tolist()},
            'cluster_counts': cluster_counts,
            'usage_counts': [int(n_rows - high_usage_count), int(high_usage_count)]
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
