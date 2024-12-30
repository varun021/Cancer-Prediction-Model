# app.py
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import csv

app = Flask(__name__)


# Load and preprocess Trash
def load_data():
    df = pd.read_csv('cancer-prediction.csv')
    df['Cancer_Risk'] = df['Probability of Cancer'].str.rstrip('%').astype(float).apply(lambda x: 1 if x > 50 else 0)
    return df


# Prepare features for modeling
def prepare_features(df):
    habit_mapping = {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3}
    frequency_mapping = {'None': 0, 'Monthly': 1, 'Weekly': 2, 'Daily': 3}
    stress_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}

    feature_df = df.copy()
    feature_df['Smoking_Encoded'] = df['Smoking Habit'].map({'None': 0, 'Occasional': 1, 'Moderate': 2, 'Heavy': 3})
    feature_df['Drinking_Encoded'] = df['Drinking Habit'].map(
        {'None': 0, 'Occasional': 1, 'Moderate': 2, 'Frequent': 3})
    feature_df['Exercise_Encoded'] = df['Exercise Frequency'].map(frequency_mapping)
    feature_df['Stress_Encoded'] = df['Stress Level'].map(stress_mapping)
    feature_df['Family_History_Encoded'] = df['Family History of Cancer'].map({'No': 0, 'Yes': 1})

    features = ['Age', 'BMI', 'Smoking_Encoded', 'Drinking_Encoded',
                'Exercise_Encoded', 'Stress_Encoded', 'Family_History_Encoded']

    return feature_df[features]


# Train model with cross-validation
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Calculate various metrics
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    # ROC curve Trash
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    return model, scaler, accuracy, cv_scores, (fpr, tpr, roc_auc), (X_test_scaled, y_test)


# Global variables
df = load_data()
X = prepare_features(df)
y = df['Cancer_Risk']
model, scaler, model_accuracy, cv_scores, roc_data, test_data = train_model(X, y)
prediction_history = []


@app.route('/')
def home():
    visualizations = {
        'age_dist': create_age_distribution(),
        'risk_by_habit': create_risk_by_habit(),
        'correlation_matrix': create_correlation_matrix(),
        'feature_importance': create_feature_importance(),
        'roc_curve': create_roc_curve(),
        'lifestyle_impact': create_lifestyle_impact(),
        'risk_by_occupation': create_risk_by_occupation(),
        'age_bmi_scatter': create_age_bmi_scatter()
    }

    stats = {
        'model_accuracy': model_accuracy,
        'cv_scores_mean': cv_scores.mean(),
        'cv_scores_std': cv_scores.std(),
        'total_samples': len(df),
        'high_risk_count': len(df[df['Cancer_Risk'] == 1]),
        'low_risk_count': len(df[df['Cancer_Risk'] == 0])
    }

    return render_template('index.html',
                           visualizations=visualizations,
                           stats=stats,
                           prediction_history=prediction_history[-5:])


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_data = np.array([[
        float(data['age']),
        float(data['bmi']),
        float(data['smoking']),
        float(data['drinking']),
        float(data['exercise']),
        float(data['stress']),
        float(data['family_history'])
    ]])

    input_scaled = scaler.transform(input_data)
    prediction_prob = model.predict_proba(input_scaled)[0]
    risk_percentage = round(prediction_prob[1] * 100, 2)

    # Store prediction in history
    prediction_record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'risk_percentage': risk_percentage,
        'input_data': data
    }
    prediction_history.append(prediction_record)

    # Generate recommendations based on input
    recommendations = generate_recommendations(data)

    return jsonify({
        'risk_percentage': risk_percentage,
        'recommendations': recommendations
    })


@app.route('/download_report', methods=['POST'])
def download_report():
    data = request.get_json()

    # Create a detailed report
    report_buffer = io.StringIO()
    writer = csv.writer(report_buffer)
    writer.writerow(['Cancer Risk Assessment Report'])
    writer.writerow(['Generated on:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow([])
    writer.writerow(['Input Parameters'])
    for key, value in data['input_data'].items():
        writer.writerow([key.replace('_', ' ').title(), value])
    writer.writerow([])
    writer.writerow(['Risk Assessment'])
    writer.writerow(['Predicted Risk:', f"{data['risk_percentage']}%"])
    writer.writerow([])
    writer.writerow(['Recommendations'])
    for rec in data['recommendations']:
        writer.writerow(['- ' + rec])

    # Create response
    output = io.BytesIO()
    output.write(report_buffer.getvalue().encode('utf-8'))
    output.seek(0)

    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'risk_assessment_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )


def generate_recommendations(data):
    recommendations = []

    # Smoking recommendations
    if float(data['smoking']) > 1:
        recommendations.append("Consider reducing or quitting smoking to lower your cancer risk")

    # Exercise recommendations
    if float(data['exercise']) < 2:
        recommendations.append("Increase your physical activity to at least 150 minutes per week")

    # BMI recommendations
    bmi = float(data['bmi'])
    if bmi > 25:
        recommendations.append("Work on maintaining a healthy BMI through diet and exercise")

    # Stress management
    if float(data['stress']) > 2:
        recommendations.append("Consider stress management techniques such as meditation or counseling")

    # General recommendations
    recommendations.append("Maintain a balanced diet rich in fruits and vegetables")
    recommendations.append("Stay up to date with regular health check-ups and screenings")

    return recommendations


def create_age_distribution():
    fig = px.histogram(df, x='Age', color='Cancer_Risk',
                       title='Age Distribution by Cancer Risk',
                       labels={'Cancer_Risk': 'Cancer Risk'},
                       barmode='overlay',
                       color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
    fig.update_layout(template='plotly_white')
    return fig.to_json()


def create_risk_by_habit():
    habits = ['Smoking Habit', 'Drinking Habit', 'Exercise Frequency']
    fig = make_subplots(rows=1, cols=3, subplot_titles=habits)

    for i, habit in enumerate(habits, 1):
        habit_risk = df.groupby(habit)['Cancer_Risk'].mean().reset_index()
        fig.add_trace(
            go.Bar(x=habit_risk[habit], y=habit_risk['Cancer_Risk'], name=habit),
            row=1, col=i
        )

    fig.update_layout(height=400, title_text="Risk Analysis by Lifestyle Habits", template='plotly_white')
    return fig.to_json()


def create_correlation_matrix():
    correlation = X.corr()
    fig = px.imshow(correlation,
                    title='Feature Correlation Matrix',
                    color_continuous_scale='RdBu')
    fig.update_layout(template='plotly_white')
    return fig.to_json()


def create_feature_importance():
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)

    fig = px.bar(importance, x='importance', y='feature',
                 title='Feature Importance',
                 orientation='h',
                 color='importance',
                 color_continuous_scale='Viridis')
    fig.update_layout(template='plotly_white')
    return fig.to_json()


def create_roc_curve():
    fpr, tpr, roc_auc = roc_data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                             name=f'ROC curve (AUC = {roc_auc:.2f})',
                             mode='lines'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             name='Random',
                             mode='lines',
                             line=dict(dash='dash')))
    fig.update_layout(title='ROC Curve',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      template='plotly_white')
    return fig.to_json()


def create_lifestyle_impact():
    lifestyle_vars = ['Exercise Frequency', 'Dietary Habit', 'Stress Level']
    fig = make_subplots(rows=1, cols=3, subplot_titles=lifestyle_vars)

    for i, var in enumerate(lifestyle_vars, 1):
        impact = df.groupby(var)['Cancer_Risk'].mean().reset_index()
        fig.add_trace(
            go.Bar(x=impact[var], y=impact['Cancer_Risk'], name=var),
            row=1, col=i
        )

    fig.update_layout(height=400, title_text="Lifestyle Factors Impact Analysis", template='plotly_white')
    return fig.to_json()


def create_risk_by_occupation():
    occupation_risk = df.groupby('Occupation')['Cancer_Risk'].mean().sort_values(ascending=True)
    fig = px.bar(occupation_risk,
                 title='Cancer Risk by Occupation',
                 orientation='h',
                 color=occupation_risk.values,
                 color_continuous_scale='RdYlBu_r')
    fig.update_layout(template='plotly_white')
    return fig.to_json()


def create_age_bmi_scatter():
    fig = px.scatter(df, x='Age', y='BMI',
                     color='Cancer_Risk',
                     title='Age vs BMI Distribution',
                     color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                     labels={'Cancer_Risk': 'Cancer Risk'})
    fig.update_layout(template='plotly_white')
    return fig.to_json()


if __name__ == '__main__':
    app.run(debug=True)
