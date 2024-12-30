# Cancer Prediction Model

A web-based cancer risk prediction system built using Flask, Random Forest Classifier, and Plotly for interactive visualizations.

## 📚 **Project Overview**
This application predicts cancer risk based on lifestyle, family history, and health parameters. It provides:
- Risk assessment reports
- Visual analysis of risk factors
- Personalized recommendations

## 🛠️ **Tech Stack**
- **Backend:** Flask
- **Frontend:** HTML, CSS (ShadCN integration)
- **Machine Learning:** Random Forest Classifier
- **Data Visualization:** Plotly, Seaborn, Matplotlib
- **Data Handling:** Pandas, NumPy

## 📊 **Features**
- Cancer risk prediction based on user inputs
- Interactive visualizations (e.g., ROC Curve, Feature Importance)
- Downloadable risk assessment reports
- Personalized health recommendations

---

## 🚀 **Setup and Installation**

### **1. Clone the Repository**
```bash
git clone <repository_url>
cd Cancer-Prediction-Model
```

### **2. Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Add Dataset**
Place your `cancer-prediction.csv` dataset in the root directory.

### **5. Run the Application**
```bash
python app.py
```
Access the app at: **http://127.0.0.1:5000**

---

## 📝 **Dataset Format**
Ensure your dataset has the following columns:
- `Age`
- `BMI`
- `Smoking Habit`
- `Drinking Habit`
- `Exercise Frequency`
- `Stress Level`
- `Family History of Cancer`
- `Probability of Cancer`

### Sample Row:
| Age | BMI | Smoking Habit | Drinking Habit | Exercise Frequency | Stress Level | Family History of Cancer | Probability of Cancer |
|-----|-----|---------------|---------------|---------------------|-------------|--------------------------|-----------------------|
| 45  | 25  | Moderate      | Occasional    | Weekly             | Medium      | Yes                      | 60%                   |

---

## 📈 **Available Visualizations**
- Age Distribution
- Risk by Lifestyle Habits
- Correlation Matrix
- Feature Importance
- ROC Curve
- Lifestyle Impact
- Risk by Occupation
- Age vs BMI Scatter Plot

---

## 📑 **Endpoints**
- `/`: Home page with visualizations and stats
- `/predict`: Predict cancer risk (POST)
- `/download_report`: Download risk report (POST)

---

## 🤝 **Contributing**
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add some feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request.

---

## 🛡️ **License**
This project is licensed under the MIT License.

---

## 📬 **Contact**
For any queries or collaboration opportunities, feel free to reach out!

---

🎯 **Happy Coding!** 🚀

