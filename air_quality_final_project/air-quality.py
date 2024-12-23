import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     TimeSeriesSplit)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

###############################################################################
# 1) Veri Yükleme
###############################################################################
data = pd.read_csv('air-quality-dataset.csv')
data.replace(-200, np.nan, inplace=True)
missing_values_before = data.isnull().sum()

# Tarih-Saat
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek

if 'Time' in data.columns:
    data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S')
    data['Hour'] = data['Time'].dt.hour
    data.drop(columns=['Time'], inplace=True)

###############################################################################
# 2) Feature Engineering
###############################################################################
data['CO_NOx_ratio'] = data['CO(GT)'] / (data['NOx(GT)'] + 1e-5)
data['CO_NO2_ratio'] = data['CO(GT)'] / (data['NO2(GT)'] + 1e-5)
data['Total_Gases'] = data[['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']].sum(axis=1)
data['CO_Hour'] = data['CO(GT)'] * data.get('Hour', 0)
data['NOx_T'] = data['NOx(GT)'] * data['T']
data['Combined_Interaction'] = data['CO_Hour'] * data['NOx_T']
data['Month_Name'] = data['Month'].map({
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
    6: "June", 7: "July", 8: "August", 9: "September",
    10: "October", 11: "November", 12: "December"
})

###############################################################################
# 3) Eksik Değerleri Doldurma
###############################################################################
def fill_missing_monthly_median(df, col_to_impute, month_col='Month'):
    df[month_col] = df[month_col].astype(int, errors='ignore')
    df[col_to_impute] = df.groupby(month_col)[col_to_impute].transform(lambda x: x.fillna(x.median()))
    return df

cols_with_nans = ['CO(GT)', 'NMHC(GT)', 'NOx(GT)', 'NO2(GT)']
for col in cols_with_nans:
    if col in data.columns:
        data = fill_missing_monthly_median(data, col)

numeric_data_after = data.select_dtypes(include=[np.number])
imp = SimpleImputer(strategy='median')
num_imputed = pd.DataFrame(imp.fit_transform(numeric_data_after),
                           columns=numeric_data_after.columns)
data_imputed = num_imputed.copy()

for ccol in ['Month_Name', 'Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 'Date']:
    if ccol in data.columns:
        data_imputed[ccol] = data[ccol].values

missing_values_after = data_imputed.isnull().sum()
print("\nEksik Veri Karşılaştırma:")
print(pd.DataFrame({
    'Feature': missing_values_before[missing_values_before > 0].index,
    'MissingBefore': missing_values_before[missing_values_before > 0].values,
    'MissingAfter': missing_values_after[missing_values_before[missing_values_before > 0].index].values
}))

###############################################################################
# 4) Tek Model (RandomForest) + GridSearch + Random Split
###############################################################################
X_cols = [c for c in data_imputed.columns
          if c not in ["CO(GT)", "Month_Name", "Date", "Day", "DayOfWeek"]]
y_col = "CO(GT)"

X = data_imputed[X_cols]
y = data_imputed[y_col]

# Random split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=1
)

scaler_main = StandardScaler()
X_train_sc = scaler_main.fit_transform(X_train)
X_test_sc = scaler_main.transform(X_test)

param_grid_rf = {
    "n_estimators": [100, 150],
    "max_depth": [10, 15],
    "min_samples_split": [10, 15],
    "min_samples_leaf": [5, 10],
    "max_features": ["sqrt", "log2"]
}
grid_search_rf = GridSearchCV(
    RandomForestRegressor(random_state=1),
    param_grid_rf, cv=10, scoring='r2', n_jobs=-1
)
grid_search_rf.fit(X_train_sc, y_train)
best_rf_model = grid_search_rf.best_estimator_

y_train_pred = best_rf_model.predict(X_train_sc)
y_test_pred = best_rf_model.predict(X_test_sc)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

print("\n--- Tek Model Sonuçları (Random Split) ---")
print("Train R²:", train_r2, "Test R²:", test_r2)
print("Train RMSE:", train_rmse, "Test RMSE:", test_rmse)

# GRAFİK 1: Train vs Test R² ve RMSE
plt.figure(figsize=(12, 5))

# Sol tarafta R²
plt.subplot(1,2,1)
plt.bar(["Train", "Test"], [train_r2, test_r2], color=['blue','orange'])
plt.title("Train vs Test R² (Tek Model)")
plt.ylim(0,1)
plt.ylabel("R² Score")

# Sağ tarafta RMSE
plt.subplot(1,2,2)
plt.bar(["Train", "Test"], [train_rmse, test_rmse], color=['blue','orange'])
plt.title("Train vs Test RMSE (Tek Model)")
plt.ylabel("RMSE")

plt.tight_layout()
plt.show()

###############################################################################
# 5) TimeSeriesSplit
###############################################################################
data_imputed.sort_values(by='Date', inplace=True, ignore_index=True)

X_ts = data_imputed[X_cols]
y_ts = data_imputed[y_col]

scaler_ts = StandardScaler()
X_ts_sc = scaler_ts.fit_transform(X_ts)

tscv = TimeSeriesSplit(n_splits=5)
model_ts = RandomForestRegressor(n_estimators=100, random_state=1)

fold_idx = 1
ts_r2_scores = []
for train_index, test_index in tscv.split(X_ts_sc):
    X_tr, X_te = X_ts_sc[train_index], X_ts_sc[test_index]
    y_tr, y_te = y_ts.iloc[train_index], y_ts.iloc[test_index]

    model_ts.fit(X_tr, y_tr)
    y_te_pred = model_ts.predict(X_te)
    fold_r2 = r2_score(y_te, y_te_pred)
    ts_r2_scores.append(fold_r2)
    print(f"TimeSeriesSplit Fold{fold_idx} => R²={fold_r2:.3f}")
    fold_idx += 1

mean_ts_r2 = np.mean(ts_r2_scores)
print("Average R² (TimeSeriesSplit):", mean_ts_r2)

# GRAFİK 2: TimeSeriesSplit R² per fold
folds = np.arange(1, len(ts_r2_scores)+1)
plt.figure(figsize=(6,4))
plt.plot(folds, ts_r2_scores, marker='o', color='red', label='Fold R²')
plt.axhline(mean_ts_r2, color='blue', linestyle='--', label=f'Mean R²={mean_ts_r2:.3f}')
plt.title("TimeSeriesSplit R² Scores (Tek Model)")
plt.xlabel("Fold")
plt.ylabel("R²")
plt.ylim(0,1)
plt.legend()
plt.show()

###############################################################################
# 6) Sadece Sensör Değişkenleri ile CO Tahmini
###############################################################################
sensor_only_feats = [
    'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',
    'PT08.S4(NO2)', 'PT08.S5(O3)',
    'T', 'RH', 'AH', 'Hour', 'Month'
]
X_sensor = data_imputed[sensor_only_feats].copy()
y_sensor = data_imputed[y_col].copy()

scaler_so = StandardScaler()
X_sensor_sc = scaler_so.fit_transform(X_sensor)

X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
    X_sensor_sc, y_sensor, test_size=0.2, random_state=1
)
rf_so = RandomForestRegressor(n_estimators=100, random_state=1)
rf_so.fit(X_s_train, y_s_train)
y_s_train_pred = rf_so.predict(X_s_train)
y_s_test_pred = rf_so.predict(X_s_test)

sens_train_r2 = r2_score(y_s_train, y_s_train_pred)
sens_test_r2 = r2_score(y_s_test, y_s_test_pred)

print("\n--- Yalnızca Sensör + Çevresel Değişkenler Senaryosu ---")
print("Train R²:", sens_train_r2, "Test R²:", sens_test_r2)

# GRAFİK 3: All Features vs Sensor-Only
plt.figure(figsize=(5,4))
plt.bar(["All Features","Sensor-Only"], [test_r2, sens_test_r2], color=['green','gray'])
plt.title("Karşılaştırma: All vs Sensor-Only (Test R²)")
plt.ylabel("R² Score")
plt.ylim(0,1)
plt.show()

###############################################################################
# 7) Sensör Drift Analizi (Ay Bazında)
###############################################################################
month_perf = []
all_months = sorted(data_imputed['Month'].unique())
for m in all_months:
    sub_df = data_imputed[data_imputed['Month'] == m].copy()
    if len(sub_df) < 50:
        continue
    X_m = sub_df.drop(columns=["CO(GT)", "Month_Name", "Date", "Day", "DayOfWeek"])
    y_m = sub_df[y_col]

    if len(sub_df) < 30:
        continue
    X_m_tr, X_m_te, y_m_tr, y_m_te = train_test_split(
        X_m, y_m, test_size=0.3, random_state=1
    )
    sc_m = StandardScaler()
    X_m_tr_sc = sc_m.fit_transform(X_m_tr)
    X_m_te_sc = sc_m.transform(X_m_te)
    rf_m = RandomForestRegressor(n_estimators=50, random_state=1)
    rf_m.fit(X_m_tr_sc, y_m_tr)
    y_m_pred = rf_m.predict(X_m_te_sc)
    r2_m = r2_score(y_m_te, y_m_pred)
    month_perf.append({
        "Month": m,
        "Month_Name": sub_df["Month_Name"].iloc[0],
        "R²": r2_m,
        "Samples": len(sub_df)
    })

drift_df = pd.DataFrame(month_perf)
print("\n--- Aylara Göre Model Performansı ---")
print(drift_df)

# GRAFİK 4: Aylık R² (Drift)
plt.figure(figsize=(8,5))
plt.bar(drift_df['Month'], drift_df['R²'], color='skyblue')
plt.title("Aylara Göre Model Performansı (R²)")
plt.xlabel("Month")
plt.ylabel("R² Score")
plt.ylim(0,1)
plt.xticks(drift_df['Month'])
plt.tight_layout()
plt.show()

###############################################################################
# 8) Cluster Tabanlı Model + Param Tuning
###############################################################################
cluster_features = [
    'T', 'RH', 'Hour',
    'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',
    'PT08.S4(NO2)', 'PT08.S5(O3)'
]
cluster_data = data_imputed[cluster_features].values
scaler_cluster = StandardScaler()
cluster_data_sc = scaler_cluster.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(cluster_data_sc)
data_imputed['ClusterID'] = kmeans.labels_

cluster_results = []
for cid in sorted(data_imputed['ClusterID'].unique()):
    cdf = data_imputed[data_imputed['ClusterID'] == cid].copy()
    if len(cdf) < 50:
        continue

    X_c = cdf.drop(columns=["CO(GT)", "Month_Name", "ClusterID", "Date", "Day", "DayOfWeek"])
    y_c = cdf[y_col]

    X_ct, X_ce, y_ct, y_ce = train_test_split(X_c, y_c, test_size=0.2, random_state=1)
    sc_c = StandardScaler()
    X_ct_sc = sc_c.fit_transform(X_ct)
    X_ce_sc = sc_c.transform(X_ce)

    # GridSearch cluster bazında
    param_grid_cl = {"n_estimators": [50, 100], "max_depth": [10, 15, None]}
    gs_cl = GridSearchCV(
        RandomForestRegressor(random_state=1),
        param_grid_cl, cv=3, scoring='r2', n_jobs=-1
    )
    gs_cl.fit(X_ct_sc, y_ct)
    best_c_model = gs_cl.best_estimator_

    y_ct_pred = best_c_model.predict(X_ct_sc)
    y_ce_pred = best_c_model.predict(X_ce_sc)
    r2_train_c = r2_score(y_ct, y_ct_pred)
    r2_test_c = r2_score(y_ce, y_ce_pred)

    cluster_results.append({
        "ClusterID": cid,
        "Train R²": r2_train_c,
        "Test R²": r2_test_c,
        "Samples": len(cdf),
        "BestParams": gs_cl.best_params_
    })

cluster_df = pd.DataFrame(cluster_results)
print("\n--- Cluster-based Model Sonuçları ---")
print(cluster_df)

# GRAFİK 5: Cluster Tabanlı Test R²
plt.figure(figsize=(6,4))
plt.bar(cluster_df['ClusterID'].astype(str), cluster_df['Test R²'], color='orange')
plt.title("Test R² Scores by Cluster")
plt.xlabel("ClusterID")
plt.ylabel("R² Score")
plt.ylim(0,1)
plt.tight_layout()
plt.show()
