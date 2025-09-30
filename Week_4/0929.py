import xml.etree.ElementTree as tr
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score

tree = tr.parse('O-A0038-003.xml')
root = tree.getroot()
ns = 'urn:cwa:gov:tw:cwacommon:0.1'

content_text = None
for content in root.findall(f'.//{{{ns}}}Content'):
    if content.text and "," in content.text:
        content_text = content.text
        break

clean_text = content_text.strip()
clean_text = re.sub(r'(E[+-]\d{2})(-)', r'\1,\2', clean_text)

numbers = []
for line in clean_text.split("\n"):
    items = line.split(',')
    if items != "":
        for item in items:
            numbers.append(float(item))

cols = 67
rows = 120
lon_start = 120.00
lat_start = 21.88
lon_step = 0.03
lat_step = 0.03

classification_data = []
regression_data = []

grid = np.array(numbers).reshape((rows, cols))

for i in range(rows):
    lat = lat_start + i * lat_step
    for j in range(cols):
        lon = lon_start + j * lon_step
        val = grid[i][j]
        if val == -999:
            classification_data.append((lon, lat, 0))
        else:
            classification_data.append((lon, lat, 1))
            regression_data.append((lon, lat, val))

df_classification = pd.DataFrame(classification_data, columns=['lon', 'lat', 'label'])
df_regression = pd.DataFrame(regression_data, columns=['lon', 'lat', 'value'])

X_cls = df_classification[['lon', 'lat']]
y_cls = df_classification['label']
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_cls, y_train_cls)
y_pred_cls = clf.predict(X_test_cls)
print("=== Classification Report ===")
print(classification_report(y_test_cls, y_pred_cls))

X_reg = df_regression[['lon', 'lat']]
y_reg = df_regression['value']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)
print("=== Regression Report ===")
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_reg):.2f}")
print(f"R²: {r2_score(y_test_reg, y_pred_reg):.2f}")

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
colors = df_classification['label'].map({0: 'black', 1: 'blue'})
plt.scatter(df_classification['lon'], df_classification['lat'], c=colors, s=10, alpha=0.6)
plt.title('Classification: Valid (blue) vs Invalid (black) Points')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

plt.subplot(1, 2, 2)
sc = plt.scatter(df_regression['lon'], df_regression['lat'], c=df_regression['value'], cmap='coolwarm', s=10, alpha=0.8)
plt.colorbar(sc, label='Temperature (°C)')
plt.title('Regression: Temperature Distribution')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

plt.tight_layout()
plt.show()
