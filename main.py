import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier


spy = yf.Ticker('SPY')

history = spy.history(period="12mo")
history = pd.DataFrame(history)

history['Target'] = history['Close'].shift(-1) > history['Close']
history.dropna(inplace=True)

# print(history)

X = history[['Open', 'Close']]
y = history['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM

svm = svm.SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)
acc = accuracy_score(y_test, svm_predictions)
svm_confusion = confusion_matrix(y_test, svm_predictions)
svm_report = classification_report(y_test, svm_predictions, zero_division=1)

print(f'SVM Accuracy: {acc:.4f}')
print('Confusion Matrix:', svm_confusion)
print('Report of SVM:', svm_report)

#MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

y_train_tensor = y_train_tensor.view(-1, 1)
y_test_tensor = y_test_tensor.view(-1, 1)


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


mlp_model = MLP(input_size=X_train_tensor.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    mlp_model.train()
    optimizer.zero_grad()

    output = mlp_model(X_train_tensor)
    loss = criterion(output, y_train_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


mlp_model.eval()
with torch.no_grad():
    predictions = mlp_model(X_test_tensor)
    predictions = predictions.round()
    accuracy = (predictions.eq(y_test_tensor).sum() / float(y_test_tensor.shape[0])).item()

print(f'Accuracy of the MLP: {accuracy:.4f}')

#LSTM

X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

X_train_tensor = torch.tensor(X_train_lstm, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_lstm, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        predictions = self.sigmoid(predictions)
        return predictions


lstm_model = LSTM(input_size=2, hidden_layer_size=50, output_size=1)

lstm_model.eval()
with torch.no_grad():

    test_predictions = lstm_model(X_test_tensor)
    test_predictions = test_predictions.round()
    correct_preds = (test_predictions.view(-1).eq(y_test_tensor.view(-1))).sum()
    accuracy = correct_preds.float() / y_test_tensor.shape[0]
    print(f'Accuracy of the LSTM: {accuracy:.4f}')


#KNN

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

knn_predictions = knn.predict(X_test_scaled)

knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_confusion = confusion_matrix(y_test, knn_predictions)
knn_report = classification_report(y_test, knn_predictions, zero_division=1)

print(f'KNN Accuracy: {knn_accuracy:.4f}')
print(f'KNN Confusion Matrix: {knn_confusion}')
print('KNN Report:', knn_report)


latest_data = spy.history(period="1d")
latest_features = latest_data[['Open', 'Close']]
latest_features_scaled = scaler.transform(latest_features)
latest_features_tensor = torch.tensor(latest_features_scaled, dtype=torch.float32)
latest_features_lstm = latest_features_scaled.reshape((1, 1, 2))  
latest_features_lstm_tensor = torch.tensor(latest_features_lstm, dtype=torch.float32)

svm_prediction = svm.predict(latest_features_scaled)
svm_prediction = svm_prediction.astype(int)
print("Tomorrow's SVM Prediction (Higher=1, Lower=0):", svm_prediction)

mlp_model.eval()
with torch.no_grad():
    mlp_prediction = mlp_model(latest_features_tensor)
    mlp_prediction = mlp_prediction.round().item()
print("Tomorrow's MLP Prediction (Higher=1, Lower=0):", mlp_prediction)

lstm_model.eval()
with torch.no_grad():
    lstm_prediction = lstm_model(latest_features_lstm_tensor)
    lstm_prediction = lstm_prediction.round().item()
print("Tomorrow's LSTM Prediction (Higher=1, Lower=0):", lstm_prediction)

knn_prediction = knn.predict(latest_features_scaled)
knn_prediction = knn_prediction.astype(int)
print("Tomorrow's KNN Prediction (Higher=1, Lower=0):", knn_prediction)

print(scaler.feature_names_in_)