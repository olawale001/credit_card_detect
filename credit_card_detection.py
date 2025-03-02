from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from google.colab import drive

drive.mount('/content/drive')

df = pd.read_csv("/content/drive/My Drive/transactions.csv")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=["is_fraud"]))

input_dim = X_scaled.shape[1]
encoding_dim = 8

input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation="relu")(input_layer)
encoded = Dense(encoding_dim, activation="relu")(encoded)

decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")


autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32)


reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)


threshold = np.percentile(mse, 95)
df["fraud_score"] = mse
df["is_anomaly"] = df["fraud_score"] > threshold


auc = roc_auc_score(df["is_fraud"], df["fraud_score"])
print(f"AUC Score: {auc}")