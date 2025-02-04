import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

df_purchase = pd.read_csv("purchase_behaviour.csv")
df_transaction = pd.read_csv("transaction_data.csv")

df_merged = df_transaction.merge(df_purchase, on="LYLTY_CARD_NBR", how="left")

encoder_lifestage = LabelEncoder()
encoder_premium = LabelEncoder()

df_merged["LIFESTAGE"] = encoder_lifestage.fit_transform(df_merged["LIFESTAGE"])
df_merged["PREMIUM_CUSTOMER"] = encoder_premium.fit_transform(df_merged["PREMIUM_CUSTOMER"])

customer_features = df_merged.groupby("LYLTY_CARD_NBR").agg({
    "TOT_SALES": "sum",
    "PROD_QTY": "sum",
    "LIFESTAGE": "first",
    "PREMIUM_CUSTOMER": "first"
}).reset_index()

customer_features["LOYAL"] = (customer_features["TOT_SALES"] > customer_features["TOT_SALES"].median()).astype(int)

X = customer_features.drop(columns=["LYLTY_CARD_NBR", "LOYAL"])
y = customer_features["LOYAL"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("classification_report.csv", index=True)

top_products = df_merged.groupby("PROD_NAME")["TOT_SALES"].sum().nlargest(3).reset_index()
top_products.to_csv("top_products.csv", index=False)

top_loyal_segments = customer_features.groupby(["LIFESTAGE", "PREMIUM_CUSTOMER"])["TOT_SALES"].sum().nlargest(3).reset_index()
top_loyal_segments.to_csv("top_loyal_segments.csv", index=False)
