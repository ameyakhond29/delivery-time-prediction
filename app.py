"""
DoorDash Delivery-Time Demo (Streamlit)
=======================================
Enter an order as it looks at placement time and get:
  * Typical ETA        - MAE-optimal HistGradientBoosting prediction
  * Promised ETA       - 75th-percentile estimate (~73% of deliveries arrive by this time)
  * Very-late risk     - P(delivery > 45 min) from a companion classifier
  * Top reasons        - the main factors pushing this ETA up or down

Run:  streamlit run app.py        (trains on deliveries_train.csv on first load, cached)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
WEATHER_SEV = {"clear": 0, "cloudy": 1, "rain": 2, "heavy_rain": 3}
NUM = ["restaurant_avg_prep_minutes", "distance_km", "items_count", "order_subtotal", "hour", "dow",
       "is_weekend", "is_lunch", "is_dinner", "is_rush", "hour_sin", "hour_cos", "weather_sev", "log_trips",
       "trips_missing", "weather_missing", "subtotal_per_item", "dist_x_wsev", "prep_x_items", "dist_sq"]
CAT = ["cuisine", "city_zone", "courier_vehicle", "weather"]
FEATURES = NUM + CAT


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ts = pd.to_datetime(d["order_placed_at"])
    d["hour"] = ts.dt.hour; d["dow"] = ts.dt.dayofweek
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    d["is_lunch"] = d["hour"].between(11, 13).astype(int)
    d["is_dinner"] = d["hour"].between(18, 20).astype(int)
    d["is_rush"] = ((d["is_lunch"] == 1) | (d["is_dinner"] == 1)).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24); d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["weather_sev"] = d["weather"].map(WEATHER_SEV)
    d["dist_x_wsev"] = d["distance_km"] * d["weather_sev"].fillna(1)
    d["prep_x_items"] = d["restaurant_avg_prep_minutes"] * d["items_count"]
    d["dist_sq"] = d["distance_km"] ** 2
    d["subtotal_per_item"] = d["order_subtotal"] / d["items_count"]
    d["log_trips"] = np.log1p(d["courier_trips_completed"].clip(upper=5000))
    d["trips_missing"] = d["courier_trips_completed"].isna().astype(int)
    d["weather_missing"] = d["weather"].isna().astype(int)
    return d


def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])


@st.cache_resource(show_spinner="Training models on 8,000 historical orders…")
def load_models():
    train = add_features(pd.read_csv(HERE / "deliveries_train.csv"))
    X = train[FEATURES]
    y = train["delivery_minutes"].to_numpy(float)
    point = Pipeline([("pre", make_pre()), ("m", HistGradientBoostingRegressor(
        learning_rate=0.03, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, random_state=42))])
    point.fit(X, np.minimum(y, 60))                       # capped training labels (see notebook Sec. 2)
    promise = Pipeline([("pre", make_pre()), ("m", HistGradientBoostingRegressor(
        loss="quantile", quantile=0.75, learning_rate=0.03, max_iter=300, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42))])
    promise.fit(X, y)                                     # uncapped: the promise must respect the tail
    late_clf = Pipeline([("pre", make_pre()), ("m", HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, random_state=42))])
    late_clf.fit(X, (y > 45).astype(int))
    return point, promise, late_clf


st.set_page_config(page_title="DoorDash ETA Demo", page_icon="🛵")
st.title("🛵 DoorDash Delivery-Time Predictor")
st.caption("Typical ETA (MAE model, CV 3.27 min) · Promised ETA (75th percentile, ~73% coverage) · "
           "Very-late risk (>45 min, ROC-AUC 0.93). Full analysis: `delivery_time_prediction.ipynb`.")

with st.sidebar:
    st.header("Order at placement time")
    distance = st.slider("Distance (km)", 0.6, 13.0, 3.0, 0.1)
    prep = st.slider("Restaurant avg prep (min)", 5.0, 22.0, 14.0, 0.5)
    items = st.slider("Items in order", 1, 8, 2)
    subtotal = st.slider("Order subtotal ($)", 8.0, 125.0, 30.0, 1.0)
    hour = st.slider("Order hour", 10, 23, 19)
    dow_name = st.selectbox("Day of week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=4)
    cuisine = st.selectbox("Cuisine", ["pizza", "indian", "thai", "sushi", "mexican", "chinese", "salads", "burgers"])
    zone = st.selectbox("City zone", ["downtown", "midtown", "uptown", "suburbs_north", "suburbs_south"])
    vehicle = st.selectbox("Courier vehicle", ["scooter", "car", "bike"])
    weather = st.selectbox("Weather", ["clear", "cloudy", "rain", "heavy_rain"])
    trips = st.slider("Courier trips completed", 1, 5000, 150)

dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(dow_name)
monday = pd.Timestamp("2025-06-16")  # a Monday
raw = pd.DataFrame([{
    "order_placed_at": monday + pd.Timedelta(days=dow) + pd.Timedelta(hours=int(hour)),
    "cuisine": cuisine, "restaurant_avg_prep_minutes": prep, "city_zone": zone,
    "distance_km": distance, "items_count": items, "order_subtotal": subtotal,
    "courier_vehicle": vehicle, "courier_trips_completed": float(trips), "weather": weather,
}])
order = add_features(raw)[FEATURES]

point, promise, late_clf = load_models()
eta = float(np.clip(point.predict(order)[0], 8, 65))
eta_q75 = float(np.clip(promise.predict(order)[0], 8, None))
risk = float(late_clf.predict_proba(order)[0, 1])

c1, c2, c3 = st.columns(3)
c1.metric("Typical ETA", f"{eta:.0f} min", help="MAE-optimal prediction — half of similar orders are faster, half slower.")
c2.metric("Promised ETA (show customer)", f"{eta_q75:.0f} min", help="75th percentile: ~73% of similar deliveries arrive by this time.")
c3.metric("Very-late risk (>45 min)", f"{risk*100:.0f}%", help="Classifier probability. Above ~50% operations should intervene.")

if risk >= 0.5:
    st.error("⚠️ High late risk — consider upgrading the courier vehicle, prioritising this order, or warning the customer early.")
elif risk >= 0.3:
    st.warning("🟡 Elevated late risk — keep an eye on this order.")
else:
    st.success("🟢 Low late risk — this order looks routine.")
st.progress(min(risk, 1.0))

reasons = []
if distance >= 6: reasons.append(f"Long trip ({distance:.1f} km) — distance is the #1 driver (~+2.7 min/km)")
if prep >= 17: reasons.append(f"Slow kitchen (avg prep {prep:.0f} min)")
if hour in range(11, 14) or hour in range(18, 21): reasons.append("Rush hour (lunch/dinner peak, ~+3 min)")
if weather == "heavy_rain": reasons.append("Heavy rain, compounded by distance")
elif weather == "rain": reasons.append("Rain slowing travel")
if vehicle == "bike" and distance > 3: reasons.append("Bike on a longer trip — slower than scooter/car")
if items >= 5: reasons.append(f"Large basket ({items} items) slows kitchen handoff")
if trips < 50: reasons.append("Novice courier (<50 trips)")
if not reasons: reasons.append("Nothing unusual — short/average inputs across the main drivers")
if distance <= 1.5 and prep <= 9: reasons.append("Short trip from a fast kitchen — pulling the ETA down")
if trips >= 1000: reasons.append("Experienced courier (1,000+ trips) — pulling the ETA down")

st.subheader("Why this estimate?")
for r in reasons:
    st.markdown(f"- {r}")
st.caption("Models: HistGradientBoosting trained on deliveries_train.csv (notebook Sections 5 & 8). "
           "Point model trained on labels capped at 60 min; quantile model uncapped; late-risk threshold >45 min.")
