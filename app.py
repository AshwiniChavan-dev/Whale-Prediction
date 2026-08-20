import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mobile Game Payer Prediction",
    page_icon="🎮",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = Path("models/whale_prediction_model.pkl")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH.name}' was not found. "
            "Please place it in the same folder as app.py."
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the prediction model.")
    st.exception(e)
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🎮 Mobile Game Payer Prediction")

st.markdown(
    """
    ### Predict whether a mobile game player is likely to become a payer

    Enter the player's engagement, gameplay, device, country and
    acquisition information below. The trained machine learning model
    will predict whether the player is likely to convert into a payer.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📊 Project Information")

    st.write("**Problem:** Binary Classification")
    st.write("**Target:** `converted_to_payer`")
    st.write("**Dataset:** Mobile Game Player Data")
    st.write("**Prediction:** Payer / Non-Payer")

    st.divider()

    st.info(
        """
        The application uses the saved machine learning pipeline.
        Preprocessing is performed automatically by the trained model.
        """
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("👤 Player Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=13,
        max_value=100,
        value=25,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        options=["Male", "Female", "Other"]
    )

    country = st.selectbox(
        "Country",
        options=[
            "USA",
            "Brazil",
            "India",
            "Indonesia",
            "Mexico",
            "Germany",
            "Philippines",
            "UK",
            "Japan",
            "Canada"
        ]
    )

    device_type = st.selectbox(
        "Device Type",
        options=["Android", "iOS"]
    )

with col2:
    acquisition_channel = st.selectbox(
        "Acquisition Channel",
        options=[
            "organic",
            "paid_social",
            "paid_search",
            "influencer",
            "referral"
        ]
    )

    days_since_install = st.number_input(
        "Days Since Install",
        min_value=0,
        max_value=3650,
        value=30,
        step=1
    )

    sessions_last_7d = st.number_input(
        "Sessions in Last 7 Days",
        min_value=0,
        max_value=500,
        value=10,
        step=1
    )

    avg_session_length_min = st.number_input(
        "Average Session Length (minutes)",
        min_value=0.0,
        max_value=1000.0,
        value=20.0,
        step=1.0
    )

    total_playtime_hours = st.number_input(
        "Total Playtime (hours)",
        min_value=0.0,
        max_value=10000.0,
        value=20.0,
        step=1.0
    )

with col3:
    levels_completed = st.number_input(
        "Levels Completed",
        min_value=0,
        max_value=10000,
        value=20,
        step=1
    )

    current_level = st.number_input(
        "Current Level",
        min_value=0,
        max_value=10000,
        value=25,
        step=1
    )

    tutorial_completed = st.number_input(
        "Tutorial Completed",
        min_value=0,
        max_value=1,
        value=1,
        step=1
    )

    num_friends_connected = st.number_input(
        "Friends Connected",
        min_value=0,
        max_value=1000,
        value=5,
        step=1
    )


# ============================================================
# ENGAGEMENT INFORMATION
# ============================================================

st.subheader("📈 Player Engagement")

col1, col2, col3 = st.columns(3)

with col1:
    push_notifications_enabled = st.number_input(
        "Push Notifications Enabled",
        min_value=0,
        max_value=1,
        value=1,
        step=1
    )

    ad_views = st.number_input(
        "Ad Views",
        min_value=0,
        max_value=10000,
        value=20,
        step=1
    )

    rewarded_ad_views = st.number_input(
        "Rewarded Ad Views",
        min_value=0,
        max_value=10000,
        value=10,
        step=1
    )

    store_visits = st.number_input(
        "Store Visits",
        min_value=0,
        max_value=10000,
        value=5,
        step=1
    )

with col2:
    items_viewed_in_store = st.number_input(
        "Items Viewed in Store",
        min_value=0,
        max_value=10000,
        value=10,
        step=1
    )

    wishlist_items = st.number_input(
        "Wishlist Items",
        min_value=0,
        max_value=10000,
        value=2,
        step=1
    )

    days_active_last_30 = st.number_input(
        "Days Active in Last 30 Days",
        min_value=0,
        max_value=30,
        value=10,
        step=1
    )

    streak_days = st.number_input(
        "Streak Days",
        min_value=0,
        max_value=3650,
        value=3,
        step=1
    )

with col3:
    rage_quit_events = st.number_input(
        "Rage Quit Events",
        min_value=0,
        max_value=10000,
        value=0,
        step=1
    )

    level_fail_rate = st.number_input(
        "Level Fail Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
        format="%.2f"
    )

    social_shares = st.number_input(
        "Social Shares",
        min_value=0,
        max_value=10000,
        value=2,
        step=1
    )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "country": [country],
    "acquisition_channel": [acquisition_channel],
    "device_type": [device_type],
    "days_since_install": [days_since_install],
    "sessions_last_7d": [sessions_last_7d],
    "avg_session_length_min": [avg_session_length_min],
    "total_playtime_hours": [total_playtime_hours],
    "levels_completed": [levels_completed],
    "current_level": [current_level],
    "tutorial_completed": [tutorial_completed],
    "num_friends_connected": [num_friends_connected],
    "push_notifications_enabled": [push_notifications_enabled],
    "ad_views": [ad_views],
    "rewarded_ad_views": [rewarded_ad_views],
    "store_visits": [store_visits],
    "items_viewed_in_store": [items_viewed_in_store],
    "wishlist_items": [wishlist_items],
    "days_active_last_30": [days_active_last_30],
    "streak_days": [streak_days],
    "rage_quit_events": [rage_quit_events],
    "level_fail_rate": [level_fail_rate],
    "social_shares": [social_shares]
})


# ============================================================
# SHOW INPUT SUMMARY
# ============================================================

with st.expander("🔎 View Input Data"):
    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🎯 Predict Payer Conversion",
    type="primary",
    use_container_width=True
)


if predict_button:

    try:
        # ---------------------------------------------
        # Prediction
        # ---------------------------------------------

        prediction = model.predict(input_data)[0]

        # ---------------------------------------------
        # Display Prediction
        # ---------------------------------------------

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success(
                "💰 Prediction: **Player is likely to become a payer**"
            )
        else:
            st.info(
                "👤 Prediction: **Player is unlikely to become a payer**"
            )

        # ---------------------------------------------
        # Probability
        # ---------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            classes = model.classes_

            probability_df = pd.DataFrame({
                "Class": classes,
                "Probability": probabilities
            })

            st.subheader("📈 Prediction Probability")

            for class_value, probability in zip(
                classes,
                probabilities
            ):
                if class_value == 1:
                    label = "Payer"
                else:
                    label = "Non-Payer"

                st.metric(
                    label=label,
                    value=f"{probability * 100:.2f}%"
                )

            st.bar_chart(
                probability_df.set_index("Class")
            )

    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Mobile Game Payer Prediction | Machine Learning + Streamlit"
)