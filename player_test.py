#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from openai import OpenAI

st.set_page_config(page_title="Custom Player Evaluator", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("📂 Custom Uploaded Player Evaluator")

# === Upload File ===
uploaded_file = st.file_uploader("📁 Upload your custom player dataset (CSV)", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# === Prepare Data ===
df = df.rename(columns={
    'player_name': 'Player Name',
    'team_name': 'Club',
    'player_match_goals': 'Goals',
    'player_match_assists': 'Assists',
    'player_match_dribbles': 'Dribbles',
    'player_match_interceptions': 'Interceptions',
    'player_match_np_xg': 'xG',
    'player_match_passing_ratio': 'PassingAccuracy',
    'player_match_minutes': 'Minutes'
})

df['Market_Value_SAR'] = df['Market_Value_EUR'] * 3.75
df['Asking_Price_SAR'] = df['Market_Value_SAR'] * np.random.uniform(1.05, 1.2, size=len(df))
df['Image'] = df['Player Name'].apply(lambda n: f"https://robohash.org/{n.replace(' ', '')}.png?set=set2")
df['Transfer_Chance'] = df['Market_Value_SAR'].apply(lambda x: random.uniform(0.6, 0.95))
df['Best_Fit_Club'] = df['Club'].apply(lambda _: random.choice(['Barcelona', 'Man United', 'Al Hilal', 'PSG']))
df['Position'] = random.choices(['Forward', 'Midfielder', 'Defender'], k=len(df))
df['Nationality'] = "Egyptian"
df['League'] = "Uploaded League"

# === Model Prediction ===
features = ['xG', 'Assists', 'Goals', 'Dribbles', 'Interceptions', 'PassingAccuracy', 'Market_Value_SAR']
X = df[features]
y = df['Market_Value_SAR'] * random.uniform(1.05, 1.15)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
df['Predicted_Year_1'] = model.predict(X)
df['Predicted_Year_2'] = df['Predicted_Year_1'] * 1.05
df['Predicted_Year_3'] = df['Predicted_Year_2'] * 1.05

# === Player Selector ===
player_names = df['Player Name'].tolist()
selected_name = st.selectbox("🔍 Select a player to evaluate", player_names)
player = df[df['Player Name'] == selected_name].iloc[0]

# === Layout
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"""
    <div class='card'>
        <h3>{player['Player Name']} ({player['Position']})</h3>
        <p><strong>Club:</strong> {player['Club']} | Age: {player['age']}</p>
        <p><strong>Market Value:</strong> €{player['Market_Value_EUR']:,.0f} / {player['Market_Value_SAR']:,.0f} SAR</p>
        <p><strong>Asking Price:</strong> {player['Asking_Price_SAR']:,.0f} SAR</p>
        <p><strong>Transfer Chance:</strong> {player['Transfer_Chance']*100:.1f}%</p>
        <p><strong>Best Fit Club:</strong> {player['Best_Fit_Club']}</p>
    </div>
    """, unsafe_allow_html=True)

    forecast_df = pd.DataFrame({
        "Year": ["2024", "2025", "2026"],
        "Predicted Value (SAR)": [
            player['Predicted_Year_1'],
            player['Predicted_Year_2'],
            player['Predicted_Year_3']
        ],
        "Club Asking Price (SAR)": [
            player['Asking_Price_SAR'],
            player['Asking_Price_SAR'],
            player['Asking_Price_SAR']
        ]
    })
    chart = alt.Chart(forecast_df).transform_fold(
        ['Predicted Value (SAR)', 'Club Asking Price (SAR)'],
        as_=['Metric', 'SAR Value']
    ).mark_line(point=True).encode(
        x='Year:N',
        y='SAR Value:Q',
        color='Metric:N'
    )
    st.altair_chart(chart, use_container_width=True)

with col2:
    st.image(player['Image'], width=100)

# === AI Commentary
st.header("🧠 Player Attitude Summary")
comment = f"{player['Player Name']} is showing promise with growing consistency and impact."
sentiment_prompt = [
    {"role": "system", "content": "You are a football sentiment expert."},
    {"role": "user", "content": f"Classify and summarize this comment: {comment}"}
]
try:
    response = client.chat.completions.create(model="gpt-4-1106-preview", messages=sentiment_prompt)
    ai_summary = response.choices[0].message.content
except Exception:
    ai_summary = "AI summary unavailable."

st.markdown(f"""
<div class='card'>
    <h4>🗨️ Public Comment:</h4>
    <p>{comment}</p>
    <h4>🤖 AI Summary:</h4>
    <p>{ai_summary}</p>
</div>
""", unsafe_allow_html=True)


# In[ ]:




