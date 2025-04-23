import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from openai import OpenAI

st.set_page_config(page_title="Football Talent Evaluator", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# 🔒 Hide Streamlit branding and profile pic
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .viewerBadge_link__1S137 {display: none !important;}
    .css-1r6slb0.egzxvld1 {display: none !important;} /* Hides profile image */
    .css-164nlkn {display: none !important;} /* Hides the Streamlit crown logo */
    </style>
    """, unsafe_allow_html=True)

# === Reference Market Values (for verification)
reference_values = {
    "Karim Hafez": 400000, "Blati": 1300000, "Ramadan Sobhi": 2500000,
    "Mostafa Fathi": 1500000, "Emmanuel Apeh": 350000, "Ali Gabr": 500000,
    "Marwan Hamdy": 1000000, "Ahmed El Shimi": 250000, "Fady Farid": 180000,
    "Ahmed Tawfik": 300000, "Ahmed Samy": 350000, "Al Mahdi Soliman": 200000,
    "Ahmed Ayman": 150000, "Mohanad Mostafa Lasheen": 750000, "Karim El Deeb": 450000
}

def validate_market_value(player_name, input_value):
    true_value = reference_values.get(player_name)
    if true_value is None:
        return "🟡 Unknown"
    return "🔴 Off by >€2M" if abs(true_value - input_value) > 2_000_000 else "🟢 Verified"

# === Data Prep Function
def prepare_data(df):
    df = df.rename(columns={
        'player_name': 'Player Name', 'team_name': 'Club',
        'player_match_goals': 'Goals', 'player_match_assists': 'Assists',
        'player_match_dribbles': 'Dribbles', 'player_match_interceptions': 'Interceptions',
        'player_match_np_xg': 'xG', 'player_match_passing_ratio': 'PassingAccuracy',
        'player_match_minutes': 'Minutes'
    })
    df['Market_Value_SAR'] = df['Market_Value_EUR'] * 3.75
    df['Asking_Price_SAR'] = df['Market_Value_SAR'] * np.random.uniform(1.05, 1.2, size=len(df))
    df['Verification'] = [validate_market_value(p, v) for p, v in zip(df['Player Name'], df['Market_Value_EUR'])]
    df['Image'] = df['Player Name'].apply(lambda n: f"https://robohash.org/{n.replace(' ', '')}.png?set=set2")
    df['Best_Fit_Club'] = df['Club'].apply(lambda _: random.choice(['Man United', 'PSG', 'Al Hilal', 'Barcelona']))
    df['Nationality'] = "Egyptian"
    df['Position'] = random.choices(['Forward', 'Midfielder', 'Defender'], k=len(df))
    df['League'] = "Custom Upload"
    df['Transfer_Chance'] = df['Market_Value_SAR'].apply(lambda x: random.uniform(0.6, 0.95))

    features = ['xG', 'Assists', 'Goals', 'Dribbles', 'Interceptions', 'PassingAccuracy', 'Market_Value_SAR']
    X = df[features]
    y = df['Market_Value_SAR'] * random.uniform(1.05, 1.15)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    df['Predicted_Year_1'] = model.predict(X)
    df['Predicted_Year_2'] = df['Predicted_Year_1'] * 1.05
    df['Predicted_Year_3'] = df['Predicted_Year_2'] * 1.05
    return df

# === UI Title & Instructions
st.title("⚽ Football Talent Evaluator")

st.markdown("""
### 📊 How to Use This App

1. **Upload a CSV file** with multiple players **OR**
2. **Fill out the form below** to evaluate a single player

#### Required Fields:
- Player Name, Team Name, Age, Goals, Assists, Dribbles, Interceptions, xG, Passing Accuracy, Minutes, Market Value (EUR)
""")

# === Manual Entry Form
with st.form("manual_input"):
    name = st.text_input("Player Name")
    team = st.text_input("Team Name")
    age = st.number_input("Age", 16, 45)
    goals = st.number_input("Goals", 0, 10)
    assists = st.number_input("Assists", 0, 10)
    dribbles = st.number_input("Dribbles", 0, 20)
    interceptions = st.number_input("Interceptions", 0, 10)
    xg = st.number_input("xG", 0.0, 5.0)
    passing = st.number_input("Passing Accuracy (%)", 0.0, 100.0)
    minutes = st.number_input("Minutes Played", 0, 120)
    market_val = st.number_input("Market Value (EUR)", 10000, 10000000, step=10000)
    submitted = st.form_submit_button("Evaluate Player")

df = None
if submitted:
    manual_df = pd.DataFrame([{
        'player_name': name,
        'team_name': team,
        'age': age,
        'player_match_goals': goals,
        'player_match_assists': assists,
        'player_match_dribbles': dribbles,
        'player_match_interceptions': interceptions,
        'player_match_np_xg': xg,
        'player_match_passing_ratio': passing,
        'player_match_minutes': minutes,
        'Market_Value_EUR': market_val
    }])
    df = prepare_data(manual_df)

# === Upload file block
uploaded_file = st.file_uploader("📁 Or upload a CSV file with multiple players", type=["csv"])
if uploaded_file and not submitted:
    df = prepare_data(pd.read_csv(uploaded_file))
    st.success("✅ File uploaded and processed.")

# === Main display
if df is not None and not df.empty:
    player = df.iloc[0]

    st.markdown(f"""
    <div class='card'>
        <h3>{player['Player Name']} ({player['Position']})</h3>
        <img src="{player['Image']}" width="100">
        <p><strong>Club:</strong> {player['Club']}</p>
        <p><strong>Age:</strong> {age}</p>
        <p><strong>Market Value:</strong> €{player['Market_Value_EUR']:,.0f} / {player['Market_Value_SAR']:,.0f} SAR</p>
        <p><strong>Asking Price:</strong> {player['Asking_Price_SAR']:,.0f} SAR</p>
        <p><strong>Predicted Next Year:</strong> {player['Predicted_Year_1']:,.0f} SAR</p>
        <p><strong>Verification:</strong> {player['Verification']}</p>
        <p><strong>Best Fit:</strong> {player['Best_Fit_Club']}</p>
    </div>
    """, unsafe_allow_html=True)

    forecast_df = pd.DataFrame({
        "Year": ["2024", "2025", "2026"],
        "Predicted Value (SAR)": [player['Predicted_Year_1'], player['Predicted_Year_2'], player['Predicted_Year_3']],
        "Club Asking Price (SAR)": [player['Asking_Price_SAR']] * 3
    })
    chart = alt.Chart(forecast_df).transform_fold(
        ['Predicted Value (SAR)', 'Club Asking Price (SAR)'],
        as_=['Metric', 'SAR Value']
    ).mark_line(point=True).encode(
        x='Year:N', y='SAR Value:Q', color='Metric:N'
    )
    st.altair_chart(chart, use_container_width=True)

    # === AI Commentary
    st.header("🧠 AI Sentiment Summary")
    comment = f"{player['Player Name']} is showing promise with growing consistency and impact."
    prompts = [{"role": "system", "content": "Classify and summarize this football player comment."},
               {"role": "user", "content": f"{comment}"}]
    try:
        response = client.chat.completions.create(model="gpt-4-1106-preview", messages=prompts)
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
