import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("ufc-master.csv")
    df = df.dropna(subset=['RedFighter', 'BlueFighter', 'Winner'])
    df = df[df['Winner'].isin(['Red', 'Blue'])]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

df = load_data()

# --- Feature Engineering ---
@st.cache_data
def add_fighter_features(df):
    fighter_stats = {}
    df_sorted = df.sort_values(by='Date')

    for fighter in pd.concat([df['RedFighter'], df['BlueFighter']]).unique():
        fights = df_sorted[(df_sorted['RedFighter'] == fighter) | (df_sorted['BlueFighter'] == fighter)]
        wins = []

        for _, row in fights.iterrows():
            if row['RedFighter'] == fighter:
                wins.append(row['Winner'] == 'Red')
            elif row['BlueFighter'] == fighter:
                wins.append(row['Winner'] == 'Blue')

        streak = 0
        for result in reversed(wins):
            if result:
                streak += 1
            else:
                break

        recent = wins[-3:]
        win_rate = sum(recent) / len(recent) if recent else 0

        fighter_stats[fighter] = {
            'WinStreak': streak,
            'RecentWinRate': win_rate,
            'TotalFights': len(fights)
        }

    df['RedWinStreak'] = df['RedFighter'].map(lambda x: fighter_stats.get(x, {}).get('WinStreak', 0))
    df['BlueWinStreak'] = df['BlueFighter'].map(lambda x: fighter_stats.get(x, {}).get('WinStreak', 0))

    df['RedRecentWinRate'] = df['RedFighter'].map(lambda x: fighter_stats.get(x, {}).get('RecentWinRate', 0))
    df['BlueRecentWinRate'] = df['BlueFighter'].map(lambda x: fighter_stats.get(x, {}).get('RecentWinRate', 0))

    df['RedTotalFights'] = df['RedFighter'].map(lambda x: fighter_stats.get(x, {}).get('TotalFights', 0))
    df['BlueTotalFights'] = df['BlueFighter'].map(lambda x: fighter_stats.get(x, {}).get('TotalFights', 0))
    
    return df

df = add_fighter_features(df)

# --- Encode Winner ---
label_enc = LabelEncoder()
df['WinnerEncoded'] = label_enc.fit_transform(df['Winner'])

# --- Features & Model ---
features = [
    'RedWinStreak', 'BlueWinStreak',
    'RedRecentWinRate', 'BlueRecentWinRate',
    'RedTotalFights', 'BlueTotalFights'
]

X = df[features]
y = df['WinnerEncoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = RandomForestClassifier(n_estimators=100, random_state=1)
model2 = GradientBoostingClassifier(n_estimators=100, random_state=1)
model3 = LogisticRegression()

ensemble = VotingClassifier(estimators=[
    ('rf', model1),
    ('gb', model2),
    ('lr', model3)
], voting='soft')

ensemble.fit(X_train, y_train)

# --- UI ---
st.title("🥋 UFC Fight Predictor")
st.write("Select two fighters to predict who would win based on historical data.")

fighters = sorted(set(df['RedFighter']).union(df['BlueFighter']))
fighter1 = st.selectbox("Fighter A", fighters)
fighter2 = st.selectbox("Fighter B", [f for f in fighters if f != fighter1])

def get_latest_stats(fighter_name):
    if fighter_name not in df['RedFighter'].values and fighter_name not in df['BlueFighter'].values:
        return {'WinStreak': 0, 'RecentWinRate': 0.0, 'TotalFights': 0}
    return {
        'WinStreak': df[df['RedFighter'] == fighter_name]['RedWinStreak'].iloc[-1]
                    if fighter_name in df['RedFighter'].values else df[df['BlueFighter'] == fighter_name]['BlueWinStreak'].iloc[-1],
        'RecentWinRate': df[df['RedFighter'] == fighter_name]['RedRecentWinRate'].iloc[-1]
                        if fighter_name in df['RedFighter'].values else df[df['BlueFighter'] == fighter_name]['BlueRecentWinRate'].iloc[-1],
        'TotalFights': df[df['RedFighter'] == fighter_name]['RedTotalFights'].iloc[-1]
                      if fighter_name in df['RedFighter'].values else df[df['BlueFighter'] == fighter_name]['BlueTotalFights'].iloc[-1]
    }

if st.button("Predict Winner"):
    stats1 = get_latest_stats(fighter1)
    stats2 = get_latest_stats(fighter2)

    input_data = pd.DataFrame([{
        'RedWinStreak': stats1['WinStreak'],
        'BlueWinStreak': stats2['WinStreak'],
        'RedRecentWinRate': stats1['RecentWinRate'],
        'BlueRecentWinRate': stats2['RecentWinRate'],
        'RedTotalFights': stats1['TotalFights'],
        'BlueTotalFights': stats2['TotalFights']
    }])

    prediction_proba = ensemble.predict_proba(input_data)[0]
    predicted_label = ensemble.predict(input_data)[0]
    predicted_color = label_enc.inverse_transform([predicted_label])[0]

    confidence = np.max(prediction_proba) * 100
    winner_name = fighter1 if predicted_color == 'Red' else fighter2

    st.success(f"🏆 **Predicted Winner:** {winner_name}")
    st.info(f"🔮 **Confidence:** {confidence:.2f}%")
