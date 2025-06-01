import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# --- Load and Clean Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("ufc-master.csv")

    # Rename columns
    df.rename(columns={
        'RedHeightCms': 'RedHeight',
        'BlueHeightCms': 'BlueHeight',
        'RedReachCms': 'RedReach',
        'BlueReachCms': 'BlueReach',
        'RedWeightLbs': 'RedWeight',
        'BlueWeightLbs': 'BlueWeight'
    }, inplace=True)

    df = df.dropna(subset=['RedFighter', 'BlueFighter', 'Winner'])
    df = df[df['Winner'].isin(['Red', 'Blue'])]

    required_columns = [
        'RedHeight', 'BlueHeight', 'RedReach', 'BlueReach',
        'RedAge', 'BlueAge', 'RedStance', 'BlueStance', 'WeightClass'
    ]
    df = df.dropna(subset=required_columns)
    return df

# --- Feature Engineering ---
def add_fighter_features(df):
    fighter_stats = {}
    df_sorted = df.sort_values(by='Date', ascending=True)

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

# --- Load Data and Engineer Features ---
df = load_data()
df = add_fighter_features(df)

stance_map = {stance: i for i, stance in enumerate(df['RedStance'].dropna().unique())}
df['RedStanceNum'] = df['RedStance'].map(stance_map)
df['BlueStanceNum'] = df['BlueStance'].map(stance_map)

df['HeightDiff'] = df['RedHeight'] - df['BlueHeight']
df['ReachDiff'] = df['RedReach'] - df['BlueReach']
df['AgeDiff'] = df['RedAge'] - df['BlueAge']
df['StanceDiff'] = df['RedStanceNum'] - df['BlueStanceNum']

weight_dummies = pd.get_dummies(df['WeightClass'], prefix='Weight')
df = pd.concat([df, weight_dummies], axis=1)

label_enc = LabelEncoder()
df['WinnerEncoded'] = label_enc.fit_transform(df['Winner'])

features = [
    'RedWinStreak', 'BlueWinStreak',
    'RedRecentWinRate', 'BlueRecentWinRate',
    'RedTotalFights', 'BlueTotalFights',
    'HeightDiff', 'ReachDiff', 'AgeDiff', 'StanceDiff'
] + list(weight_dummies.columns)

X = df[features]
y = df['WinnerEncoded']

imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

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
accuracy = accuracy_score(y_test, ensemble.predict(X_test))

# --- Streamlit UI ---
st.title("🥋 UFC Fight Predictor")
st.markdown(f"Model Accuracy: **{accuracy:.2%}**")

fighter_list = sorted(set(df['RedFighter'].unique()).union(df['BlueFighter'].unique()))
fighter1 = st.selectbox("Select Fighter A", fighter_list)
fighter2 = st.selectbox("Select Fighter B", [f for f in fighter_list if f != fighter1])

def get_latest_stats(fighter_name):
    row = df[(df['RedFighter'] == fighter_name) | (df['BlueFighter'] == fighter_name)].iloc[-1]
    stance_num = stance_map.get(row['RedStance'] if row['RedFighter'] == fighter_name else row['BlueStance'], 0)
    return {
        'WinStreak': row['RedWinStreak'] if row['RedFighter'] == fighter_name else row['BlueWinStreak'],
        'RecentWinRate': row['RedRecentWinRate'] if row['RedFighter'] == fighter_name else row['BlueRecentWinRate'],
        'TotalFights': row['RedTotalFights'] if row['RedFighter'] == fighter_name else row['BlueTotalFights'],
        'Height': row['RedHeight'] if row['RedFighter'] == fighter_name else row['BlueHeight'],
        'Reach': row['RedReach'] if row['RedFighter'] == fighter_name else row['BlueReach'],
        'Age': row['RedAge'] if row['RedFighter'] == fighter_name else row['BlueAge'],
        'Stance': stance_num,
        'WeightClass': row['WeightClass']
    }

if st.button("Predict Winner"):
    stats1 = get_latest_stats(fighter1)
    stats2 = get_latest_stats(fighter2)

    input_row = {
        'RedWinStreak': stats1['WinStreak'],
        'BlueWinStreak': stats2['WinStreak'],
        'RedRecentWinRate': stats1['RecentWinRate'],
        'BlueRecentWinRate': stats2['RecentWinRate'],
        'RedTotalFights': stats1['TotalFights'],
        'BlueTotalFights': stats2['TotalFights'],
        'HeightDiff': stats1['Height'] - stats2['Height'],
        'ReachDiff': stats1['Reach'] - stats2['Reach'],
        'AgeDiff': stats1['Age'] - stats2['Age'],
        'StanceDiff': stats1['Stance'] - stats2['Stance'],
    }

    for col in weight_dummies.columns:
        input_row[col] = 1 if stats1['WeightClass'] in col else 0

    input_df = pd.DataFrame([input_row])
    input_df = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)

    prediction_proba = ensemble.predict_proba(input_df)[0]
    predicted_label = ensemble.predict(input_df)[0]
    predicted_color = label_enc.inverse_transform([predicted_label])[0]
    confidence = np.max(prediction_proba) * 100
    predicted_name = fighter1 if predicted_color == 'Red' else fighter2

    st.subheader("🏆 Prediction")
    st.success(f"**Predicted Winner:** {predicted_name}")
    st.markdown(f"🔮 **Confidence:** {confidence:.2f}%")
