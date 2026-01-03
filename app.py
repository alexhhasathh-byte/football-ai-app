import streamlit as st
import requests
import pandas as pd
import numpy as np
from math import exp, factorial
from sklearn.ensemble import RandomForestClassifier

# --- 1. محرك البيانات (Data Layer) ---
class FootballEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {'x-rapidapi-key': api_key, 'x-rapidapi-host': "v3.football.api-sports.io"}
        self.url = "https://v3.football.api-sports.io"

    def get_team_stats(self, team_id, league_id):
        # في الواقع، هذا الرابط يجلب البيانات الحقيقية من الـ API
        # لغرض العرض، سنقوم بمحاكاة البيانات إذا لم يتوفر مفتاح API فعال
        try:
            endpoint = f"{self.url}/teams/statistics?league={league_id}&season=2025&team={team_id}"
            res = requests.get(endpoint, headers=self.headers).json()
            avg_goals = res['response']['goals']['for']['average']['total']
            return float(avg_goals)
        except:
            return 1.5 # قيمة افتراضية

# --- 2. محرك الذكاء الاصطناعي (AI Layer) ---
def train_ai_logic():
    # بيانات تدريبية نموذجية (أهداف، أيام راحة، استحواذ -> نتيجة)
    data = {
        'avg_goals': [2.5, 1.0, 0.5, 3.0, 1.2, 2.0, 0.8, 1.7],
        'rest_days': [5, 2, 3, 7, 4, 3, 2, 5],
        'possession': [60, 40, 35, 65, 45, 55, 30, 50],
        'outcome': [2, 0, 0, 2, 1, 2, 0, 1] # 2: Win, 1: Draw, 0: Loss
    }
    df = pd.DataFrame(data)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(df[['avg_goals', 'rest_days', 'possession']], df['outcome'])
    return model

# --- 3. محرك بويسان للنتائج الدقيقة (Math Layer) ---
def poisson_prob(actual, average):
    return (exp(-average) * (average**actual)) / factorial(actual)

def get_exact_score_matrix(h_avg, a_avg):
    matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            matrix[i, j] = poisson_prob(i, h_avg) * poisson_prob(j, a_avg)
    return matrix

# --- 4. واجهة التطبيق (UI Layer) ---
st.set_page_config(page_title="AI Sports Predictor Pro", layout="wide")

st.title("⚽ المحلل الرياضي الخارق (AI + Poisson)")
st.sidebar.header("⚙️ الإعدادات والتحكم")

api_key = st.sidebar.text_input("أدخل API Key الخاص بك", type="password")
selected_league = st.sidebar.selectbox("اختر الدوري", ["الدوري الإنجليزي (39)", "الدوري الإسباني (140)", "الدوري السعودي (307)"])

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 الفريق المستضيف")
    h_id = st.text_input("ID الفريق الأرضي", "40")
    h_rest = st.slider("أيام الراحة (للأرض)", 1, 10, 5)
    h_pos = st.slider("متوسط الاستحواذ % (الأرض)", 30, 70, 50)

with col2:
    st.subheader("🚀 الفريق الضيف")
    a_id = st.text_input("ID الفريق الضيف", "33")
    a_rest = st.slider("أيام الراحة (للضيف)", 1, 10, 5)
    a_pos = st.slider("متوسط الاستحواذ % (للضيف)", 30, 70, 50)

if st.button("🔥 تشغيل التحليل العميق"):
    if not api_key:
        st.error("يرجى إدخال مفتاح الـ API أولاً!")
    else:
        engine = FootballEngine(api_key)
        ai_model = train_ai_logic()
        
        with st.spinner('جاري معالجة البيانات وتحليل الأنماط...'):
            # جلب البيانات
            h_avg = engine.get_team_stats(h_id, 39)
            a_avg = engine.get_team_stats(a_id, 39)
            
            # 1. تحليل بويسان (الإحصائي)
            matrix = get_exact_score_matrix(h_avg, a_avg)
            h_win_p = np.sum(np.tril(matrix, -1))
            a_win_p = np.sum(np.triu(matrix, 1))
            draw_p = np.trace(matrix)
            
            # 2. تحليل الذكاء الاصطناعي (النمطي)
            ai_input = [[h_avg, h_rest, h_pos]]
            ai_probs = ai_model.predict_proba(ai_input)[0]

            # عرض النتائج
            st.markdown("---")
            res1, res2 = st.columns(2)
            
            with res1:
                st.header("📊 التوقع الإحصائي (Poisson)")
                st.write(f"فوز الأرض: {h_win_p:.1%}")
                st.write(f"تعادل: {draw_p:.1%}")
                st.write(f"فوز الضيف: {a_win_p:.1%}")
                
            with res2:
                st.header("🧠 توقع الذكاء الاصطناعي")
                st.write(f"ثقة الفوز: {ai_probs[2]:.1%}")
                st.write(f"ثقة التعادل: {ai_probs[1]:.1%}")
                st.write(f"ثقة الخسارة: {ai_probs[0]:.1%}")

            # مصفوفة النتائج الدقيقة
            st.subheader("🎯 مصفوفة النتائج الأكثر احتمالية")
            df_m = pd.DataFrame(matrix * 100, index=[f"{i}" for i in range(5)], columns=[f"{i}" for i in range(5)])
            st.dataframe(df_m.style.background_gradient(cmap='Greens'))
            
            best_score = np.unravel_index(matrix.argmax(), matrix.shape)
            st.success(f"✅ النتيجة المقترحة: {best_score[0]} - {best_score[1]}")