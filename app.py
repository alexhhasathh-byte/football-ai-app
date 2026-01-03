import streamlit as st
import numpy as np
import plotly.graph_objects as go
from math import exp, factorial

# --- إعدادات الهوية البصرية لـ Yaya Score ---
st.set_page_config(page_title="Yaya Score | AI Predictions", layout="wide", page_icon="⚽")

# تصميم واجهة احترافية مخصصة
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    .app-header {
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .yaya-logo { font-size: 45px; font-weight: bold; letter-spacing: 2px; }
    .prediction-box {
        background-color: #1e293b;
        border: 2px solid #3b82f6;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    </style>
    <div class="app-header">
        <div class="yaya-logo">YAYA SCORE</div>
        <p>المحرك الأقوى لتوقعات كرة القدم بالذكاء الاصطناعي</p>
    </div>
    """, unsafe_allow_html=True)

# --- محرك التحليل الإحصائي ---
def poisson_analysis(h_avg, a_avg):
    matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            prob = (exp(-h_avg) * (h_avg**i) / factorial(i)) * (exp(-a_avg) * (a_avg**j) / factorial(j))
            matrix[i, j] = prob
    return matrix

# --- القائمة الجانبية ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5323/5323483.png", width=100)
    st.title("Yaya Score Admin")
    st.info("مرحباً بك في لوحة تحكم Yaya Score. هذا النظام يحلل البيانات بناءً على قوة الهجوم والدفاع.")

# --- منطقة تحليل المباريات ---
col1, col2 = st.columns(2)

with col1:
    home_t = st.text_input("الفريق المضيف", "مانشستر يونايتد")
    h_power = st.slider("مستوى هجوم الأرض", 0.5, 5.0, 2.0)

with col2:
    away_t = st.text_input("الفريق الضيف", "تشيلسي")
    a_power = st.slider("مستوى هجوم الضيف", 0.5, 5.0, 1.5)

if st.button("إظهار توقع YAYA SCORE", use_container_width=True):
    m = poisson_analysis(h_power, a_power)
    
    # حساب الاحتمالات
    h_win = np.sum(np.tril(m, -1)) * 100
    draw = np.trace(m) * 100
    a_win = np.sum(np.triu(m, 1)) * 100
    
    st.markdown("---")
    
    # عرض النتيجة الأكثر توقعاً بشكل ضخم
    score = np.unravel_index(m.argmax(), m.shape)
    st.markdown(f"""
        <div class="prediction-box">
            <h3 style="color: #94a3b8;">النتيجة المتوقعة</h3>
            <h1 style="color: #fbbf24; font-size: 80px;">{score[0]} - {score[1]}</h1>
            <p style="color: #3b82f6;">ثقة النظام: {m.max()*100:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)

    # الرسوم البيانية
    fig = go.Figure(data=[go.Bar(
        x=[home_t, 'تعادل', away_t],
        y=[h_win, draw, a_win],
        marker_color=['#2563eb', '#64748b', '#dc2626']
    )])
    fig.update_layout(title="نسب احتمالات الفوز", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<p style='text-align: center; color: #475569;'>© 2026 Yaya Score - All Rights Reserved</p>", unsafe_allow_html=True)
