import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import random

# Настройка страницы
st.set_page_config(page_title="Вибродиагностика", layout="wide", page_icon="🔧")

# ============================================================
# БАЗА ДАННЫХ ПОДШИПНИКОВ
# ============================================================
BEARINGS_DB = {
    '6205':  {'n': 9,  'd': 7.94,  'D': 38.5,  'alpha': 0,  'type': 'Радиальный шариковый'},
    '6208':  {'n': 9,  'd': 12.70, 'D': 57.0,  'alpha': 0,  'type': 'Радиальный шариковый'},
    '6305':  {'n': 8,  'd': 11.00, 'D': 43.0,  'alpha': 0,  'type': 'Радиальный шариковый'},
    '6308':  {'n': 8,  'd': 15.08, 'D': 65.0,  'alpha': 0,  'type': 'Радиальный шариковый'},
    '6310':  {'n': 8,  'd': 18.00, 'D': 80.0,  'alpha': 0,  'type': 'Радиальный шариковый'},
    '6312':  {'n': 8,  'd': 22.00, 'D': 95.0,  'alpha': 0,  'type': 'Радиальный шариковый'},
    '22210': {'n': 28, 'd': 16.0,  'D': 68.0,  'alpha': 0,  'type': 'Сферический роликовый'},
    '22220': {'n': 28, 'd': 32.0,  'D': 138.0, 'alpha': 0,  'type': 'Сферический роликовый'},
    '30210': {'n': 18, 'd': 17.0,  'D': 68.0,  'alpha': 15, 'type': 'Конический роликовый'},
    '7014':  {'n': 20, 'd': 11.0,  'D': 90.0,  'alpha': 15, 'type': 'Угловой контактный'},
}

# ============================================================
# ФУНКЦИИ
# ============================================================
def calculate_frequencies(bearing_name, rpm):
    b = BEARINGS_DB[bearing_name]
    fr = rpm / 60.0
    d_D = b['d'] / b['D']
    cos_a = np.cos(np.radians(b['alpha']))
    bpfo = (b['n'] / 2) * fr * (1 - d_D * cos_a)
    bpfi = (b['n'] / 2) * fr * (1 + d_D * cos_a)
    bsf = (b['D'] / (2 * b['d'])) * fr * (1 - (d_D * cos_a)**2)
    ftf = (fr / 2) * (1 - d_D * cos_a)
    return {'fr': fr, 'BPFO': bpfo, 'BPFI': bpfi, 'BSF': bsf, 'FTF': ftf, '2xBSF': 2 * bsf}

def generate_fault_signal(rpm, bearing_name, defect_type, stage, duration=0.3, fs=50000, resonance_freq=2500):
    t = np.arange(0, duration, 1/fs)
    freqs = calculate_frequencies(bearing_name, rpm)
    fr = freqs['fr']
    signal_data = np.random.normal(0, 0.02, len(t))
    
    if defect_type == 'none':
        return t, signal_data, freqs
    
    if defect_type == 'BPFO': fault_freq, mod_freq = freqs['BPFO'], None
    elif defect_type == 'BPFI': fault_freq, mod_freq = freqs['BPFI'], fr
    elif defect_type == 'BSF': fault_freq, mod_freq = freqs['2xBSF'], 2 * freqs['FTF']
    elif defect_type == 'FTF': fault_freq, mod_freq = freqs['FTF'], fr
    else: return t, signal_data, freqs
    
    amp_dict = {1: 0.15, 2: 0.4, 3: 0.8, 4: 1.5}
    amplitude = amp_dict.get(stage, 0.5)
    mod_dict = {1: 0.1, 2: 0.3, 3: 0.6, 4: 0.85}
    mod_depth = mod_dict.get(stage, 0.3) if mod_freq else 0
    harm_dict = {1: 1, 2: 2, 3: 3, 4: 5}
    n_harmonics = harm_dict.get(stage, 2)
    
    for h in range(1, n_harmonics + 1):
        current_freq = fault_freq * h
        current_amp = amplitude / h
        impulse_times = np.arange(0, duration, 1.0 / current_freq)
        for imp_time in impulse_times:
            modulation = (1 + mod_depth * np.sin(2 * np.pi * mod_freq * imp_time)) if mod_freq else 1.0
            n_samples = int(0.003 * fs)
            impulse_t = np.arange(n_samples) / fs
            resonance = current_amp * modulation * np.sin(2 * np.pi * resonance_freq * impulse_t)
            decay = np.exp(-800 * impulse_t)
            start_idx = int(imp_time * fs)
            if start_idx + n_samples < len(signal_data):
                signal_data[start_idx:start_idx + n_samples] += resonance * decay
    return t, signal_data, freqs

def process_envelope(sig, fs, fc=2500, bandwidth='1/3_octave'):
    if bandwidth == '1/3 октавы':
        f_low, f_high = fc / 1.122, fc * 1.122
    else:
        f_low, f_high = fc / 1.414, fc * 1.414
    if f_high >= fs / 2: f_high = (fs / 2) * 0.9
    sos = signal.butter(4, [f_low, f_high], btype='band', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, sig)
    rectified = np.abs(filtered)
    sos_lp = signal.butter(4, min(1500, fs / 4), btype='low', fs=fs, output='sos')
    envelope = signal.sosfilt(sos_lp, rectified)
    N = len(envelope)
    yf = fft(envelope)
    xf = fftfreq(N, 1/fs)[:N//2]
    amplitude = 2.0/N * np.abs(yf[:N//2])
    mask = xf <= 1000
    return xf[mask], amplitude[mask], envelope

# ============================================================
# ИНТЕРФЕЙС STREAMLIT
# ============================================================
st.title("🔧 Виртуальный стенд вибродиагностики")
st.markdown("**Генератор спектров огибающей с дефектами подшипников**")

# Боковая панель
with st.sidebar:
    st.header("⚙️ Параметры")
    
    bearing = st.selectbox('🔩 Подшипник', list(BEARINGS_DB.keys()), index=3)
    st.caption(f"Тип: {BEARINGS_DB[bearing]['type']}")
    
    rpm = st.slider('🔄 Обороты (об/мин)', 300, 3600, 1500, 50)
    
    fc = st.selectbox('📡 Центральная частота fc (Гц)', 
                      [800, 1600, 2500, 4000, 5000, 6300], index=2)
    
    bandwidth = st.selectbox('🔧 Тип фильтра', 
                             ['1/3 октавы', '1 октава'], index=0)
    
    defect = st.selectbox('⚠️ Тип дефекта', 
                          ['none', 'BPFO', 'BPFI', 'BSF', 'FTF'], index=1)
    
    stage = st.slider('📈 Стадия дефекта', 1, 4, 2)

# ============================================================
# РАСЧЁТЫ
# ============================================================
freqs = calculate_frequencies(bearing, rpm)
t, sig, _ = generate_fault_signal(rpm, bearing, defect, stage, resonance_freq=fc)
xf_env, amp_env, envelope = process_envelope(sig, fs=50000, fc=fc, bandwidth=bandwidth)

# ============================================================
# ОТОБРАЖЕНИЕ
# ============================================================

# Информация о частотах
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Расчетные частоты")
    st.markdown(f"""
    **{bearing}** — {BEARINGS_DB[bearing]['type']}  
    Обороты: **{rpm} об/мин** ({freqs['fr']:.2f} Гц)
    
    | Параметр | Частота, Гц |
    |:---|---:|
    | BPFO (наружное) | **{freqs['BPFO']:.2f}** |
    | BPFI (внутреннее) | **{freqs['BPFI']:.2f}** |
    | BSF (тело качения) | **{freqs['BSF']:.2f}** |
    | FTF (сепаратор) | **{freqs['FTF']:.2f}** |
    | 2×BSF (дефект тела!) | **{freqs['2xBSF']:.2f}** |
    """)

with col2:
    st.subheader("📚 Шпаргалка")
    st.markdown("""
    **Признаки дефектов:**
    - **BPFO**: пики + гармоники, **НЕТ** боковых полос
    - **BPFI**: пик + боковые полосы (шаг = 1× оборотов)
    - **BSF**: пик на **2×BSF** + боковые полосы (шаг = 2×FTF)
    - **FTF**: очень низкая частота (~0.4× оборотов)
    
    💡 **Совет:** боковые полосы видны только в логарифмической шкале!
    """)

# Графики
st.subheader("📈 Графики анализа")

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
fig.suptitle(f'{bearing} | {rpm} об/мин | {defect} | Стадия {stage} | fc={fc} Гц | {bandwidth}', 
             fontsize=13, fontweight='bold')

# 1. Временной сигнал
axes[0].plot(t * 1000, sig, linewidth=0.3, color='steelblue')
axes[0].set_title('Исходный временной сигнал')
axes[0].set_ylabel('Ускорение, g')
axes[0].set_xlim(0, 50)
axes[0].grid(True, alpha=0.3)

# 2. Огибающая
axes[1].plot(t * 1000, envelope, linewidth=0.5, color='red')
axes[1].set_title(f'Огибающая (fc={fc} Гц, фильтр {bandwidth})')
axes[1].set_ylabel('Огибающая, g')
axes[1].set_xlim(0, 50)
axes[1].grid(True, alpha=0.3)

# 3. Линейный спектр
axes[2].plot(xf_env, amp_env, linewidth=0.8, color='darkgreen', label='Спектр')
axes[2].set_title('Спектр огибающей (линейная шкала)')
axes[2].set_ylabel('Амплитуда, g')
axes[2].set_xlim(0, 500)
axes[2].grid(True, alpha=0.3)

# 4. Логарифмический спектр
amp_db = 20 * np.log10(amp_env + 1e-10)
axes[3].plot(xf_env, amp_db, linewidth=0.8, color='purple', label='Спектр')
axes[3].set_title('Спектр огибающей (логарифмическая шкала) — видны боковые полосы!')
axes[3].set_xlabel('Частота, Гц')
axes[3].set_ylabel('Амплитуда, дБ')
axes[3].set_xlim(0, 500)
axes[3].set_ylim(-80, np.max(amp_db) + 5)  # Улучшенный диапазон
axes[3].grid(True, alpha=0.3)

# === ВЕРТИКАЛЬНЫЕ ЛИНИИ И ЛЕГЕНДА ДЛЯ ОБОИХ СПЕКТРОВ ===
colors = {
    'BPFO': 'red',
    'BPFI': 'blue', 
    'BSF': 'green',
    'FTF': 'orange',
    '2xBSF': 'magenta'
}

for ax_idx, ax in enumerate([axes[2], axes[3]]):
    for defect_name, color in colors.items():
        f_val = freqs.get(defect_name, 0)
        if f_val > 0 and f_val < 500:
            # Основная частота
            ax.axvline(x=f_val, color=color, linestyle='--', 
                      linewidth=1.5, alpha=0.7, 
                      label=f'{defect_name}={f_val:.1f} Гц')
            
            # Гармоники
            for h in [2, 3]:
                harmonic = f_val * h
                if harmonic < 500:
                    ax.axvline(x=harmonic, color=color, linestyle=':', 
                              linewidth=1, alpha=0.5)
    
    # Легенда на обоих графиках
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    
    # АННОТАЦИИ (подписи частот) — только на логарифмическом для наглядности
    if ax_idx == 1:  # Только для логарифмического
        for defect_name, color in colors.items():
            f_val = freqs.get(defect_name, 0)
            if f_val > 0 and f_val < 500:
                idx = np.argmin(np.abs(xf_env - f_val))
                amp_at_freq = amp_db[idx]
                
                # Подпись только если амплитуда значимая
                if amp_at_freq > -60:
                    ax.annotate(
                        f'{defect_name}\n{f_val:.1f} Гц',
                        xy=(f_val, amp_at_freq),
                        xytext=(5, 15),
                        textcoords='offset points',
                        fontsize=7,
                        color=color,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                 facecolor='white', alpha=0.8, 
                                 edgecolor=color)
                    )

plt.tight_layout()
st.pyplot(fig)

# ============================================================
# ИГРА "УГАДАЙ ДЕФЕКТ"
# ============================================================
st.markdown("---")
st.subheader("🎓 Игра 'Угадай дефект'")

if 'exam_data' not in st.session_state:
    st.session_state.exam_data = None
    st.session_state.show_answer = False

col1, col2 = st.columns(2)

with col1:
    if st.button("🎲 Новый вопрос", use_container_width=True):
        b = random.choice(list(BEARINGS_DB.keys()))
        r = random.choice([750, 1000, 1500, 2000, 3000])
        d = random.choice(['BPFO', 'BPFI', 'BSF', 'FTF'])
        s = random.randint(1, 4)
        f = random.choice([1600, 2500, 4000])
        bw = random.choice(['1/3 октавы', '1 октава'])
        
        st.session_state.exam_data = {
            'bearing': b, 'rpm': r, 'defect': d, 
            'stage': s, 'fc': f, 'bandwidth': bw,
            'freqs': calculate_frequencies(b, r)
        }
        st.session_state.show_answer = False
        st.rerun()

with col2:
    if st.button("👁️ Показать ответ", use_container_width=True, 
                 disabled=st.session_state.exam_data is None):
        st.session_state.show_answer = True
        st.rerun()

if st.session_state.exam_data:
    e = st.session_state.exam_data
    
    st.info(f"""
    **Экзаменационный вопрос:**
    - Подшипник: **{e['bearing']}** ({BEARINGS_DB[e['bearing']]['type']})
    - Обороты: **{e['rpm']} об/мин**
    - Центральная частота: **{e['fc']} Гц**
    - Фильтр: **{e['bandwidth']}**
    - Стадия: **{e['stage']}**
    
    **Определите тип дефекта по спектру ниже!**
    """)
    
    # График экзамена
    t_ex, sig_ex, _ = generate_fault_signal(
        e['rpm'], e['bearing'], e['defect'], e['stage'], 
        resonance_freq=e['fc']
    )
    xf_ex, amp_ex, _ = process_envelope(
        sig_ex, fs=50000, fc=e['fc'], bandwidth=e['bandwidth']
    )
    
    fig_ex, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    ax1.plot(xf_ex, amp_ex, linewidth=0.8, color='darkgreen')
    ax1.set_title('Линейный спектр — определите дефект')
    ax1.set_xlim(0, 500)
    ax1.grid(True, alpha=0.3)
    
    amp_db_ex = 20 * np.log10(amp_ex + 1e-10)
    ax2.plot(xf_ex, amp_db_ex, linewidth=0.8, color='purple')
    ax2.set_title('Логарифмический спектр — ищите боковые полосы!')
    ax2.set_xlabel('Частота, Гц')
    ax2.set_xlim(0, 500)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_ex)
    
    if st.session_state.show_answer:
        defect = e['defect']
        f = e['freqs']
        
        answers = {
            'BPFO': f"Частота: **{f['BPFO']:.2f} Гц**. Пики на BPFO + гармоники, НЕТ боковых полос",
            'BPFI': f"Частота: **{f['BPFI']:.2f} Гц**. Пик на BPFI + боковые полосы с шагом {f['fr']:.1f} Гц",
            'BSF': f"Частота: **{f['2xBSF']:.2f} Гц** (2×BSF!). Боковые полосы с шагом {2*f['FTF']:.2f} Гц",
            'FTF': f"Частота: **{f['FTF']:.2f} Гц** (очень низкая, ~0.4× оборотов)"
        }
        
        st.success(f"""
        ✅ **Правильный ответ: {defect}**
        
        {answers[defect]}
        
        Стадия {e['stage']}: {'начальная' if e['stage']==1 else 'ранняя' if e['stage']==2 else 'развитая' if e['stage']==3 else 'критическая'}
        """)

# Подвал
st.markdown("---")
st.caption("© Виртуальный стенд вибродиагностики | Метод спектра огибающей для СД-23")
