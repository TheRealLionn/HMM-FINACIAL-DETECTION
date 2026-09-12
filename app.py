"""
Phase 4 — Streamlit Dashboard / Tableau de bord Streamlit
Sprint Tasks 4.1-4.4 (+ bonus "Conseils & Prévisions" section)

EN: Reads the artefacts Phase 3 produces (run `python -m src.model` first):
        models/gaussian_hmm_3state.joblib   <- trained model + label map
        data/processed/regimes.csv          <- decoded regime per trading day
    This file does NOT re-fit the model — training happens once in
    src/model.py; the dashboard only loads and visualises the result, so
    opening the app is instant and every user sees the same regimes.

FR: Charge les artefacts produits par la Phase 3 (lancez d'abord
    `python -m src.model`) :
        models/gaussian_hmm_3state.joblib   <- modèle entraîné + label map
        data/processed/regimes.csv          <- régime décodé par jour
    Ce fichier ne ré-entraîne jamais le modèle — l'entraînement se fait une
    seule fois dans src/model.py ; le dashboard se contente de charger et
    d'afficher le résultat, donc l'app s'ouvre instantanément et tout le
    monde voit les mêmes régimes.

Run / Lancer :  streamlit run app.py   (from the project root / racine du projet)

NOTE (Windows console encoding):
EN: If you see UnicodeEncodeError in a Windows terminal, it comes from
    print() statements elsewhere (e.g. src/model.py), NOT from this file --
    everything here is served as HTML to the browser, which is always UTF-8
    safe regardless of the console's code page.
FR: Si vous voyez une UnicodeEncodeError dans un terminal Windows, elle vient
    d'un print() ailleurs (ex. src/model.py), PAS de ce fichier — tout ici
    est envoyé en HTML au navigateur, qui est toujours compatible UTF-8
    quel que soit l'encodage de la console.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.model import REGIMES_CSV

# ===========================================================================
# Bilingual regime metadata (color, glow, emoji, EN + FR explanations)
# EN: single source of truth so every widget (chart, card, table, advice
#     section) uses the exact same colors and wording.
# FR: source unique pour que chaque widget (graphique, carte, tableau,
#     section conseils) utilise exactement les mêmes couleurs et le même
#     texte.
# ===========================================================================
REGIME_INFO = {
    "Bull": {
        "emoji": "\U0001F7E2",
        "color": "#00FFA3",
        "glow": "rgba(0, 255, 163, 0.16)",
        "en": ("Bull market: prices are broadly rising and volatility tends "
               "to be low. Historically a period of investor confidence."),
        "fr": ("Marché haussier : les prix montent globalement et la "
               "volatilité est plutôt basse. Historiquement, une période de "
               "confiance des investisseurs."),
    },
    "Bear": {
        "emoji": "\U0001F534",
        "color": "#FF3366",
        "glow": "rgba(255, 51, 102, 0.18)",
        "en": ("Bear market: prices are broadly falling, often with sharper "
               "swings (higher volatility). Historically linked to "
               "uncertainty or crisis periods."),
        "fr": ("Marché baissier : les prix baissent globalement, souvent "
               "avec des mouvements plus brusques (volatilité plus élevée). "
               "Historiquement associé à des périodes d'incertitude ou de "
               "crise."),
    },
    "Sideways": {
        "emoji": "\U0001F7E1",
        "color": "#FFD23F",
        "glow": "rgba(255, 210, 63, 0.16)",
        "en": ("Sideways market: prices oscillate without a clear trend — "
               "neither a real rally nor a real decline."),
        "fr": ("Marché neutre : les prix oscillent sans tendance claire, ni "
               "vraie hausse ni vraie baisse marquée."),
    },
}
CYAN = "#00F0FF"
MAGENTA = "#B026FF"
BG_DEEP = "#05070D"
TEXT_MAIN = "#E6F1FF"
TEXT_DIM = "#7C8DB5"


def bi(en: str, fr: str, sep: str = "  /  ") -> str:
    """EN: Join an English and a French label into one bilingual string.
    FR: Combine un libellé anglais et français en une seule chaîne bilingue."""
    return f"{en}{sep}{fr}"


# ===========================================================================
# Theme — tech / futuristic CSS injection
# EN: pure CSS + Google Fonts, no external JS framework — safe to inject via
#     st.markdown(unsafe_allow_html=True) once at the top of the page.
# FR: uniquement du CSS + Google Fonts, aucun framework JS externe — sûr à
#     injecter via st.markdown(unsafe_allow_html=True) une fois en haut de
#     la page.
# ===========================================================================
def inject_theme() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText {{
        font-family: 'JetBrains Mono', monospace;
    }}
    h1, h2, h3, h4 {{ font-family: 'Orbitron', sans-serif !important; letter-spacing: 0.4px; }}

    .stApp {{
        background:
            radial-gradient(circle at 12% 8%, rgba(0,240,255,0.07), transparent 42%),
            radial-gradient(circle at 88% 92%, rgba(176,38,255,0.09), transparent 46%),
            repeating-linear-gradient(0deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 3px),
            {BG_DEEP};
        color: {TEXT_MAIN};
    }}

    .gradient-title {{
        font-size: 2.4rem; font-weight: 900; margin-bottom: 0;
        background: linear-gradient(90deg, {CYAN}, {MAGENTA}, {CYAN});
        background-size: 200% auto;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: shine 7s linear infinite;
    }}
    @keyframes shine {{ to {{ background-position: 200% center; }} }}

    .hud-line {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
        color: {TEXT_DIM}; letter-spacing: 1px; margin-top: 2px;
    }}
    .pulse-dot {{
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: {CYAN}; margin-right: 6px; box-shadow: 0 0 8px {CYAN};
        animation: pulse 1.6s ease-in-out infinite;
    }}
    @keyframes pulse {{
        0%   {{ opacity: 1;   transform: scale(1); }}
        50%  {{ opacity: 0.35; transform: scale(1.5); }}
        100% {{ opacity: 1;   transform: scale(1); }}
    }}

    .glow-card {{
        background: rgba(15, 20, 34, 0.72);
        border: 1px solid rgba(0, 240, 255, 0.28);
        border-radius: 14px;
        padding: 1.05rem 1.3rem;
        box-shadow: 0 0 22px rgba(0, 240, 255, 0.08), inset 0 0 24px rgba(0, 240, 255, 0.03);
        backdrop-filter: blur(10px);
        height: 100%;
    }}
    .kpi-label {{ font-size: 0.72rem; color: {TEXT_DIM}; letter-spacing: 0.5px; text-transform: uppercase; }}
    .kpi-value {{ font-family: 'Orbitron', sans-serif; font-size: 1.65rem; font-weight: 700; color: {CYAN}; }}

    .regime-card {{
        border-radius: 14px; padding: 1rem 1.2rem; backdrop-filter: blur(8px);
        border: 1px solid var(--rc-color); background: var(--rc-glow);
        box-shadow: 0 0 20px var(--rc-glow);
    }}
    .regime-bar-track {{ background: rgba(255,255,255,0.08); border-radius: 6px; height: 8px; overflow: hidden; margin-top: 4px; }}
    .regime-bar-fill {{ height: 100%; border-radius: 6px; }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #070A12, #05070D);
        border-right: 1px solid rgba(0, 240, 255, 0.15);
    }}
    section[data-testid="stSidebar"] * {{ color: {TEXT_MAIN}; }}

    [data-testid="stMetric"] {{
        background: rgba(15, 20, 34, 0.72);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 12px; padding: 12px 16px;
        box-shadow: 0 0 16px rgba(0, 240, 255, 0.08);
    }}
    [data-testid="stMetricValue"] {{ color: {CYAN} !important; font-family: 'Orbitron', sans-serif !important; }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_DIM} !important; }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 10px; overflow: hidden;
    }}

    hr {{ border-color: rgba(0, 240, 255, 0.2) !important; }}
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: {BG_DEEP}; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(0, 240, 255, 0.35); border-radius: 6px; }}
    </style>
    """, unsafe_allow_html=True)


# ===========================================================================
# Task 4.2 — load the trained model + decoded regimes
# ===========================================================================
@st.cache_resource
def load_model_bundle() -> dict:
    return joblib.load(config.MODEL_FILE)


@st.cache_data
def load_regimes() -> pd.DataFrame:
    return pd.read_csv(REGIMES_CSV, index_col="Date", parse_dates=True)


def _regime_segments(regimes: pd.DataFrame) -> list[dict]:
    """EN: Collapse the day-by-day regime column into contiguous runs — a
    vrect per run, not per day, is what makes the background actually show
    each regime's DURATION.
    FR: Regroupe la colonne régime jour par jour en segments continus — un
    vrect par segment (pas par jour) donne un fond qui montre vraiment la
    DURÉE de chaque régime."""
    segments, current_regime, seg_start, prev_date = [], None, None, None
    for date, regime in regimes["regime"].items():
        if regime != current_regime:
            if current_regime is not None:
                segments.append({"start": seg_start, "end": prev_date, "regime": current_regime})
            current_regime, seg_start = regime, date
        prev_date = date
    if current_regime is not None:
        segments.append({"start": seg_start, "end": prev_date, "regime": current_regime})
    return segments


def _add_glow_line(fig: go.Figure, x, y, color: str, name: str) -> None:
    """EN: Fake a neon glow by stacking wider, fainter copies of the same
    line underneath a crisp top line.
    FR: Simule un effet de halo néon en empilant des copies plus larges et
    plus transparentes de la même ligne sous une ligne nette au-dessus."""
    for width, opacity in ((9, 0.05), (6, 0.10), (3, 0.22)):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width),
                                  opacity=opacity, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=1.6),
                              name=name, hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>"))


def _dark_layout(fig: go.Figure, **kwargs) -> None:
    xaxis = {"gridcolor": "rgba(255,255,255,0.06)", "zeroline": False}
    xaxis.update(kwargs.pop("xaxis", {}) or {})
    yaxis = {"gridcolor": "rgba(255,255,255,0.06)", "zeroline": False}
    yaxis.update(kwargs.pop("yaxis", {}) or {})
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,7,13,0.35)",
        font=dict(family="JetBrains Mono, monospace", color=TEXT_MAIN, size=12),
        margin=dict(t=40, b=40, l=10, r=10),
        xaxis=xaxis,
        yaxis=yaxis,
        **kwargs,
    )


# ===========================================================================
# Task 4.3 — interactive time-series chart with regime-colored background
# ===========================================================================
def build_price_figure(regimes: pd.DataFrame, show_split_line: bool) -> go.Figure:
    fig = go.Figure()
    for seg in _regime_segments(regimes):
        info = REGIME_INFO.get(seg["regime"])
        fig.add_vrect(x0=seg["start"], x1=seg["end"],
                      fillcolor=info["glow"] if info else "rgba(150,150,150,0.10)",
                      line_width=0, layer="below")

    _add_glow_line(fig, regimes.index, regimes["price"], CYAN, bi("S&P 500", "S&P 500"))

    train_dates = regimes.index[regimes["split"] == "train"]
    if show_split_line and len(train_dates):
        fig.add_vline(x=train_dates.max(), line_dash="dash", line_color=MAGENTA, opacity=0.7,
                      annotation_text=bi("train / test split", "split entraînement / test"),
                      annotation_font_color=TEXT_DIM, annotation_position="top")

    for regime, info in REGIME_INFO.items():
        if regime in regimes["regime"].unique():
            fig.add_trace(go.Scatter(x=[regimes.index[0]], y=[regimes["price"].iloc[0]],
                                      mode="markers", marker=dict(size=10, color=info["color"], opacity=0.7),
                                      name=f"{info['emoji']} {regime}", showlegend=True, hoverinfo="skip"))

    _dark_layout(fig, title=f"{config.TICKER} — {bi('decoded market regime', 'régime de marché décodé')}",
                 height=520, hovermode="x unified",
                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def build_volatility_figure(regimes: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for seg in _regime_segments(regimes):
        info = REGIME_INFO.get(seg["regime"])
        fig.add_vrect(x0=seg["start"], x1=seg["end"],
                      fillcolor=info["glow"] if info else "rgba(150,150,150,0.10)",
                      line_width=0, layer="below")
    _add_glow_line(fig, regimes.index, regimes["volatility"], "#FF8A3D",
                   bi("volatility", "volatilité"))
    _dark_layout(fig, height=220, showlegend=False,
                 yaxis=dict(title=f"{config.VOLATILITY_WINDOW}d {bi('rolling volatility', 'volatilité roulante')}",
                            gridcolor="rgba(255,255,255,0.06)"))
    return fig


def build_transition_heatmap(model, label_map: dict) -> go.Figure:
    """EN: Neon heatmap of the learned transition matrix.
    FR: Heatmap néon de la matrice de transition apprise."""
    order = sorted(label_map)
    labels = [label_map[s] for s in order]
    matrix = model.transmat_[np.ix_(order, order)]
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=labels, y=labels,
        colorscale=[[0, "#05070D"], [0.5, "#3A1C71"], [1, CYAN]],
        text=[[f"{v:.1%}" for v in row] for row in matrix],
        texttemplate="%{text}", textfont=dict(family="JetBrains Mono, monospace", size=13, color=TEXT_MAIN),
        showscale=False, xgap=4, ygap=4,
    ))
    _dark_layout(fig, height=260,
                 title=bi("Transition probabilities", "Probabilités de transition"),
                 yaxis=dict(autorange="reversed"))
    return fig


# ===========================================================================
# Task 4.4 — sidebar controls
# ===========================================================================
def sidebar_controls(regimes: pd.DataFrame):
    st.sidebar.markdown(f"""<div class="hud-line"><span class="pulse-dot"></span>
        {bi('SYSTEM ONLINE', 'SYSTÈME EN LIGNE')}</div>""", unsafe_allow_html=True)
    st.sidebar.header(bi("Controls", "Contrôles"))

    min_date, max_date = regimes.index.min().date(), regimes.index.max().date()
    date_range = st.sidebar.date_input(
        bi("Date range", "Plage de dates"), value=(min_date, max_date),
        min_value=min_date, max_value=max_date,
    )
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    regime_filter = st.sidebar.multiselect(
        bi("Regimes", "Régimes"), options=list(REGIME_INFO.keys()),
        default=list(REGIME_INFO.keys()),
    )
    show_split_line = st.sidebar.checkbox(bi("Show train/test split", "Afficher le split train/test"), value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{bi('Legend', 'Légende')}**")
    for regime, info in REGIME_INFO.items():
        if regime in regimes["regime"].unique():
            st.sidebar.markdown(
                f'<span style="color:{info["color"]}; text-shadow:0 0 6px {info["color"]};">&#9632;</span> '
                f'{info["emoji"]} {regime}',
                unsafe_allow_html=True,
            )

    return pd.Timestamp(start_date), pd.Timestamp(end_date), regime_filter, show_split_line


# ===========================================================================
# KPI + regime summary cards (custom HTML for full "glow" control)
# ===========================================================================
def kpi_card(col, label_en: str, label_fr: str, value: str) -> None:
    col.markdown(f"""
    <div class="glow-card">
        <div class="kpi-label">{bi(label_en, label_fr)}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def regime_summary_card(col, regime: str, info: dict, stats_row: dict) -> None:
    col.markdown(f"""
    <div class="regime-card" style="--rc-color:{info['color']}; --rc-glow:{info['glow']};">
        <div style="font-family:'Orbitron',sans-serif; font-size:1.05rem; color:{info['color']};">
            {info['emoji']} {regime}
        </div>
        <div style="font-size:0.78rem; color:{TEXT_DIM}; margin:6px 0 10px 0;">
            {bi('Mean daily return', 'Rendement quotidien moyen')}: <b style="color:{TEXT_MAIN};">{stats_row['ret']}</b><br>
            {bi('Mean volatility', 'Volatilité moyenne')}: <b style="color:{TEXT_MAIN};">{stats_row['vol']}</b><br>
            {bi('Expected duration', 'Durée attendue')}: <b style="color:{TEXT_MAIN};">{stats_row['dur']}</b>
        </div>
        <div class="kpi-label">{bi('Share of sample', "Part de l'échantillon")}: {stats_row['share']}</div>
        <div class="regime-bar-track"><div class="regime-bar-fill" style="width:{stats_row['share_pct']}%; background:{info['color']}; box-shadow:0 0 8px {info['color']};"></div></div>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# NEW — Conseils & Prévisions / Advice & Forecast
# EN: plain-language regime explanation + a simple forecast read directly
#     off the learned transition matrix (geometric-distribution expected
#     duration, most likely next regime). Purely descriptive of what the
#     model already learned — not a market prediction.
# FR: explication en langage simple + une prévision simple lue directement
#     dans la matrice de transition apprise (durée attendue via une loi
#     géométrique, régime suivant le plus probable). Ceci décrit seulement
#     ce que le modèle a appris — ce n'est pas une prédiction de marché.
# ===========================================================================
def render_advice_section(regimes: pd.DataFrame, model, label_map: dict) -> None:
    st.divider()
    st.markdown(f"## \U0001F4E1 {bi('Advice & Forecast', 'Conseils & Prévisions')}")

    current_regime = regimes["regime"].iloc[-1]
    order = sorted(label_map)
    labels = [label_map[s] for s in order]
    trans_df = pd.DataFrame(model.transmat_[np.ix_(order, order)], index=labels, columns=labels)

    p_stay = trans_df.loc[current_regime, current_regime]
    p_switch = 1 - p_stay
    expected_remaining_days = 1 / p_switch if p_switch > 0 else float("inf")
    other = trans_df.loc[current_regime].drop(current_regime)
    next_likely_regime = other.idxmax()
    next_likely_prob = other.max()
    info = REGIME_INFO[current_regime]

    col_main, col_matrix = st.columns([2, 1])

    with col_main:
        col_main.markdown(f"""
        <div class="regime-card" style="--rc-color:{info['color']}; --rc-glow:{info['glow']};">
            <div style="font-family:'Orbitron',sans-serif; font-size:1.3rem; color:{info['color']};">
                {info['emoji']} {bi('Current regime', 'Régime actuel')}: {current_regime}
            </div>
            <p style="color:{TEXT_MAIN}; margin-top:10px;"><b>EN</b> — {info['en']}</p>
            <p style="color:{TEXT_MAIN};"><b>FR</b> — {info['fr']}</p>
            <hr style="margin:10px 0;">
            <ul style="color:{TEXT_MAIN}; line-height:1.7;">
                <li>{bi('Probability of staying in', 'Probabilité de rester en')} <b>{current_regime}</b> {bi('tomorrow', 'demain')}:
                    <b style="color:{CYAN};">{p_stay:.1%}</b></li>
                <li>{bi('Probability of switching regime tomorrow', 'Probabilité de changer de régime demain')}:
                    <b style="color:{CYAN};">{p_switch:.1%}</b></li>
                <li>{bi('Expected time before a regime change', 'Durée attendue avant un changement de régime')}:
                    <b style="color:{CYAN};">\u2248 {expected_remaining_days:.0f} {bi('trading days', 'jours de bourse')}</b></li>
                <li>{bi('If it changes, the most likely next regime is', 'Si le régime change, le plus probable ensuite est')}:
                    <b style="color:{REGIME_INFO[next_likely_regime]['color']};">{REGIME_INFO[next_likely_regime]['emoji']} {next_likely_regime}</b>
                    ({next_likely_prob:.1%} {bi('among the alternative regimes', 'parmi les régimes alternatifs')})</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.caption("\u26A0\uFE0F " + bi(
            "These figures come only from the transition matrix learned on historical data "
            "(how often regimes changed in the past). This is not a guaranteed prediction of "
            "future markets, and not financial advice.",
            "Ces chiffres viennent uniquement de la matrice de transition apprise sur les "
            "données historiques (fréquence des changements de régime passés). Ce n'est pas "
            "une prédiction garantie du marché futur, ni un conseil financier."
        ))

    with col_matrix:
        st.plotly_chart(build_transition_heatmap(model, label_map), use_container_width=True)

    st.markdown(f"#### {bi('Quick reference — all 3 regimes', 'Pour rappel — les 3 régimes')}")
    cols = st.columns(3)
    for col, (regime, r_info) in zip(cols, REGIME_INFO.items()):
        highlight = f" ({bi('current', 'actuel')})" if regime == current_regime else ""
        col.markdown(f"""
        <div class="regime-card" style="--rc-color:{r_info['color']}; --rc-glow:{r_info['glow']};">
            <b style="color:{r_info['color']};">{r_info['emoji']} {regime}{highlight}</b>
            <p style="font-size:0.8rem; color:{TEXT_DIM}; margin-top:6px;">
                <b>EN</b> {r_info['en']}<br><b>FR</b> {r_info['fr']}
            </p>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# Task 4.1 — page scaffold
# ===========================================================================
def main() -> None:
    st.set_page_config(page_title="S&P 500 Regime Radar", page_icon="\U0001F4E1", layout="wide")
    inject_theme()

    st.markdown('<div class="gradient-title">S&amp;P 500 REGIME RADAR</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="hud-line">
        {bi('GAUSSIAN HMM', 'HMM GAUSSIEN')} &middot; BAUM-WELCH + VITERBI &middot;
        ZCAS UNIVERSITY — {bi('GROUP', 'GROUPE')} 11
    </div><br>""", unsafe_allow_html=True)

    try:
        bundle = load_model_bundle()
        regimes = load_regimes()
    except FileNotFoundError:
        st.markdown(f"""
        <div class="glow-card" style="border-color:#FF3366; box-shadow:0 0 22px rgba(255,51,102,0.25);">
        <b style="color:#FF3366;">\u26A0 {bi('No trained model found', 'Aucun modèle entraîné trouvé')}</b><br><br>
        {bi('Run the Phase 3 pipeline first', "Lancez d'abord la pipeline de la Phase 3")}:
        <pre style="color:{CYAN};">python -m src.model</pre>
        {bi('then reload this page.', 'puis rechargez cette page.')}
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    model, label_map = bundle["model"], bundle["label_map"]
    start_date, end_date, regime_filter, show_split_line = sidebar_controls(regimes)

    view = regimes.loc[start_date:end_date]
    if regime_filter:
        view = view[view["regime"].isin(regime_filter)]

    if view.empty:
        st.warning(bi("No data in the selected range.", "Aucune donnée dans la plage sélectionnée."))
        st.stop()

    latest = regimes.iloc[-1]
    streak = int((regimes["regime"] != regimes["regime"].shift()).cumsum().eq(
        (regimes["regime"] != regimes["regime"].shift()).cumsum().iloc[-1]).sum())

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Current regime", "Régime actuel",
             f"{REGIME_INFO[latest['regime']]['emoji']} {latest['regime']}")
    kpi_card(c2, "As of", "Au", str(regimes.index[-1].date()))
    kpi_card(c3, "Current streak", "Série actuelle", f"{streak} {bi('days', 'jours')}")
    kpi_card(c4, "Days shown", "Jours affichés", f"{len(view):,}")

    st.write("")
    st.plotly_chart(build_price_figure(view, show_split_line), use_container_width=True)
    st.plotly_chart(build_volatility_figure(view), use_container_width=True)

    st.markdown(f"### {bi('Regime statistics', 'Statistiques par régime')}")
    counts = regimes["regime"].value_counts()
    cols = st.columns(3)
    for col, (regime, info) in zip(cols, REGIME_INFO.items()):
        state_idx = next((s for s, name in label_map.items() if name == regime), None)
        if state_idx is None or regime not in regimes["regime"].unique():
            continue
        diag = model.transmat_[state_idx, state_idx]
        share = counts.get(regime, 0) / len(regimes)
        stats_row = {
            "ret": f"{model.means_[state_idx, 0]:+.4%}",
            "vol": f"{model.means_[state_idx, 1]:.4f}",
            "dur": f"\u2248{1 / max(1 - diag, 1e-6):.1f} {bi('days', 'jours')}",
            "share": f"{share:.1%}",
            "share_pct": round(share * 100, 1),
        }
        regime_summary_card(col, regime, info, stats_row)

    render_advice_section(regimes, model, label_map)

    with st.expander(bi("Selected-range data", "Données de la plage sélectionnée")):
        st.dataframe(view[["price", "log_return", "volatility", "regime", "split"]],
                     use_container_width=True)

    st.markdown(f"""<div class="hud-line" style="margin-top:24px; text-align:center;">
        MODEL: GaussianHMM(n={config.N_REGIMES}, cov="{config.COVARIANCE_TYPE}")
        &middot; TRAIN/TEST SPLIT: {config.TRAIN_END_DATE} &middot;
        <span class="pulse-dot"></span>{bi('STATUS: ONLINE', 'STATUT : EN LIGNE')}
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
