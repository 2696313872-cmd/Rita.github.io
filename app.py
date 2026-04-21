import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from sklearn.ensemble import HistGradientBoostingRegressor

# ==========================================
# 页面配置 / Page Config
# ==========================================
st.set_page_config(
    page_title="Formation Enthalpy Predictor | 生成焓预测",
    page_icon="🧪",
    layout="centered"
)

# ==========================================
# 语言切换 / Language Toggle
# ==========================================
lang = st.sidebar.radio("🌐 Language / 语言", ["English", "中文"])

def t(en, zh):
    return en if lang == "English" else zh

# ==========================================
# 元素性质数据库 / Element Properties Database
# ==========================================
elem_props = {
    'H':  [1, 1.008, 0.31, 2.20], 'He': [2, 4.003, 0.28, 0.0],
    'Li': [3, 6.941, 1.28, 0.98], 'Be': [4, 9.012, 0.96, 1.57],
    'B':  [5, 10.81, 0.84, 2.04], 'C':  [6, 12.01, 0.76, 2.55],
    'N':  [7, 14.01, 0.71, 3.04], 'O':  [8, 16.00, 0.66, 3.44],
    'F':  [9, 19.00, 0.57, 3.98], 'Ne': [10, 20.18, 0.58, 0.0],
    'Na': [11, 22.99, 1.66, 0.93], 'Mg': [12, 24.31, 1.41, 1.31],
    'Al': [13, 26.98, 1.21, 1.61], 'Si': [14, 28.09, 1.11, 1.90],
    'P':  [15, 30.97, 1.07, 2.19], 'S':  [16, 32.06, 1.05, 2.58],
    'Cl': [17, 35.45, 1.02, 3.16], 'K':  [19, 39.10, 2.03, 0.82],
    'Ca': [20, 40.08, 1.76, 1.00], 'Sc': [21, 44.96, 1.70, 1.36],
    'Ti': [22, 47.87, 1.60, 1.54], 'V':  [23, 50.94, 1.53, 1.63],
    'Cr': [24, 52.00, 1.39, 1.66], 'Mn': [25, 54.94, 1.39, 1.55],
    'Fe': [26, 55.85, 1.32, 1.83], 'Co': [27, 58.93, 1.26, 1.88],
    'Ni': [28, 58.69, 1.24, 1.91], 'Cu': [29, 63.55, 1.32, 1.90],
    'Zn': [30, 65.38, 1.22, 1.65], 'Ga': [31, 69.72, 1.22, 1.81],
    'Ge': [32, 72.63, 1.20, 2.01], 'As': [33, 74.92, 1.19, 2.18],
    'Se': [34, 78.96, 1.20, 2.55], 'Br': [35, 79.90, 1.20, 2.96],
    'Rb': [37, 85.47, 2.20, 0.82], 'Sr': [38, 87.62, 1.95, 0.95],
    'Y':  [39, 88.91, 1.90, 1.22], 'Zr': [40, 91.22, 1.75, 1.33],
    'Nb': [41, 92.91, 1.64, 1.60], 'Mo': [42, 95.96, 1.54, 2.16],
    'Tc': [43, 98.00, 1.47, 1.90], 'Ru': [44, 101.1, 1.46, 2.20],
    'Rh': [45, 102.9, 1.42, 2.28], 'Pd': [46, 106.4, 1.39, 2.20],
    'Ag': [47, 107.9, 1.45, 1.93], 'Cd': [48, 112.4, 1.44, 1.69],
    'In': [49, 114.8, 1.42, 1.78], 'Sn': [50, 118.7, 1.39, 1.96],
    'Sb': [51, 121.8, 1.39, 2.05], 'Te': [52, 127.6, 1.38, 2.10],
    'I':  [53, 126.9, 1.39, 2.66], 'Cs': [55, 132.9, 2.44, 0.79],
    'Ba': [56, 137.3, 2.15, 0.89], 'La': [57, 138.9, 2.07, 1.10],
    'Hf': [72, 178.5, 1.75, 1.30], 'Ta': [73, 180.9, 1.70, 1.50],
    'W':  [74, 183.8, 1.62, 2.36], 'Re': [75, 186.2, 1.51, 1.90],
    'Os': [76, 190.2, 1.44, 2.20], 'Ir': [77, 192.2, 1.41, 2.20],
    'Pt': [78, 195.1, 1.36, 2.28], 'Au': [79, 197.0, 1.36, 2.54],
    'Hg': [80, 200.6, 1.32, 2.00], 'Tl': [81, 204.4, 1.45, 1.62],
    'Pb': [82, 207.2, 1.46, 2.33], 'Bi': [83, 209.0, 1.48, 2.02]
}

# ==========================================
# 特征工程 / Feature Engineering
# ==========================================
def parse_formula(formula):
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    elements = re.findall(pattern, formula)
    composition = {}
    total = 0
    for (elem, count) in elements:
        count = float(count) if count else 1.0
        composition[elem] = composition.get(elem, 0) + count
        total += count
    if total == 0:
        return {}
    for elem in composition:
        composition[elem] /= total
    return composition

def get_advanced_features(formula):
    comp = parse_formula(formula)
    if not comp:
        return None, []
    props_matrix = []
    fractions = []
    unknown_elems = []
    for elem, frac in comp.items():
        if elem in elem_props:
            props_matrix.append(elem_props[elem])
            fractions.append(frac)
        else:
            unknown_elems.append(elem)
    if not props_matrix:
        return None, unknown_elems
    props_matrix = np.array(props_matrix)
    fractions = np.array(fractions).reshape(-1, 1)
    weighted_avg = np.sum(props_matrix * fractions, axis=0)
    max_diff = np.ptp(props_matrix, axis=0) if len(props_matrix) > 1 else np.zeros(4)
    return np.concatenate([weighted_avg, max_diff]), unknown_elems

# ==========================================
# 模型训练 / Model Training
# ==========================================
@st.cache_resource(show_spinner=True)
def train_model(csv_path):
    df = pd.read_csv(csv_path)
    features_list, labels_list = [], []
    for formula, label in zip(df['formula'], df['expt_form_e']):
        feat, _ = get_advanced_features(formula)
        if feat is not None:
            features_list.append(feat)
            labels_list.append(label)
    X = np.nan_to_num(np.array(features_list))
    y = np.array(labels_list)
    model = HistGradientBoostingRegressor(
        learning_rate=0.1, max_iter=500,
        max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=0.5, random_state=42
    )
    model.fit(X, y)
    return model, len(df)

# ==========================================
# 侧边栏 / Sidebar
# ==========================================
with st.sidebar:
    st.header(t("📂 Data Settings", "📂 数据设置"))
    st.markdown(t(
        "Upload your CSV training data below.",
        "请在下方上传训练数据 CSV 文件。"
    ))

    uploaded_file = st.file_uploader(
        t("Upload CSV File", "上传 CSV 文件"),
        type=["csv"],
        help=t(
            "CSV must contain 'formula' and 'expt_form_e' columns.",
            "CSV 需包含 'formula' 和 'expt_form_e' 两列。"
        )
    )

    local_path = st.text_input(
        t("Or enter local file path", "或输入本地文件路径"),
        value="expt_formation_enthalpy_kingsbury.csv"
    )

    st.markdown("---")
    st.markdown(t("**📋 Supported Formula Formats**", "**📋 支持的化学式格式**"))
    st.markdown("""
    - `Al2O3` → Aluminium Oxide / 氧化铝
    - `NaCl`  → Sodium Chloride / 氯化钠
    - `Fe2O3` → Iron Oxide / 氧化铁
    - `TiO2`  → Titanium Dioxide / 二氧化钛
    - `H2O`   → Water / 水
    """)

# ==========================================
# 主页面标题 / Main Title
# ==========================================
st.title(t("🧪 Formation Enthalpy Predictor", "🧪 化合物生成焓预测系统"))
st.markdown(t(
    "Powered by **HistGradientBoosting ML model**. Enter a chemical formula to predict its experimental formation enthalpy (eV/atom).",
    "基于 **HistGradientBoosting 机器学习模型**，输入化学式即可预测其实验生成焓 (eV/atom)。"
))
st.markdown("---")

# ==========================================
# 模型加载 / Model Loading
# ==========================================
model = None
total_samples = 0

if uploaded_file is not None:
    save_path = "uploaded_data.csv"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(t("✅ File uploaded successfully!", "✅ 文件上传成功！"))
    model, total_samples = train_model(save_path)
elif os.path.exists(local_path):
    model, total_samples = train_model(local_path)
else:
    st.warning(t(
        "⚠️ Please upload a CSV file in the sidebar, or verify your local file path.",
        "⚠️ 请在左侧侧边栏上传 CSV 文件，或确认本地路径正确。"
    ))

# ==========================================
# 主功能区 / Main Interface
# ==========================================
if model is not None:

    # 状态栏 / Status Bar
    col1, col2, col3 = st.columns(3)
    col1.metric(t("📊 Training Samples", "📊 训练数据量"), f"{total_samples:,}")
    col2.metric(t("🤖 Model", "🤖 模型"), "HistGBR")
    col3.metric(t("🎯 Status", "🎯 状态"), t("✅ Ready", "✅ 已就绪"))

    st.markdown("---")

    # ── 单个预测 / Single Prediction ──
    st.subheader(t("🔬 Single Prediction", "🔬 单个预测"))

    input_col, btn_col = st.columns([4, 1])
    with input_col:
        formula_input = st.text_input(
            label=t("Chemical Formula", "化学式"),
            value="Al2O3",
            placeholder=t("e.g. Fe2O3, NaCl, TiO2, H2O", "例如：Fe2O3, NaCl, TiO2, H2O"),
            label_visibility="collapsed"
        )
    with btn_col:
        predict_btn = st.button(t("🚀 Predict", "🚀 预测"), type="primary", use_container_width=True)

    # ── 批量预测 / Batch Prediction ──
    with st.expander(t("📋 Batch Prediction (multiple compounds)", "📋 批量预测（多个化合物）")):
        batch_input = st.text_area(
            t("One formula per line", "每行输入一个化学式"),
            value="Al2O3\nNaCl\nFe2O3\nTiO2\nMgO",
            height=150
        )
        batch_btn = st.button(t("🚀 Batch Predict", "🚀 批量预测"), type="secondary")

    st.markdown("---")

    # ── 单个预测结果 / Single Prediction Result ──
    if predict_btn and formula_input:
        features, unknown = get_advanced_features(formula_input.strip())
        comp = parse_formula(formula_input.strip())

        if not comp:
            st.error(t(
                "❌ Could not parse this formula. Please check the format (e.g. Al2O3).",
                "❌ 无法识别该化学式，请检查输入格式（例如：Al2O3）。"
            ))
        elif features is None:
            st.error(t(
                f"❌ All elements in '{formula_input}' are outside the database. Cannot predict.",
                f"❌ '{formula_input}' 中的元素均不在数据库中，无法预测。"
            ))
        else:
            # 仅在有未知元素时才显示警告 / Only show warning when unknown elements exist
            if unknown:
                st.warning(t(
                    f"⚠️ The following elements are not in the database and were ignored: {', '.join(unknown)}",
                    f"⚠️ 以下元素不在数据库中，已自动忽略：{', '.join(unknown)}"
                ))

            prediction = model.predict([features])[0]

            st.subheader(t("🎯 Prediction Result", "🎯 预测结果"))
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.metric(
                    label=t(f"Formation Enthalpy of {formula_input}", f"{formula_input} 的生成焓"),
                    value=f"{prediction:.4f} eV/atom"
                )
                if prediction < -1.0:
                    st.success(t("💚 Highly Stable (strongly negative ΔHf)", "💚 高度稳定（强负生成焓）"))
                elif prediction < 0:
                    st.info(t("🔵 Thermodynamically Stable (negative ΔHf)", "🔵 热力学稳定（负生成焓）"))
                elif prediction == 0:
                    st.warning(t("🟡 Thermodynamically Neutral", "🟡 热力学中性"))
                else:
                    st.error(t("🔴 Thermodynamically Unstable (positive ΔHf)", "🔴 热力学不稳定（正生成焓）"))

            with res_col2:
                st.markdown(t("**Elemental Composition (mole fraction)**", "**元素组成（摩尔分数）**"))
                comp_df = pd.DataFrame(
                    list(comp.items()),
                    columns=[t("Element", "元素"), t("Mole Fraction", "摩尔分数")]
                )
                comp_df[t("Mole Fraction", "摩尔分数")] = comp_df[t("Mole Fraction", "摩尔分数")].round(4)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

            with st.expander(t("🔍 View Underlying Physical Features", "🔍 查看底层物理特征")):
                feat_names_en = [
                    "Avg Atomic Number (Z)", "Avg Atomic Mass",
                    "Avg Atomic Radius (Å)", "Avg Electronegativity",
                    "ΔZ (range)", "ΔMass (range)",
                    "ΔRadius (Å, range)", "ΔElectronegativity (range)"
                ]
                feat_names_zh = [
                    "平均原子序数 (Z)", "平均原子质量",
                    "平均原子半径 (Å)", "平均电负性",
                    "原子序数差值", "质量差值",
                    "半径差值 (Å)", "电负性差值"
                ]
                feat_names = feat_names_en if lang == "English" else feat_names_zh
                feat_df = pd.DataFrame({
                    t("Feature", "特征名称"): feat_names,
                    t("Value", "数值"): [f"{v:.4f}" for v in features]
                })
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── 批量预测结果 / Batch Results ──
    if batch_btn:
        formulas = [f.strip() for f in batch_input.strip().split('\n') if f.strip()]
        results = []
        for f in formulas:
            feat, unknown = get_advanced_features(f)
            if feat is not None:
                pred = model.predict([feat])[0]
                if pred < -1.0:
                    stability = t("💚 Highly Stable", "💚 高度稳定")
                elif pred < 0:
                    stability = t("🔵 Stable", "🔵 稳定")
                elif pred == 0:
                    stability = t("🟡 Neutral", "🟡 中性")
                else:
                    stability = t("🔴 Unstable", "🔴 不稳定")
                results.append({
                    t("Formula", "化学式"): f,
                    t("Predicted ΔHf (eV/atom)", "预测生成焓 (eV/atom)"): round(pred, 4),
                    t("Stability", "稳定性"): stability
                })
            else:
                results.append({
                    t("Formula", "化学式"): f,
                    t("Predicted ΔHf (eV/atom)", "预测生成焓 (eV/atom)"): t("Cannot predict", "无法预测"),
                    t("Stability", "稳定性"): t("❓ Unknown", "❓ 未知")
                })

        st.subheader(t("📊 Batch Prediction Results", "📊 批量预测结果"))
        result_df = pd.DataFrame(results)
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        csv_out = result_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label=t("⬇️ Download Results as CSV", "⬇️ 下载预测结果 CSV"),
            data=csv_out,
            file_name="enthalpy_predictions.csv",
            mime="text/csv"
        )