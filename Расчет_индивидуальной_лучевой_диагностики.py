import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass, field
from typing import List
import numpy as np

# Настройка шрифтов для кириллицы в графиках
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. СПРАВОЧНИКИ И ФИЗИЧЕСКАЯ МОДЕЛЬ
# ==========================================
ORGAN_NAMES_RU = {
    "gonads": "Половые железы", "bone_marrow": "Красный костный мозг",
    "colon": "Толстая кишка", "lung": "Легкие", "stomach": "Желудок",
    "breast": "Молочные железы", "liver": "Печень", "esophagus": "Пищевод",
    "thyroid": "Щитовидная железа", "skin": "Кожа", "brain": "Головной мозг",
    "salivary_glands": "Слюнные железы", "lens": "Хрусталик глаза",
    "bladder": "Мочевой пузырь", "other": "Прочие ткани"
}
REVERSE_MAP = {v: k for k, v in ORGAN_NAMES_RU.items()}

ORGAN_WEIGHTS = {
    "gonads": 0.08, "bone_marrow": 0.12, "colon": 0.12, "lung": 0.12,
    "stomach": 0.12, "breast": 0.12, "liver": 0.04, "esophagus": 0.04,
    "thyroid": 0.04, "skin": 0.01, "brain": 0.01, "salivary_glands": 0.01,
    "lens": 0.01, "bladder": 0.04, "other": 0.12
}

SCATTER_MODEL = {
    "brain": {"ct": {"salivary_glands": 0.30, "thyroid": 0.05, "lens": 0.40, "skin": 0.15},
              "xray": {"thyroid": 0.02, "skin": 0.10}},
    "lung": {"ct": {"breast": 0.35, "esophagus": 0.30, "thyroid": 0.08, "skin": 0.20, "bone_marrow": 0.05},
             "xray": {"breast": 0.25, "esophagus": 0.20, "thyroid": 0.05, "skin": 0.10}},
    "stomach": {"ct": {"liver": 0.90, "colon": 0.85, "gonads": 0.15, "bone_marrow": 0.10, "skin": 0.25},
                "xray": {"liver": 0.70, "colon": 0.60, "gonads": 0.10, "skin": 0.15}},
    "liver": {"ct": {"stomach": 0.80, "colon": 0.70, "gonads": 0.20, "bone_marrow": 0.08, "skin": 0.20},
              "xray": {"stomach": 0.60, "colon": 0.50, "gonads": 0.12, "skin": 0.10}},
    "colon": {"ct": {"gonads": 0.25, "bladder": 0.15, "bone_marrow": 0.12, "skin": 0.20},
              "xray": {"gonads": 0.15, "bladder": 0.10, "skin": 0.10}}
}

DOSE_LIMITS = {
    "public_annual": 1.0,
    "worker_annual": 20.0,
    "procedure_threshold": 10.0
}


# ==========================================
# 2. КЛАССЫ РАСЧЁТА
# ==========================================
@dataclass
class OrganDose:
    organ_name: str
    dose_mgy: float


@dataclass
class Exam:
    patient: str
    study: str
    modality: str
    age: int
    organ_doses: List[OrganDose] = field(default_factory=list)


class RadiationModule:
    def __init__(self):
        self.threshold_msv = DOSE_LIMITS["procedure_threshold"]

    def calculate_exam(self, exam: Exam) -> dict:
        age_factor = 1.5 if exam.age < 10 else (1.2 if exam.age < 18 else 1.0)
        total_effective = 0.0
        organ_results = []

        for od in exam.organ_doses:
            weight = ORGAN_WEIGHTS.get(od.organ_name, ORGAN_WEIGHTS['other'])
            effective_contrib = od.dose_mgy * weight * age_factor
            total_effective += effective_contrib
            organ_results.append({
                "organ_en": od.organ_name,
                "dose_mgy": od.dose_mgy,
                "weight": weight,
                "effective_msv": round(effective_contrib, 4)
            })

        status = "Превышение" if total_effective >= self.threshold_msv else "Норма"

        if total_effective < 1.0:
            risk_level, risk_color = "Низкий", "🟢"
        elif total_effective < 5.0:
            risk_level, risk_color = "Умеренный", "🟡"
        elif total_effective < self.threshold_msv:
            risk_level, risk_color = "Повышенный", "🟠"
        else:
            risk_level, risk_color = "Высокий", "🔴"

        return {
            "total_msv": round(total_effective, 3),
            "age_factor": age_factor,
            "status": status,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "percent_of_threshold": round(total_effective / self.threshold_msv * 100, 1),
            "organs": organ_results
        }

    def process_exams(self, exams: List[Exam]) -> pd.DataFrame:
        rows = []
        for exam in exams:
            res = self.calculate_exam(exam)
            row = {
                "Пациент": exam.patient, "Исследование": exam.study,
                "Тип": exam.modality, "Возраст": exam.age,
                "Эффективная_доза_общая_мЗв": res["total_msv"],
                "Статус": res["status"],
                "Уровень_риска": res["risk_level"]
            }
            for org in res["organs"]:
                ru = ORGAN_NAMES_RU.get(org["organ_en"], org["organ_en"])
                row[f"Погл_{ru}_мГр"] = org["dose_mgy"]
                row[f"Эфф_{ru}_мЗв"] = org["effective_msv"]
            rows.append(row)

        df = pd.DataFrame(rows)
        fixed = ["Пациент", "Исследование", "Тип", "Возраст", "Эффективная_доза_общая_мЗв", "Статус", "Уровень_риска"]
        others = sorted([c for c in df.columns if c not in fixed])
        return df[fixed + others] if not df.empty else pd.DataFrame()

    def create_chart(self, df: pd.DataFrame):
        eff_cols = [c for c in df.columns if c.startswith("Эфф_") and c.endswith("_мЗв")]
        if not eff_cols: return None

        fig, ax = plt.subplots(figsize=(10, 6))
        patients = df["Пациент"]
        x = range(len(patients))
        colors = plt.cm.Set2(np.linspace(0, 1, len(eff_cols)))
        bottom = [0.0] * len(patients)

        for i, col in enumerate(eff_cols):
            label = col.replace("Эфф_", "").replace("_мЗв", "")
            ax.bar(x, df[col].fillna(0), bottom=bottom, label=label, color=colors[i], width=0.6)
            bottom = [b + v for b, v in zip(bottom, df[col].fillna(0))]

        ax.plot(x, df["Эффективная_доза_общая_мЗв"], color='black', marker='o', linestyle='--', linewidth=2,
                label='Общая доза')
        for i, val in enumerate(df["Эффективная_доза_общая_мЗв"]):
            color = 'green' if df.loc[i, "Статус"] == "Норма" else 'red'
            ax.text(i, val + 0.1, f"{val:.2f}", ha='center', color=color, fontweight='bold')

        y_max = max(df["Эффективная_доза_общая_мЗв"].max() * 1.3,
                    self.threshold_msv * 1.5) if not df.empty else self.threshold_msv * 1.5
        ax.axhline(self.threshold_msv, color='red', linestyle=':', linewidth=1.5, alpha=0.8,
                   label=f'Порог ({self.threshold_msv} мЗв)')
        ax.axhspan(self.threshold_msv, y_max, color='red', alpha=0.1)
        ax.set_title("Эффективная доза по органам (мЗв)", fontsize=14)
        ax.set_xlabel("Пациент")
        ax.set_ylabel("Эффективная доза, мЗв")
        ax.set_xticks(x)
        ax.set_xticklabels(patients, rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(0, y_max)
        fig.tight_layout()
        return fig


# ==========================================
# 3. КОМПОНЕНТЫ ИНТЕРФЕЙСА
# ==========================================
def render_status_card(dose_msv, status, risk_level, risk_color, percent):
    bg = "#d4edda" if status == "Норма" else "#f8d7da"
    text = "#155724" if status == "Норма" else "#721c24"
    border = "#28a745" if status == "Норма" else "#dc3545"
    icon = "✅" if status == "Норма" else "⚠️"
    msg = "Доза находится в пределах допустимых значений. Риск минимален." if status == "Норма" else "Доза превышает установленный порог. Требуется внимание."

    st.markdown(f"""
    <div style="background-color: {bg}; color: {text}; padding: 20px; border-radius: 10px; border-left: 6px solid {border}; margin: 10px 0;">
        <h2 style="margin: 0 0 10px 0;">{icon} Статус: {status}</h2>
        <p style="margin: 5px 0; font-size: 1.1em;"><strong>Эффективная доза:</strong> {dose_msv} мЗв</p>
        <p style="margin: 5px 0; font-size: 1.1em;"><strong>Уровень риска:</strong> {risk_color} {risk_level}</p>
        <p style="margin: 5px 0; font-size: 1.1em;"><strong>От порога (10 мЗв):</strong> {percent}%</p>
        <p style="margin: 10px 0 0 0; font-style: italic; font-weight: 500;">{msg}</p>
    </div>
    """, unsafe_allow_html=True)


def render_final_summary(df, module):
    """Вывод итогового заключения по всем исследованиям"""
    if df.empty: return

    total_exams = len(df)
    normal_exams = len(df[df["Статус"] == "Норма"])
    exceeded_exams = total_exams - normal_exams
    total_dose = df["Эффективная_доза_общая_мЗв"].sum()
    max_dose = df["Эффективная_доза_общая_мЗв"].max()
    max_patient = df.loc[df["Эффективная_доза_общая_мЗв"].idxmax(), "Пациент"]

    st.divider()
    st.subheader("📋 Итоговое заключение")

    if exceeded_exams == 0:
        st.success(
            f"✅ **Все {total_exams} исследований(я) в норме.**\n"
            f"Общая накопленная эффективная доза: **{total_dose:.2f} мЗв**\n"
            f"Превышений порога не обнаружено."
        )
    else:
        st.error(
            f"⚠️ **Обнаружены превышения!**\n"
            f"Всего исследований: {total_exams} | В норме: {normal_exams} | **Превышений: {exceeded_exams}**\n"
            f"Общая накопленная эффективная доза: **{total_dose:.2f} мЗв**\n"
            f"Максимальная доза у пациента: **{max_patient} ({max_dose:.2f} мЗв)**"
        )

        st.info("📝 **Рекомендации:**\n"
                "• Зафиксировать превышения в карте пациента\n"
                "• Обосновать необходимость исследований с превышением порога\n"
                "• Рассмотреть альтернативные методы визуализации (МРТ, УЗИ)\n"
                "• Назначить следующий контроль дозы не ранее чем через 12 месяцев")

    st.caption(f"Пороговое значение: {module.threshold_msv} мЗв. Расчет по МКРЗ 103.")


def render_progress_bar(dose_msv, threshold):
    percent = min(dose_msv / threshold * 100, 100)
    color = "#28a745" if percent < 50 else ("#ffc107" if percent < 100 else "#dc3545")
    st.markdown(f"""
    <div style="margin: 15px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>Заполнение порога {threshold} мЗв</span>
            <span><strong>{percent:.1f}%</strong></span>
        </div>
        <div style="width: 100%; height: 24px; background-color: #e9ecef; border-radius: 12px; overflow: hidden;">
            <div style="width: {percent}%; height: 100%; background-color: {color}; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_results(module, exams):
    df = module.process_exams(exams)
    last_exam = exams[-1]
    last_res = module.calculate_exam(last_exam)

    st.divider()
    st.subheader("📊 Итоговый отчёт по последнему исследованию")

    render_status_card(
        last_res["total_msv"], last_res["status"],
        last_res["risk_level"], last_res["risk_color"],
        last_res["percent_of_threshold"]
    )
    render_progress_bar(last_res["total_msv"], module.threshold_msv)

    st.markdown(f"""
    <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;">
        <strong>📝 Вывод:</strong> 
        Доза пациента <strong>{last_exam.patient}</strong> составляет <strong>{last_res['total_msv']} мЗв</strong>. 
        {last_res['status'].upper()}. 
        Уровень риска: {last_res['risk_color']} {last_res['risk_level'].upper()}.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📈 Сводная таблица всех исследований")
    styled_df = df.style.map(
        lambda x: "background-color: #d4edda; color: #155724; font-weight: bold" if x == "Норма" else (
            "background-color: #f8d7da; color: #721c24; font-weight: bold" if x == "Превышение" else ""),
        subset=['Статус']
    ).map(
        lambda x: "background-color: #e2e3e5; font-weight: bold" if x in ["Низкий", "Умеренный", "Повышенный",
                                                                          "Высокий"] else "",
        subset=['Уровень_риска']
    )
    st.dataframe(styled_df, use_container_width=True, height=300)

    render_final_summary(df, module)

    st.subheader("📊 График распределения")
    fig = module.create_chart(df)
    if fig: st.pyplot(fig)

    csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8")
    st.download_button("📥 Скачать CSV", data=csv, file_name="radiation_report.csv", mime="text/csv",
                       use_container_width=True)

    if st.button("🗑 Очистить все данные", type="secondary"):
        st.session_state.exams = []
        st.session_state.dose_df = pd.DataFrame(columns=["Орган", "Погл_мГр"])
        st.session_state.calc_done = False
        st.rerun()


# ==========================================
# 4. ОСНОВНОЕ ПРИЛОЖЕНИЕ
# ==========================================
def main():
    st.set_page_config(layout="wide", page_title="☢️ Калькулятор лучевой нагрузки")
    st.title("☢️ Калькулятор индивидуальной лучевой нагрузки")
    st.caption("Учет дозы на целевой и соседние органы • Сравнение с нормативами МКРЗ")

    if "exams" not in st.session_state: st.session_state.exams = []
    if "dose_df" not in st.session_state: st.session_state.dose_df = pd.DataFrame(columns=["Орган", "Погл_мГр"])
    if "calc_done" not in st.session_state: st.session_state.calc_done = False

    organ_list_ru = list(ORGAN_NAMES_RU.values())

    st.header("📥 Ввод данных исследования")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    patient = col1.text_input("ФИО Пациента", placeholder="Иванов И.И.")
    age = col2.number_input("Возраст", 1, 120, 35)
    study = col3.text_input("Название исследования", placeholder="КТ грудной клетки")
    modality = col4.selectbox("Метод", ["ct", "xray", "fluoro"],
                              format_func=lambda x: {"ct": "КТ", "xray": "Рентген", "fluoro": "Флюороскопия"}[x])
    target_organ_ru = col5.selectbox("Целевой орган", organ_list_ru)
    target_dose = col6.number_input("Доза целевого органа (мГр)", 0.01, 5000.0, 50.0, step=1.0)

    if st.button("🔄 Рассчитать соседние органы", use_container_width=True, type="primary"):
        target_en = REVERSE_MAP[target_organ_ru]
        data = {"Орган": [target_organ_ru], "Погл_мГр": [target_dose]}
        scatter = SCATTER_MODEL.get(target_en, {}).get(modality, {})
        for org_en, ratio in scatter.items():
            dose = round(target_dose * ratio, 3)
            if dose > 0.01:
                data["Орган"].append(ORGAN_NAMES_RU[org_en])
                data["Погл_мГр"].append(dose)
        st.session_state.dose_df = pd.DataFrame(data)
        st.session_state.calc_done = True
        st.success(f"✅ Рассчитано {len(data['Орган'])} органов")

    if st.session_state.calc_done:
        st.subheader("📝 Таблица доз (редактируемая)")
        edited_df = st.data_editor(
            st.session_state.dose_df, hide_index=True, use_container_width=True,
            column_config={
                "Орган": st.column_config.SelectboxColumn(options=organ_list_ru),
                "Погл_мГр": st.column_config.NumberColumn(min_value=0.0, step=0.1, label="Погл. доза (мГр)")
            }
        )

        if st.button("➕ Добавить в отчет и рассчитать", use_container_width=True, type="primary"):
            if not patient or not study:
                st.error("❌ Укажите пациента и исследование!")
            else:
                organ_list = []
                for _, row in edited_df.iterrows():
                    if pd.isna(row["Погл_мГр"]) or row["Погл_мГр"] <= 0: continue
                    en_key = REVERSE_MAP.get(row["Орган"])
                    if en_key: organ_list.append(OrganDose(en_key, float(row["Погл_мГр"])))

                if not organ_list:
                    st.error("❌ Нет валидных доз!")
                else:
                    st.session_state.exams.append(Exam(patient, study, modality, age, organ_list))
                    st.session_state.calc_done = False
                    st.session_state.dose_df = pd.DataFrame(columns=["Орган", "Погл_мГр"])
                    st.success("✅ Исследование добавлено! Результаты расчета ниже 👇")

    if st.session_state.exams:
        render_results(RadiationModule(), st.session_state.exams)


if __name__ == "__main__":
    main()