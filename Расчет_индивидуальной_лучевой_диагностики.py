import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass, field
from typing import List
import numpy as np

# Настройка шрифтов для кириллицы
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. СПРАВОЧНИКИ И ФИЗИЧЕСКАЯ МОДЕЛЬ
# ==========================================

# Базовый список всех возможных органов
ALL_ORGAN_KEYS = [
    # Голова и шея
    "brain", "lens", "thyroid", "salivary_glands", "skin_head_neck",
    # Грудная клетка
    "lung", "breast", "esophagus", "heart_pericardium", "bone_marrow_thoracic", "skin_chest",
    # Брюшная полость
    "stomach", "liver", "pancreas", "colon", "small_intestine", "kidneys", "spleen",
    "bone_marrow_abdominal", "skin_abdomen",
    # Таз - общие
    "bladder", "rectum", "bone_marrow_pelvic", "skin_pelvis",
    # Таз - мужские
    "prostate",
    # Таз - женские
    "uterus_ovaries",
    # Прочие
    "other"
]

# Названия органов на русском
ORGAN_NAMES_RU = {
    "brain": "Головной мозг",
    "lens": "Хрусталик глаза",
    "thyroid": "Щитовидная железа",
    "salivary_glands": "Слюнные железы",
    "skin_head_neck": "Кожа головы и шеи",

    "lung": "Легкие",
    "breast": "Молочные железы",
    "esophagus": "Пищевод",
    "heart_pericardium": "Сердце/Перикард",
    "bone_marrow_thoracic": "Костный мозг грудной клетки",
    "skin_chest": "Кожа груди",

    "stomach": "Желудок",
    "liver": "Печень",
    "pancreas": "Поджелудочная железа",
    "colon": "Толстая кишка",
    "small_intestine": "Тонкая кишка",
    "kidneys": "Почки",
    "spleen": "Селезенка",
    "bone_marrow_abdominal": "Костный мозг брюшной полости",
    "skin_abdomen": "Кожа живота",

    "bladder": "Мочевой пузырь",
    "rectum": "Прямая кишка",
    "bone_marrow_pelvic": "Костный мозг таза",
    "skin_pelvis": "Кожа таза",

    # Мужские специфические
    "prostate": "Предстательная железа",
    # Женские специфические
    "uterus_ovaries": "Матка и яичники",

    "other": "Прочие ткани"
}

REVERSE_MAP = {v: k for k, v in ORGAN_NAMES_RU.items()}

# Весовые коэффициенты (ICRP 103)
# Примечание: Для gonads используется общий вес 0.08, распределяемый на prostate или uterus_ovaries
ORGAN_WEIGHTS = {
    "brain": 0.01, "lens": 0.01, "thyroid": 0.04, "salivary_glands": 0.01,
    "skin_head_neck": 0.01,
    "lung": 0.12, "breast": 0.12, "esophagus": 0.04, "heart_pericardium": 0.05,
    "bone_marrow_thoracic": 0.06, "skin_chest": 0.01,
    "stomach": 0.12, "liver": 0.04, "pancreas": 0.01, "colon": 0.12,
    "small_intestine": 0.12, "kidneys": 0.01, "spleen": 0.01,
    "bone_marrow_abdominal": 0.06, "skin_abdomen": 0.01,
    "bladder": 0.04, "rectum": 0.01,
    "bone_marrow_pelvic": 0.06, "skin_pelvis": 0.01,

    # Репродуктивные органы (вес 0.08 для каждого типа)
    "prostate": 0.08,
    "uterus_ovaries": 0.08,

    "other": 0.12
}

# Модель рассеяния с учетом пола
# Ключи: Целевой орган -> Метод -> Соседний орган -> Коэффициент
SCATTER_MODEL = {
    # --- ГОЛОВА И ШЕЯ (Не зависят от пола) ---
    "brain": {
        "ct": {"lens": 0.40, "salivary_glands": 0.30, "thyroid": 0.05, "skin_head_neck": 0.15},
        "xray": {"thyroid": 0.02, "skin_head_neck": 0.10}
    },
    "thyroid": {
        "ct": {"lung": 0.10, "esophagus": 0.20, "salivary_glands": 0.15, "skin_chest": 0.10},
        "xray": {"lung": 0.05, "esophagus": 0.15, "skin_chest": 0.05}
    },
    "salivary_glands": {
        "ct": {"brain": 0.30, "thyroid": 0.10, "skin_head_neck": 0.20},
        "xray": {"thyroid": 0.05, "skin_head_neck": 0.10}
    },
    "skin_head_neck": {
        "ct": {"brain": 0.20, "thyroid": 0.10, "salivary_glands": 0.15},
        "xray": {"thyroid": 0.05, "salivary_glands": 0.10}
    },
    "lens": {
        "ct": {"brain": 0.50, "skin_head_neck": 0.20},
        "xray": {"brain": 0.30, "skin_head_neck": 0.15}
    },

    # --- ГРУДНАЯ КЛЕТКА (Не зависят от пола существенно) ---
    "lung": {
        "ct": {
            "breast": 0.35, "esophagus": 0.30, "heart_pericardium": 0.25,
            "thyroid": 0.08, "bone_marrow_thoracic": 0.05, "skin_chest": 0.20
        },
        "xray": {
            "breast": 0.25, "esophagus": 0.20, "heart_pericardium": 0.15,
            "thyroid": 0.05, "skin_chest": 0.10
        }
    },
    "breast": {
        "ct": {"lung": 0.40, "esophagus": 0.20, "heart_pericardium": 0.30, "bone_marrow_thoracic": 0.05,
               "skin_chest": 0.15},
        "xray": {"lung": 0.30, "esophagus": 0.15, "heart_pericardium": 0.20, "skin_chest": 0.10}
    },
    "esophagus": {
        "ct": {"lung": 0.40, "heart_pericardium": 0.30, "stomach": 0.20, "thyroid": 0.10, "skin_chest": 0.15},
        "xray": {"lung": 0.30, "heart_pericardium": 0.20, "stomach": 0.15, "thyroid": 0.05, "skin_chest": 0.10}
    },
    "heart_pericardium": {
        "ct": {"lung": 0.45, "esophagus": 0.30, "stomach": 0.15, "bone_marrow_thoracic": 0.10, "skin_chest": 0.20},
        "xray": {"lung": 0.35, "esophagus": 0.20, "stomach": 0.10, "bone_marrow_thoracic": 0.05, "skin_chest": 0.15}
    },
    "bone_marrow_thoracic": {
        "ct": {"lung": 0.30, "heart_pericardium": 0.20, "esophagus": 0.15, "skin_chest": 0.10},
        "xray": {"lung": 0.20, "heart_pericardium": 0.15, "esophagus": 0.10, "skin_chest": 0.05}
    },
    "skin_chest": {
        "ct": {"lung": 0.20, "breast": 0.15, "heart_pericardium": 0.10},
        "xray": {"lung": 0.15, "breast": 0.10, "heart_pericardium": 0.05}
    },

    # --- БРЮШНАЯ ПОЛОСТЬ (Не зависят от пола) ---
    "stomach": {
        "ct": {
            "liver": 0.90, "pancreas": 0.70, "colon": 0.85, "small_intestine": 0.60,
            "kidneys": 0.40, "spleen": 0.50, "bone_marrow_abdominal": 0.10, "skin_abdomen": 0.25
        },
        "xray": {
            "liver": 0.70, "pancreas": 0.50, "colon": 0.60, "small_intestine": 0.40,
            "kidneys": 0.30, "spleen": 0.40, "bone_marrow_abdominal": 0.08, "skin_abdomen": 0.15
        }
    },
    "liver": {
        "ct": {
            "stomach": 0.80, "pancreas": 0.60, "colon": 0.70, "kidneys": 0.50,
            "small_intestine": 0.40, "bone_marrow_abdominal": 0.08, "skin_abdomen": 0.20
        },
        "xray": {
            "stomach": 0.60, "pancreas": 0.40, "colon": 0.50, "kidneys": 0.40,
            "small_intestine": 0.30, "bone_marrow_abdominal": 0.06, "skin_abdomen": 0.10
        }
    },
    "pancreas": {
        "ct": {"stomach": 0.70, "liver": 0.60, "colon": 0.50, "kidneys": 0.40, "small_intestine": 0.50,
               "bone_marrow_abdominal": 0.10, "skin_abdomen": 0.20},
        "xray": {"stomach": 0.50, "liver": 0.40, "colon": 0.30, "kidneys": 0.30, "small_intestine": 0.40,
                 "bone_marrow_abdominal": 0.08, "skin_abdomen": 0.10}
    },
    "colon": {
        "ct": {
            "small_intestine": 0.60, "bladder": 0.15, "rectum": 0.40,
            "bone_marrow_pelvic": 0.12, "skin_pelvis": 0.20, "stomach": 0.30, "liver": 0.40
        },
        "xray": {
            "small_intestine": 0.40, "bladder": 0.10, "rectum": 0.30,
            "bone_marrow_pelvic": 0.08, "skin_pelvis": 0.10, "stomach": 0.20, "liver": 0.30
        }
    },
    "small_intestine": {
        "ct": {"stomach": 0.60, "colon": 0.70, "pancreas": 0.50, "liver": 0.40, "kidneys": 0.30,
               "bone_marrow_abdominal": 0.15, "skin_abdomen": 0.20},
        "xray": {"stomach": 0.40, "colon": 0.50, "pancreas": 0.30, "liver": 0.30, "kidneys": 0.20,
                 "bone_marrow_abdominal": 0.10, "skin_abdomen": 0.15}
    },
    "kidneys": {
        "ct": {"liver": 0.50, "pancreas": 0.40, "colon": 0.40, "stomach": 0.30, "small_intestine": 0.30,
               "bone_marrow_abdominal": 0.10, "skin_abdomen": 0.15},
        "xray": {"liver": 0.30, "pancreas": 0.30, "colon": 0.30, "stomach": 0.20, "small_intestine": 0.20,
                 "bone_marrow_abdominal": 0.08, "skin_abdomen": 0.10}
    },
    "spleen": {
        "ct": {"stomach": 0.60, "liver": 0.50, "kidneys": 0.40, "colon": 0.40, "bone_marrow_abdominal": 0.10,
               "skin_abdomen": 0.20},
        "xray": {"stomach": 0.40, "liver": 0.30, "kidneys": 0.30, "colon": 0.30, "bone_marrow_abdominal": 0.08,
                 "skin_abdomen": 0.15}
    },
    "bone_marrow_abdominal": {
        "ct": {"stomach": 0.20, "liver": 0.20, "colon": 0.20, "kidneys": 0.15, "skin_abdomen": 0.10},
        "xray": {"stomach": 0.15, "liver": 0.15, "colon": 0.15, "kidneys": 0.10, "skin_abdomen": 0.05}
    },
    "skin_abdomen": {
        "ct": {"stomach": 0.15, "liver": 0.15, "colon": 0.15, "skin_chest": 0.10, "skin_pelvis": 0.10},
        "xray": {"stomach": 0.10, "liver": 0.10, "colon": 0.10, "skin_chest": 0.05, "skin_pelvis": 0.05}
    },

    # --- ТАЗ (Зависят от пола) ---
    # Общие соседи для тазовых органов
    "bladder": {
        "ct": {"rectum": 0.30, "bone_marrow_pelvic": 0.10, "skin_pelvis": 0.15, "colon": 0.20},
        "xray": {"rectum": 0.20, "bone_marrow_pelvic": 0.08, "skin_pelvis": 0.10, "colon": 0.15}
    },
    "rectum": {
        "ct": {"colon": 0.40, "bladder": 0.30, "bone_marrow_pelvic": 0.15, "skin_pelvis": 0.15},
        "xray": {"colon": 0.30, "bladder": 0.20, "bone_marrow_pelvic": 0.10, "skin_pelvis": 0.10}
    },
    "bone_marrow_pelvic": {
        "ct": {"colon": 0.20, "bladder": 0.15, "rectum": 0.15, "skin_pelvis": 0.10},
        "xray": {"colon": 0.15, "bladder": 0.10, "rectum": 0.10, "skin_pelvis": 0.05}
    },
    "skin_pelvis": {
        "ct": {"bladder": 0.15, "rectum": 0.15, "skin_abdomen": 0.10},
        "xray": {"bladder": 0.10, "rectum": 0.10, "skin_abdomen": 0.05}
    },

    # Специфичные для мужчин
    "prostate": {
        "ct": {"bladder": 0.40, "rectum": 0.30, "bone_marrow_pelvic": 0.15, "skin_pelvis": 0.15, "colon": 0.20},
        "xray": {"bladder": 0.30, "rectum": 0.20, "bone_marrow_pelvic": 0.10, "skin_pelvis": 0.10, "colon": 0.15}
    },

    # Специфичные для женщин
    "uterus_ovaries": {
        "ct": {"bladder": 0.35, "rectum": 0.30, "colon": 0.25, "bone_marrow_pelvic": 0.15, "skin_pelvis": 0.15},
        "xray": {"bladder": 0.25, "rectum": 0.20, "colon": 0.15, "bone_marrow_pelvic": 0.10, "skin_pelvis": 0.10}
    }
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
    gender: str
    organ_doses: List[OrganDose] = field(default_factory=list)


class RadiationModule:
    def __init__(self):
        self.threshold_msv = DOSE_LIMITS["procedure_threshold"]

    def calculate_exam(self, exam: Exam) -> dict:
        # Коэффициент возраста
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
            risk_level, risk_color = "Низкий", "#28a745"
        elif total_effective < 5.0:
            risk_level, risk_color = "Умеренный", "#ffc107"
        elif total_effective < self.threshold_msv:
            risk_level, risk_color = "Повышенный", "#fd7e14"
        else:
            risk_level, risk_color = "Высокий", "#dc3545"

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
        all_organ_keys = set()

        for exam in exams:
            res = self.calculate_exam(exam)
            row = {
                "Пациент": exam.patient,
                "Пол": exam.gender,
                "Исследование": exam.study,
                "Метод": exam.modality,
                "Возраст": exam.age,
                "Эфф. доза общая (мЗв)": res["total_msv"],
                "Статус": res["status"],
                "Уровень риска": res["risk_level"]
            }
            for org in res["organs"]:
                ru = ORGAN_NAMES_RU.get(org["organ_en"], org["organ_en"])
                col_abs = f"Погл_{ru}_мГр"
                col_eff = f"Эфф_{ru}_мЗв"
                row[col_abs] = org["dose_mgy"]
                row[col_eff] = org["effective_msv"]
                all_organ_keys.update([col_abs, col_eff])
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Гарантируем наличие всех колонок
        for col in all_organ_keys:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

        fixed_cols = ["Пациент", "Пол", "Исследование", "Метод", "Возраст", "Эфф. доза общая (мЗв)", "Статус",
                      "Уровень риска"]
        other_cols = sorted([c for c in df.columns if c not in fixed_cols])

        return df[fixed_cols + other_cols]

    def create_chart(self, df: pd.DataFrame):
        eff_cols = sorted([c for c in df.columns if c.startswith("Эфф_") and c.endswith("_мЗв")])
        if not eff_cols or df.empty:
            st.warning("Недостаточно данных для построения графика.")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        patients = df["Пациент"]
        x = range(len(patients))
        colors = plt.cm.Set2(np.linspace(0, 1, len(eff_cols)))
        bottom = [0.0] * len(patients)

        for i, col in enumerate(eff_cols):
            label = col.replace("Эфф_", "").replace("_мЗв", "")
            ax.bar(x, df[col].values, bottom=bottom, label=label, color=colors[i], width=0.6)
            bottom = np.add(bottom, df[col].values)

        ax.plot(x, df["Эфф. доза общая (мЗв)"].values, color='black', marker='o', linestyle='--', linewidth=2,
                label='Общая эффективная доза')

        for i, val in enumerate(df["Эфф. доза общая (мЗв)"].values):
            color = '#28a745' if df.loc[i, "Статус"] == "Норма" else '#dc3545'
            ax.text(i, val + 0.1, f"{val:.2f}", ha='center', color=color, fontweight='bold')

        y_max = max(df["Эфф. доза общая (мЗв)"].max() * 1.3, self.threshold_msv * 1.5)
        ax.axhline(self.threshold_msv, color='#dc3545', linestyle=':', linewidth=1.5, alpha=0.8,
                   label=f'Порог ({self.threshold_msv} мЗв)')
        ax.axhspan(self.threshold_msv, y_max, color='#dc3545', alpha=0.1)

        ax.set_title("Распределение эффективной дозы по органам (мЗв)", fontsize=14)
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
    icon = "[ОК]" if status == "Норма" else "[ВНИМАНИЕ]"
    msg = "Доза находится в пределах допустимых значений." if status == "Норма" else "Доза превышает установленный порог. Требуется внимание."

    st.markdown(f"""
    <div style="background-color: {bg}; color: {text}; padding: 20px; border-radius: 10px; border-left: 6px solid {border}; margin: 10px 0;">
        <h2 style="margin: 0 0 10px 0;">{icon} Статус: {status}</h2>
        <p style="margin: 5px 0; font-size: 1.1em;"><strong>Эффективная доза:</strong> {dose_msv} мЗв</p>
        <p style="margin: 5px 0; font-size: 1.1em;"><strong>Уровень риска:</strong> <span style="color:{risk_color}; font-weight:bold;">{risk_level}</span></p>
        <p style="margin: 5px 0; font-size: 1.1em;"><strong>Использование порога (10 мЗв):</strong> {percent}%</p>
        <p style="margin: 10px 0 0 0; font-style: italic; font-weight: 500;">{msg}</p>
    </div>
    """, unsafe_allow_html=True)


def render_final_summary(df, module):
    if df.empty: return

    total_exams = len(df)
    normal_exams = len(df[df["Статус"] == "Норма"])
    exceeded_exams = total_exams - normal_exams
    total_dose = df["Эфф. доза общая (мЗв)"].sum()
    max_dose = df["Эфф. доза общая (мЗв)"].max()
    max_patient = df.loc[df["Эфф. доза общая (мЗв)"].idxmax(), "Пациент"]

    st.divider()
    st.subheader("Итоговое заключение")

    if exceeded_exams == 0:
        st.success(
            f"Все {total_exams} исследование(ий) в норме.\n"
            f"Общая накопленная эффективная доза: {total_dose:.2f} мЗв\n"
            f"Превышений порога не обнаружено."
        )
    else:
        st.error(
            f"Обнаружены превышения!\n"
            f"Всего исследований: {total_exams} | В норме: {normal_exams} | Превышений: {exceeded_exams}\n"
            f"Общая накопленная эффективная доза: {total_dose:.2f} мЗв\n"
            f"Максимальная доза у пациента: {max_patient} ({max_dose:.2f} мЗв)"
        )

        st.info("Рекомендации:\n"
                "- Зафиксировать превышения в карте пациента\n"
                "- Обосновать необходимость исследований с высоким уровнем дозы\n"
                "- Рассмотреть альтернативные методы визуализации (МРТ, УЗИ)\n"
                "- Назначить следующий контроль дозы не ранее чем через 12 месяцев")

    st.caption(f"Пороговое значение: {module.threshold_msv} мЗв. Расчет выполнен согласно рекомендациям МКРЗ 103.")


def render_progress_bar(dose_msv, threshold):
    percent = min(dose_msv / threshold * 100, 100)
    color = "#28a745" if percent < 50 else ("#ffc107" if percent < 100 else "#dc3545")
    st.markdown(f"""
    <div style="margin: 15px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>Заполнение порога ({threshold} мЗв)</span>
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
    st.subheader("Отчет по последнему исследованию")

    render_status_card(
        last_res["total_msv"], last_res["status"],
        last_res["risk_level"], last_res["risk_color"],
        last_res["percent_of_threshold"]
    )
    render_progress_bar(last_res["total_msv"], module.threshold_msv)

    st.markdown(f"""
    <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;">
        <strong>Вывод:</strong> 
        Доза пациента <strong>{last_exam.patient}</strong> составляет <strong>{last_res['total_msv']} мЗв</strong>. 
        {last_res['status'].upper()}. 
        Уровень риска: {last_res['risk_level'].upper()}.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Сводная таблица всех исследований")
    styled_df = df.style.map(
        lambda x: "background-color: #d4edda; color: #155724; font-weight: bold" if x == "Норма" else (
            "background-color: #f8d7da; color: #721c24; font-weight: bold" if x == "Превышение" else ""),
        subset=['Статус']
    ).map(
        lambda x: "background-color: #e2e3e5; font-weight: bold" if x in ["Низкий", "Умеренный", "Повышенный",
                                                                          "Высокий"] else "",
        subset=['Уровень риска']
    )
    st.dataframe(styled_df, use_container_width=True, height=300)

    render_final_summary(df, module)

    st.subheader("График распределения дозы")
    fig = module.create_chart(df)
    if fig: st.pyplot(fig)

    csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8")
    st.download_button("Скачать отчет в CSV", data=csv, file_name="radiation_report.csv", mime="text/csv",
                       use_container_width=True)

    if st.button("Очистить все данные", type="secondary"):
        st.session_state.exams = []
        st.session_state.dose_df = pd.DataFrame(columns=["Орган", "Погл_доза_мГр"])
        st.session_state.calc_done = False
        st.rerun()


# ==========================================
# 4. ОСНОВНОЕ ПРИЛОЖЕНИЕ
# ==========================================
def main():
    st.set_page_config(layout="wide", page_title="Калькулятор лучевой нагрузки")
    st.title("Калькулятор индивидуальной лучевой нагрузки")
    st.caption("Учет дозы на целевой и соседние органы • Сравнение с нормативами МКРЗ")

    if "exams" not in st.session_state: st.session_state.exams = []
    if "dose_df" not in st.session_state: st.session_state.dose_df = pd.DataFrame(columns=["Орган", "Погл_доза_мГр"])
    if "calc_done" not in st.session_state: st.session_state.calc_done = False

    # Определение списка органов в зависимости от пола (по умолчанию мужской, пока не выбран)
    # Но так как selectbox требует статический список при рендеринге, мы будем фильтровать динамически после выбора пола

    st.header("Ввод данных исследования")

    # Разделим ввод на две строки для удобства
    col1, col2, col3, col4 = st.columns(4)
    patient = col1.text_input("ФИО Пациента", placeholder="Иванов И.И.")
    gender = col2.selectbox("Пол", ["Мужской", "Женский"])
    age = col3.number_input("Возраст", 1, 120, 35)
    study = col4.text_input("Название исследования", placeholder="КТ грудной клетки")

    col5, col6, col7 = st.columns(3)
    modality = col5.selectbox("Метод исследования", ["ct", "xray", "fluoro"],
                              format_func=lambda x: {"ct": "КТ", "xray": "Рентген", "fluoro": "Флюороскопия"}[x])

    # Фильтрация органов по полу
    male_specific = ["prostate"]
    female_specific = ["uterus_ovaries"]

    available_organs_keys = ALL_ORGAN_KEYS.copy()
    if gender == "Мужской":
        available_organs_keys = [k for k in available_organs_keys if k not in female_specific]
    else:
        available_organs_keys = [k for k in available_organs_keys if k not in male_specific]

    available_organs_ru = [ORGAN_NAMES_RU[k] for k in available_organs_keys]

    target_organ_ru = col6.selectbox("Целевой орган", available_organs_ru)
    target_dose = col7.number_input("Доза целевого органа (мГр)", 0.01, 5000.0, 50.0, step=0.1)

    if st.button("Рассчитать дозы соседних органов", use_container_width=True, type="primary"):
        target_en = REVERSE_MAP[target_organ_ru]
        data = {"Орган": [target_organ_ru], "Погл_доза_мГр": [target_dose]}

        # Получаем базовые коэффициенты рассеяния
        scatter_base = SCATTER_MODEL.get(target_en, {}).get(modality, SCATTER_MODEL.get(target_en, {}).get("xray", {}))

        # Добавляем ВСЕ соседние органы из модели
        for org_en, ratio in scatter_base.items():
            dose = round(target_dose * ratio, 4)
            org_ru = ORGAN_NAMES_RU.get(org_en, org_en)

            # ВАЖНО: Проверяем, соответствует ли соседний орган полу пациента
            # Если соседний орган специфичен для другого пола, пропускаем его
            is_male_organ = org_en in male_specific
            is_female_organ = org_en in female_specific

            include_organ = True
            if gender == "Мужской" and is_female_organ:
                include_organ = False
            elif gender == "Женский" and is_male_organ:
                include_organ = False

            if include_organ:
                data["Орган"].append(org_ru)
                data["Погл_доза_мГр"].append(dose)

        st.session_state.dose_df = pd.DataFrame(data)
        st.session_state.calc_done = True
        n_neighbors = len(data['Орган']) - 1
        if n_neighbors > 0:
            st.success(f"Рассчитаны дозы: 1 целевой + {n_neighbors} соседних органов")
        else:
            st.info(
                f"Для {target_organ_ru} ({modality}) коэффициенты рассеяния не заданы или не применимы к полу. Добавлен только целевой орган.")

    if st.session_state.calc_done:
        st.subheader("Таблица доз (редактируемая)")
        edited_df = st.data_editor(
            st.session_state.dose_df, hide_index=True, use_container_width=True,
            column_config={
                "Орган": st.column_config.SelectboxColumn(options=available_organs_ru),
                "Погл_доза_мГр": st.column_config.NumberColumn(min_value=0.0, step=0.01, label="Поглощенная доза (мГр)")
            }
        )

        if st.button("Добавить в отчет и рассчитать итог", use_container_width=True, type="primary"):
            if not patient.strip() or not study.strip():
                st.error("Пожалуйста, укажите пациента и название исследования!")
            else:
                organ_list = []
                for _, row in edited_df.iterrows():
                    if pd.isna(row["Погл_доза_мГр"]) or row["Погл_доза_мГр"] <= 0 or pd.isna(row["Орган"]):
                        continue
                    en_key = REVERSE_MAP.get(row["Орган"])
                    if en_key:
                        organ_list.append(OrganDose(en_key, float(row["Погл_доза_мГр"])))

                if not organ_list:
                    st.error("Нет валидных доз в таблице!")
                else:
                    st.session_state.exams.append(
                        Exam(patient.strip(), study.strip(), modality, age, gender, organ_list))
                    st.session_state.calc_done = False
                    st.session_state.dose_df = pd.DataFrame(columns=["Орган", "Погл_доза_мГр"])
                    st.success("Исследование добавлено! Результаты расчета ниже.")

    if st.session_state.exams:
        render_results(RadiationModule(), st.session_state.exams)


if __name__ == "__main__":
    main()
