import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Gender(Enum):
    MALE = "мужской"
    FEMALE = "женский"


class XRayType(Enum):
    FILM = "пленочный"
    DIGITAL = "цифровой"


@dataclass
class OrganDose:
    organ_name: str
    organ_category: str = ""
    direct_dose: float = 0.0
    scatter_dose: float = 0.0
    total_dose: float = 0.0
    is_target: bool = False

    def calculate_total(self):
        self.total_dose = round(self.direct_dose + self.scatter_dose, 4)


@dataclass
class Exam:
    patient: str
    study: str
    modality: str
    dose_value: float
    age: int
    gender: str
    target_zone: str
    xray_type: Optional[XRayType] = None


class RadiationModule:
    def __init__(self):
        self.coefficients = {
            "xray": 0.002,
            "ct": 0.015,
            "fluoro": 0.03,
            "fluorography": 0.025,
            "mammography": 0.008
        }

        self.zone_options = {
            "fluorography": ["органы грудной клетки"],
            "fluoro": ["пищеварительная система", "кардиология", "диафрагма", "пульмонология"],
            "xray": ["органы головы", "органы грудной клетки", "органы малого таза", "коленные суставы", "стопы"],
            "ct": ["органы головы", "органы грудной клетки", "органы малого таза"],
            "mammography": ["молочные железы"]
        }

        self.fluorography_structure = {
            "органы грудной клетки": {
                "target": "легкие",
                "organs": {
                    "легкие": {"scatter_factor": 1.0, "relative_sensitivity": 1.0, "category": "пульмонология"},
                    "сердце": {"scatter_factor": 0.25, "relative_sensitivity": 1.3, "category": "кардиология"},
                    "аорта": {"scatter_factor": 0.15, "relative_sensitivity": 1.1, "category": "кардиология"},
                    "перикард": {"scatter_factor": 0.20, "relative_sensitivity": 1.0, "category": "кардиология"},
                }
            }
        }

        self.fluoro_structure = {
            "пищеварительная система": {
                "target": "пищевод",
                "organs": {
                    "пищевод": {"scatter_factor": 0.30, "relative_sensitivity": 1.1,
                                "category": "пищеварительная_система"},
                    "желудок": {"scatter_factor": 0.18, "relative_sensitivity": 1.0,
                                "category": "пищеварительная_система"},
                    "печень": {"scatter_factor": 0.12, "relative_sensitivity": 1.1,
                               "category": "пищеварительная_система"},
                }
            },
            "кардиология": {
                "target": "сердце",
                "organs": {
                    "сердце": {"scatter_factor": 1.0, "relative_sensitivity": 1.3, "category": "кардиология"},
                    "аорта": {"scatter_factor": 0.35, "relative_sensitivity": 1.1, "category": "кардиология"},
                    "перикард": {"scatter_factor": 0.30, "relative_sensitivity": 1.0, "category": "кардиология"},
                    "легочная_артерия": {"scatter_factor": 0.20, "relative_sensitivity": 1.2,
                                         "category": "кардиология"},
                }
            },
            "диафрагма": {
                "target": "диафрагма",
                "organs": {
                    "диафрагма": {"scatter_factor": 1.0, "relative_sensitivity": 0.9, "category": "пульмонология"},
                    "френический_нерв": {"scatter_factor": 0.10, "relative_sensitivity": 0.8,
                                         "category": "пульмонология"},
                }
            },
            "пульмонология": {
                "target": "бронхи",
                "organs": {
                    "бронхи": {"scatter_factor": 0.80, "relative_sensitivity": 1.2, "category": "пульмонология"},
                    "трахея": {"scatter_factor": 0.35, "relative_sensitivity": 1.0, "category": "пульмонология"},
                    "плевра": {"scatter_factor": 0.40, "relative_sensitivity": 0.9, "category": "пульмонология"},
                }
            }
        }

        self.xray_structure = {
            "органы головы": {
                "target": "череп",
                "organs": {
                    "череп": {"scatter_factor": 1.0, "relative_sensitivity": 0.7, "category": "органы_головы"},
                    "придаточные_пазухи": {"scatter_factor": 0.40, "relative_sensitivity": 0.8,
                                           "category": "органы_головы"},
                    "височная_кость": {"scatter_factor": 0.30, "relative_sensitivity": 0.7,
                                       "category": "органы_головы"},
                }
            },
            "органы грудной клетки": {
                "target": "грудная_клетка",
                "organs": {
                    "грудная_клетка": {"scatter_factor": 1.0, "relative_sensitivity": 0.8,
                                       "category": "органы_грудной_клетки"},
                    "ребра": {"scatter_factor": 0.60, "relative_sensitivity": 0.6, "category": "органы_грудной_клетки"},
                    "грудинная_кость": {"scatter_factor": 0.25, "relative_sensitivity": 0.7,
                                        "category": "органы_грудной_клетки"},
                }
            },
            "органы малого таза": {
                "target": "таз",
                "organs": {
                    "таз": {"scatter_factor": 1.0, "relative_sensitivity": 0.9, "category": "органы_малого_таза"},
                    "крестцовая_кость": {"scatter_factor": 0.45, "relative_sensitivity": 0.7,
                                         "category": "органы_малого_таза"},
                    "тазобедренные_суставы": {"scatter_factor": 0.35, "relative_sensitivity": 0.8,
                                              "category": "органы_малого_таза"},
                }
            },
            "коленные суставы": {
                "target": "коленный_сустав",
                "organs": {
                    "коленный_сустав": {"scatter_factor": 1.0, "relative_sensitivity": 0.7,
                                        "category": "коленные_суставы"},
                    "мениск": {"scatter_factor": 0.50, "relative_sensitivity": 0.6, "category": "коленные_суставы"},
                    "связки_колена": {"scatter_factor": 0.40, "relative_sensitivity": 0.6,
                                      "category": "коленные_суставы"},
                }
            },
            "стопы": {
                "target": "стопа",
                "organs": {
                    "стопа": {"scatter_factor": 1.0, "relative_sensitivity": 0.6, "category": "стопы"},
                    "плюсневые_кости": {"scatter_factor": 0.55, "relative_sensitivity": 0.6, "category": "стопы"},
                    "пятка": {"scatter_factor": 0.45, "relative_sensitivity": 0.6, "category": "стопы"},
                }
            }
        }

        self.ct_structure = {
            "органы головы": {
                "target": "мозг",
                "organs": {
                    "мозг": {"scatter_factor": 1.0, "relative_sensitivity": 1.0, "category": "органы_головы"},
                    "глаза": {"scatter_factor": 0.15, "relative_sensitivity": 1.2, "category": "органы_головы"},
                    "височные_доли": {"scatter_factor": 0.25, "relative_sensitivity": 1.0, "category": "органы_головы"},
                    "мозжечок": {"scatter_factor": 0.20, "relative_sensitivity": 1.1, "category": "органы_головы"},
                }
            },
            "органы грудной клетки": {
                "target": "легкие",
                "organs": {
                    "легкие": {"scatter_factor": 1.0, "relative_sensitivity": 1.0, "category": "органы_грудной_клетки"},
                    "сердце": {"scatter_factor": 0.20, "relative_sensitivity": 1.3,
                               "category": "органы_грудной_клетки"},
                    "аорта": {"scatter_factor": 0.12, "relative_sensitivity": 1.1, "category": "органы_грудной_клетки"},
                    "пищевод": {"scatter_factor": 0.08, "relative_sensitivity": 1.1,
                                "category": "органы_грудной_клетки"},
                }
            },
            "органы малого таза": {
                "target": "мочевой_пузырь",
                "organs": {
                    "мочевой_пузырь": {"scatter_factor": 1.0, "relative_sensitivity": 1.3,
                                       "category": "органы_малого_таза"},
                    "матка": {"scatter_factor": 0.14, "relative_sensitivity": 1.6, "category": "органы_малого_таза"},
                    "яичники": {"scatter_factor": 0.10, "relative_sensitivity": 1.8, "category": "органы_малого_таза"},
                    "прямая_кишка": {"scatter_factor": 0.20, "relative_sensitivity": 1.0,
                                     "category": "органы_малого_таза"},
                }
            }
        }

        self.mammography_structure = {
            "молочные железы": {
                "target": "молочная_железа",
                "organs": {
                    "молочная_железа": {"scatter_factor": 1.0, "relative_sensitivity": 1.8,
                                        "category": "молочные_железы"},
                    "акинус": {"scatter_factor": 0.30, "relative_sensitivity": 1.5, "category": "молочные_железы"},
                    "молочные_протоки": {"scatter_factor": 0.25, "relative_sensitivity": 1.4,
                                         "category": "молочные_железы"},
                }
            }
        }

        self.dose_thresholds = {
            "норма": 1.0,
            "повышенное": 10.0,
            "критическое": 50.0
        }

    def get_zone_options(self, modality: str) -> List[str]:
        return self.zone_options.get(modality, [])

    def validate_exam(self, exam: Exam):
        if not exam.patient:
            raise ValueError("Не указано имя пациента")
        if not exam.study:
            raise ValueError("Не указано исследование")
        if exam.modality not in self.coefficients:
            raise ValueError(f"Неизвестный тип исследования: {exam.modality}")
        if exam.dose_value < 0:
            raise ValueError("Доза не может быть отрицательной")
        if exam.age <= 0:
            raise ValueError("Возраст должен быть положительным")
        if exam.gender not in ["мужской", "женский"]:
            raise ValueError(f"Некорректно указан пол пациента: {exam.gender}")
        if exam.modality == "xray" and exam.xray_type is None:
            raise ValueError("Для рентгена необходимо указать тип")
        if exam.modality == "mammography" and exam.gender != "женский":
            raise ValueError("Маммография доступна только для женщин")
        if exam.target_zone not in self.get_zone_options(exam.modality):
            raise ValueError(f"Недопустимая зона '{exam.target_zone}' для модальности {exam.modality}")

    def _get_structure(self, modality: str) -> Dict:
        structures = {
            "fluorography": self.fluorography_structure,
            "fluoro": self.fluoro_structure,
            "xray": self.xray_structure,
            "ct": self.ct_structure,
            "mammography": self.mammography_structure
        }
        return structures.get(modality, {})

    def calculate_organ_doses(self, exam: Exam) -> Dict[str, OrganDose]:
        self.validate_exam(exam)
        age_factor = 1.2 if exam.age < 18 else 1.0
        base_dose = exam.dose_value * self.coefficients[exam.modality] * age_factor

        if exam.modality == "xray" and exam.xray_type == XRayType.FILM:
            base_dose *= 1.5

        organ_doses = {}
        structure = self._get_structure(exam.modality)

        if exam.target_zone in structure:
            zone_data = structure[exam.target_zone]
            target_organ = zone_data["target"]
            target_params = zone_data["organs"][target_organ]

            target_dose = OrganDose(
                organ_name=target_organ,
                organ_category=target_params["category"],
                direct_dose=round(base_dose, 4),
                scatter_dose=0.0,
                is_target=True
            )
            target_dose.calculate_total()
            organ_doses[f"{exam.target_zone}_{target_organ}"] = target_dose

            for organ_name, params in zone_data["organs"].items():
                if organ_name == target_organ:
                    continue

                scatter_dose = base_dose * params["scatter_factor"] * params["relative_sensitivity"]

                organ_dose = OrganDose(
                    organ_name=organ_name,
                    organ_category=params["category"],
                    direct_dose=0.0,
                    scatter_dose=round(scatter_dose, 4),
                    is_target=False
                )
                organ_dose.calculate_total()
                organ_doses[f"{exam.target_zone}_{organ_name}"] = organ_dose

        return organ_doses

    def get_status(self, dose: float) -> str:
        if dose >= self.dose_thresholds["критическое"]:
            return "Критическое превышение"
        elif dose >= self.dose_thresholds["повышенное"]:
            return "Превышение"
        elif dose >= self.dose_thresholds["норма"]:
            return "Повышенное"
        return "Норма"

    def process_exams(self, exams: List[Exam]) -> pd.DataFrame:
        rows = []
        for exam in exams:
            try:
                organ_doses = self.calculate_organ_doses(exam)
            except ValueError as e:
                st.error(f"Ошибка для {exam.patient}: {e}")
                continue

            for key, dose_info in organ_doses.items():
                status = self.get_status(dose_info.total_dose)
                rows.append({
                    "Пациент": exam.patient,
                    "Исследование": exam.study,
                    "Модальность": exam.modality,
                    "Тип_рентгена": exam.xray_type.value if exam.xray_type else "-",
                    "Пол": exam.gender,
                    "Зона": exam.target_zone,
                    "Орган": dose_info.organ_name,
                    "Категория": dose_info.organ_category,
                    "Тип_воздействия": "Прямое" if dose_info.is_target else "Рассеянное",
                    "Прямая_доза_мЗв": dose_info.direct_dose,
                    "Рассеянная_доза_мЗв": dose_info.scatter_dose,
                    "Суммарная_доза_мЗв": dose_info.total_dose,
                    "Возраст": exam.age,
                    "Статус": status
                })
        return pd.DataFrame(rows)

    def get_patient_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = df.groupby("Пациент").agg({
            "Суммарная_доза_мЗв": "sum",
            "Исследование": "first",
            "Модальность": "first",
            "Тип_рентгена": "first",
            "Пол": "first",
            "Возраст": "first"
        }).reset_index()
        summary["Статус"] = summary["Суммарная_доза_мЗв"].apply(self.get_status)
        summary = summary.rename(columns={"Суммарная_доза_мЗв": "Общая_доза_мЗв"})
        return summary


st.set_page_config(page_title="Радиационный модуль", layout="wide")

st.title("Модуль расчета лучевой нагрузки")
st.markdown(
    "Расчет доз облучения для различных медицинских исследований с учетом рассеянного излучения на соседние органы")

module = RadiationModule()

with st.sidebar:
    st.header("Добавить исследование")

    patient = st.text_input("Имя пациента", "Пациент 1")
    study = st.text_input("Название исследования", "КТ головы")

    modality = st.selectbox(
        "Модальность",
        ["ct", "xray", "fluoro", "fluorography", "mammography"],
        format_func=lambda x: {
            "ct": "КТ",
            "xray": "Рентген",
            "fluoro": "Флюроскопия",
            "fluorography": "Флюорография",
            "mammography": "Маммография"
        }[x]
    )

    xray_type = None
    if modality == "xray":
        xray_type_str = st.selectbox("Тип рентгена", ["пленочный", "цифровой"])
        xray_type = XRayType.FILM if xray_type_str == "пленочный" else XRayType.DIGITAL

    if modality == "mammography":
        gender = "женский"
        st.info("Маммография доступна только для женщин")
    else:
        gender = st.selectbox("Пол", ["мужской", "женский"])

    zone_options = module.get_zone_options(modality)
    target_zone = st.selectbox("Зона исследования", zone_options)

    dose_value = st.number_input("Исходная доза (мГр)", min_value=0.0, value=680.0, step=10.0)
    age = st.number_input("Возраст", min_value=1, max_value=120, value=35)

    if st.button("Добавить в список", type="primary"):
        try:
            new_exam = Exam(
                patient=patient,
                study=study,
                modality=modality,
                dose_value=dose_value,
                age=age,
                gender=gender,
                target_zone=target_zone,
                xray_type=xray_type
            )
            module.validate_exam(new_exam)

            if "exams" not in st.session_state:
                st.session_state.exams = []
            st.session_state.exams.append(new_exam)
            st.success(f"Добавлено: {study} для {patient}")
        except ValueError as e:
            st.error(f"Ошибка: {e}")

    if st.button("Очистить список"):
        st.session_state.exams = []
        st.rerun()

if "exams" not in st.session_state:
    st.session_state.exams = []

st.subheader("Список исследований")
if len(st.session_state.exams) == 0:
    st.info("Список исследований пуст. Добавьте исследование через боковую панель.")
else:
    exams_data = []
    for i, exam in enumerate(st.session_state.exams):
        exams_data.append({
            "№": i + 1,
            "Пациент": exam.patient,
            "Исследование": exam.study,
            "Модальность": exam.modality,
            "Тип рентгена": exam.xray_type.value if exam.xray_type else "-",
            "Пол": exam.gender,
            "Зона": exam.target_zone,
            "Доза (мГр)": exam.dose_value,
            "Возраст": exam.age
        })
    st.dataframe(pd.DataFrame(exams_data), use_container_width=True)

    if st.button("Рассчитать дозы", type="primary"):
        with st.spinner("Выполняется расчет..."):
            df = module.process_exams(st.session_state.exams)

            if not df.empty:
                st.success("Расчет завершен!")

                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Детальная таблица", "Сводка по пациентам", "Графики", "Анализ одного пациента"])

                with tab1:
                    st.subheader("Детальная таблица по органам")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        filter_patient = st.multiselect("Фильтр по пациенту", df["Пациент"].unique())
                    with col2:
                        filter_modality = st.multiselect("Фильтр по модальности", df["Модальность"].unique())
                    with col3:
                        filter_category = st.multiselect("Фильтр по категории", df["Категория"].unique())

                    filtered_df = df.copy()
                    if filter_patient:
                        filtered_df = filtered_df[filtered_df["Пациент"].isin(filter_patient)]
                    if filter_modality:
                        filtered_df = filtered_df[filtered_df["Модальность"].isin(filter_modality)]
                    if filter_category:
                        filtered_df = filtered_df[filtered_df["Категория"].isin(filter_category)]


                    def highlight_status(val):
                        if "Критическое" in str(val):
                            return "background-color: #ffcccc"
                        elif "Превышение" in str(val):
                            return "background-color: #ffe6cc"
                        elif "Повышенное" in str(val):
                            return "background-color: #ffffcc"
                        return "background-color: #ccffcc"


                    try:
                        styled_df = filtered_df.style.map(highlight_status, subset=["Статус"])
                    except AttributeError:
                        styled_df = filtered_df.style.applymap(highlight_status, subset=["Статус"])

                    st.dataframe(styled_df, use_container_width=True)

                    csv = filtered_df.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button("Скачать CSV", csv, "radiation_results.csv", "text/csv")

                with tab2:
                    st.subheader("Сводная таблица по пациентам")
                    summary = module.get_patient_summary(df)


                    def color_dose(val):
                        if val >= 50:
                            return "color: darkred; font-weight: bold"
                        elif val >= 10:
                            return "color: red"
                        elif val >= 1:
                            return "color: orange"
                        return "color: green"


                    try:
                        styled_summary = summary.style.map(color_dose, subset=["Общая_доза_мЗв"])
                    except AttributeError:
                        styled_summary = summary.style.applymap(color_dose, subset=["Общая_доза_мЗв"])

                    st.dataframe(styled_summary, use_container_width=True)

                    csv_summary = summary.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button("Скачать сводку", csv_summary, "radiation_summary.csv", "text/csv")

                with tab3:
                    st.subheader("Общая визуализация")

                    col1, col2 = st.columns(2)

                    with col1:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        colors = []
                        for status in summary["Статус"]:
                            if "Критическое" in status:
                                colors.append("darkred")
                            elif "Превышение" in status:
                                colors.append("red")
                            elif "Повышенное" in status:
                                colors.append("orange")
                            else:
                                colors.append("green")

                        bars = ax.bar(summary["Пациент"], summary["Общая_доза_мЗв"], color=colors)
                        ax.set_title("Суммарная лучевая нагрузка по пациентам")
                        ax.set_xlabel("Пациент")
                        ax.set_ylabel("Доза, мЗв")
                        ax.grid(axis="y", linestyle="--", alpha=0.5)

                        for bar in bars:
                            y = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width() / 2, y + 0.1, f"{y:.3f}",
                                    ha="center", va="bottom")

                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)

                    with col2:
                        fig2, ax2 = plt.subplots(figsize=(8, 8))
                        cat_data = df.groupby("Категория")["Суммарная_доза_мЗв"].sum()
                        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(cat_data)))
                        ax2.pie(cat_data.values, labels=cat_data.index, autopct="%1.1f%%",
                                colors=colors_pie, startangle=90)
                        ax2.set_title("Распределение по категориям органов")
                        plt.tight_layout()
                        st.pyplot(fig2)

                    st.subheader("Тепловая карта по органам")
                    pivot = df.pivot_table(index="Пациент", columns="Орган",
                                           values="Суммарная_доза_мЗв", fill_value=0)
                    fig3, ax3 = plt.subplots(figsize=(14, 6))
                    im = ax3.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
                    ax3.set_xticks(np.arange(len(pivot.columns)))
                    ax3.set_yticks(np.arange(len(pivot.index)))
                    ax3.set_xticklabels(pivot.columns, rotation=45, ha="right")
                    ax3.set_yticklabels(pivot.index)

                    for i in range(len(pivot.index)):
                        for j in range(len(pivot.columns)):
                            ax3.text(j, i, f"{pivot.values[i, j]:.3f}",
                                     ha="center", va="center", color="black", fontsize=8)

                    plt.colorbar(im, ax=ax3, label="Доза, мЗв")
                    plt.tight_layout()
                    st.pyplot(fig3)

                with tab4:
                    st.subheader("Детальный анализ распределения дозы по органам")

                    selected_patient = st.selectbox("Выберите пациента", df["Пациент"].unique())
                    patient_df = df[df["Пациент"] == selected_patient].copy()

                    if not patient_df.empty:
                        patient_info = patient_df.iloc[0]
                        st.markdown(
                            f"**Пациент:** {selected_patient} | **Исследование:** {patient_info['Исследование']} | **Модальность:** {patient_info['Модальность']} | **Зона:** {patient_info['Зона']}")

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            fig, ax = plt.subplots(figsize=(12, 7))

                            organs = patient_df["Орган"].tolist()
                            direct_doses = patient_df["Прямая_доза_мЗв"].tolist()
                            scatter_doses = patient_df["Рассеянная_доза_мЗв"].tolist()
                            total_doses = patient_df["Суммарная_доза_мЗв"].tolist()
                            is_target = patient_df["Тип_воздействия"].tolist()

                            y_pos = np.arange(len(organs))
                            bar_height = 0.6

                            colors_direct = ["#d62728" if t == "Прямое" else "#ff7f0e" for t in is_target]
                            colors_scatter = ["#9467bd" if t == "Прямое" else "#8c564b" for t in is_target]

                            bars1 = ax.barh(y_pos, direct_doses, bar_height, label="Прямая доза", color=colors_direct,
                                            alpha=0.9, edgecolor="black", linewidth=0.5)
                            bars2 = ax.barh(y_pos, scatter_doses, bar_height, left=direct_doses,
                                            label="Рассеянная доза", color=colors_scatter, alpha=0.7, edgecolor="black",
                                            linewidth=0.5)

                            for i, (d, s, total, organ, target) in enumerate(
                                    zip(direct_doses, scatter_doses, total_doses, organs, is_target)):
                                label = " [ЦЕЛЕВОЙ]" if target == "Прямое" else " [соседний]"
                                ax.text(total + 0.02, i, f"{total:.3f} мЗв", va="center", ha="left", fontsize=10,
                                        fontweight="bold")
                                ax.text(0.02, i, f"{organ}{label}", va="center", ha="left", fontsize=9, color="white",
                                        fontweight="bold")

                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(["" for _ in organs])
                            ax.set_xlabel("Доза, мЗв", fontsize=12)
                            ax.set_title(f"Распределение лучевой нагрузки по органам\n{selected_patient}", fontsize=14,
                                         fontweight="bold")
                            ax.legend(loc="lower right", fontsize=10)
                            ax.grid(axis="x", linestyle="--", alpha=0.4)
                            ax.set_xlim(0, max(total_doses) * 1.3)

                            plt.tight_layout()
                            st.pyplot(fig)

                        with col2:
                            st.subheader("Статистика")

                            target_organ = patient_df[patient_df["Тип_воздействия"] == "Прямое"].iloc[0] if len(
                                patient_df[patient_df["Тип_воздействия"] == "Прямое"]) > 0 else None
                            adjacent_organs = patient_df[patient_df["Тип_воздействия"] == "Рассеянное"]

                            if target_organ is not None:
                                st.markdown(f"**Целевой орган:** {target_organ['Орган']}")
                                st.markdown(f"**Доза на целевой орган:** {target_organ['Суммарная_доза_мЗв']:.3f} мЗв")

                            st.markdown(f"**Количество соседних органов:** {len(adjacent_organs)}")

                            total_scatter = adjacent_organs["Суммарная_доза_мЗв"].sum()
                            st.markdown(f"**Суммарная рассеянная доза:** {total_scatter:.3f} мЗв")

                            if target_organ is not None:
                                percent_scatter = (total_scatter / (
                                            target_organ['Суммарная_доза_мЗв'] + total_scatter)) * 100
                                st.markdown(f"**Доля рассеяния:** {percent_scatter:.1f}%")

                            st.markdown("---")
                            st.markdown("**Топ-3 органа по дозе:**")
                            top3 = patient_df.nlargest(3, "Суммарная_доза_мЗв")[
                                ["Орган", "Суммарная_доза_мЗв", "Тип_воздействия"]]
                            for _, row in top3.iterrows():
                                icon = "🎯" if row["Тип_воздействия"] == "Прямое" else "📍"
                                st.markdown(f"{icon} {row['Орган']}: **{row['Суммарная_доза_мЗв']:.3f} мЗв**")

                        st.markdown("---")
                        st.subheader("Сравнительная диаграмма: целевой vs соседние органы")

                        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                        if target_organ is not None:
                            labels_pie = [target_organ['Орган']] + adjacent_organs["Орган"].tolist()
                            sizes_pie = [target_organ['Суммарная_доза_мЗв']] + adjacent_organs[
                                "Суммарная_доза_мЗв"].tolist()
                            colors_pie = ["#d62728"] + ["#ffbb78"] * len(adjacent_organs)
                            explode = [0.05] + [0] * len(adjacent_organs)

                            ax1.pie(sizes_pie, labels=labels_pie, autopct="%1.1f%%", colors=colors_pie,
                                    explode=explode, startangle=90, shadow=True)
                            ax1.set_title("Доля дозы по органам", fontsize=12, fontweight="bold")

                        scatter_sorted = adjacent_organs.sort_values("Суммарная_доза_мЗв", ascending=True)
                        colors_bar = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(scatter_sorted)))

                        bars = ax2.barh(scatter_sorted["Орган"], scatter_sorted["Суммарная_доза_мЗв"], color=colors_bar,
                                        edgecolor="black")
                        ax2.set_xlabel("Рассеянная доза, мЗв", fontsize=11)
                        ax2.set_title("Рассеянная доза на соседние органы", fontsize=12, fontweight="bold")
                        ax2.grid(axis="x", linestyle="--", alpha=0.4)

                        for bar in bars:
                            width = bar.get_width()
                            ax2.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                                     f"{width:.3f}", ha="left", va="center", fontsize=9)

                        plt.tight_layout()
                        st.pyplot(fig2)

            else:
                st.warning("Нет данных для расчета")
