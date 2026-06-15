import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


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
                                "category": "пищеварительная система"},
                    "желудок": {"scatter_factor": 0.18, "relative_sensitivity": 1.0,
                                "category": "пищеварительная система"},
                    "печень": {"scatter_factor": 0.12, "relative_sensitivity": 1.1,
                               "category": "пищеварительная система"},
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
                    "френический нерв": {"scatter_factor": 0.10, "relative_sensitivity": 0.8,
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
                    "череп": {"scatter_factor": 1.0, "relative_sensitivity": 0.7, "category": "органы головы"},
                    "придаточные пазухи": {"scatter_factor": 0.40, "relative_sensitivity": 0.8,
                                           "category": "органы головы"},
                    "височная кость": {"scatter_factor": 0.30, "relative_sensitivity": 0.7,
                                       "category": "органы головы"},
                }
            },
            "органы грудной клетки": {
                "target": "грудная_клетка",
                "organs": {
                    "грудная_клетка": {"scatter_factor": 1.0, "relative_sensitivity": 0.8,
                                       "category": "органы грудной клетки"},
                    "ребра": {"scatter_factor": 0.60, "relative_sensitivity": 0.6, "category": "органы грудной клетки"},
                    "грудинная_кость": {"scatter_factor": 0.25, "relative_sensitivity": 0.7,
                                        "category": "органы грудной клетки"},
                }
            },
            "органы малого таза": {
                "target": "таз",
                "organs": {
                    "таз": {"scatter_factor": 1.0, "relative_sensitivity": 0.9, "category": "органы малого таза"},
                    "крестцовая_кость": {"scatter_factor": 0.45, "relative_sensitivity": 0.7,
                                         "category": "органы малого таза"},
                    "тазобедренные_суставы": {"scatter_factor": 0.35, "relative_sensitivity": 0.8,
                                              "category": "органы малого таза"},
                }
            },
            "коленные суставы": {
                "target": "коленный сустав",
                "organs": {
                    "коленный сустав": {"scatter_factor": 1.0, "relative_sensitivity": 0.7,
                                        "category": "коленные суставы"},
                    "мениск": {"scatter_factor": 0.50, "relative_sensitivity": 0.6, "category": "коленные_суставы"},
                    "связки_колена": {"scatter_factor": 0.40, "relative_sensitivity": 0.6,
                                      "category": "коленные суставы"},
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
                    "мозг": {"scatter_factor": 1.0, "relative_sensitivity": 1.0, "category": "органы головы"},
                    "глаза": {"scatter_factor": 0.15, "relative_sensitivity": 1.2, "category": "органы головы"},
                    "височные доли": {"scatter_factor": 0.25, "relative_sensitivity": 1.0, "category": "органы головы"},
                    "мозжечок": {"scatter_factor": 0.20, "relative_sensitivity": 1.1, "category": "органы головы"},
                }
            },
            "органы грудной клетки": {
                "target": "легкие",
                "organs": {
                    "легкие": {"scatter_factor": 1.0, "relative_sensitivity": 1.0, "category": "органы грудной клетки"},
                    "сердце": {"scatter_factor": 0.20, "relative_sensitivity": 1.3,
                               "category": "органы грудной клетки"},
                    "аорта": {"scatter_factor": 0.12, "relative_sensitivity": 1.1, "category": "органы грудной клетки"},
                    "пищевод": {"scatter_factor": 0.08, "relative_sensitivity": 1.1,
                                "category": "органы грудной клетки"},
                }
            },
            "органы малого таза": {
                "target": "мочевой пузырь",
                "organs": {
                    "мочевой пузырь": {"scatter_factor": 1.0, "relative_sensitivity": 1.3,
                                       "category": "органы малого таза"},
                    "прямая кишка": {"scatter_factor": 0.20, "relative_sensitivity": 1.0,
                                     "category": "органы малого таза"},
                    "тестикулы": {"scatter_factor": 0.14, "relative_sensitivity": 1.6,
                                  "category": "органы малого таза", "gender": "мужской"},

                    "матка": {"scatter_factor": 0.14, "relative_sensitivity": 1.6,
                              "category": "органы малого таза", "gender": "женский"},
                    "яичники": {"scatter_factor": 0.10, "relative_sensitivity": 1.8,
                                "category": "органы малого таза", "gender": "женский"},
                }
            }
        }

        self.mammography_structure = {
            "молочные железы": {
                "target": "молочная железа",
                "organs": {
                    "молочная железа": {"scatter_factor": 1.0, "relative_sensitivity": 1.8,
                                        "category": "молочные железы"},
                    "ацинус": {"scatter_factor": 0.30, "relative_sensitivity": 1.5, "category": "молочные железы"},
                    "молочные протоки": {"scatter_factor": 0.25, "relative_sensitivity": 1.4,
                                         "category": "молочные железы"},
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
            raise ValueError("Не указано ФИО пациента")
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
                if organ_name == target_organ or ("gender" in params and params["gender"] != exam.gender):
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
                    "Тип рентгена": exam.xray_type.value if exam.xray_type else "",
                    "Пол": exam.gender,
                    "Зона": exam.target_zone,
                    "Орган": dose_info.organ_name,
                    "Тип воздействия": "Прямое" if dose_info.is_target else "Рассеянное",
                    "Прямая доза (мЗв)": dose_info.direct_dose,
                    "Рассеянная доза (мЗв)": dose_info.scatter_dose,
                    "Суммарная доза (мЗв)": dose_info.total_dose,
                    "Возраст": exam.age,
                    "Статус": status
                })
        return pd.DataFrame(rows)

    def get_patient_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = df.groupby("Пациент").agg({
            "Суммарная доза (мЗв)": "sum",
            "Исследование": "first",
            "Модальность": "first",
            "Тип рентгена": "first",
            "Пол": "first",
            "Возраст": "first"
        }).reset_index()
        summary["Статус"] = summary["Суммарная доза (мЗв)"].apply(self.get_status)
        summary = summary.rename(columns={"Суммарная доза (мЗв)": "Общая доза (мЗв)"})
        return summary


def get_ru(x):
    return {
        "ct": "КТ",
        "xray": "Рентген",
        "fluoro": "Флюроскопия",
        "fluorography": "Флюорография",
        "mammography": "Маммография"
    }[x]


st.set_page_config(page_title="Радиационный модуль", layout="wide")

st.title("Модуль расчета лучевой нагрузки")
st.markdown(
    "Расчет доз облучения для различных медицинских исследований с учетом рассеянного излучения на соседние органы")

module = RadiationModule()

with st.sidebar:
    st.header("Добавить исследование")

    patient = st.text_input("ФИО пациента", "Пациент 1")
    study = st.text_input("Название исследования", "КТ головы")

    modality = st.selectbox(
        "Модальность",
        ["ct", "xray", "fluoro", "fluorography", "mammography"],
        format_func=get_ru
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
            "Модальность": get_ru(exam.modality),
            "Тип рентгена": exam.xray_type.value if exam.xray_type else "",
            "Пол": exam.gender,
            "Зона": exam.target_zone,
            "Доза (мГр)": exam.dose_value,
            "Возраст": exam.age
        })
    st.dataframe(pd.DataFrame(exams_data), use_container_width=True, hide_index=True)

    if st.button("Рассчитать дозы", type="primary"):
        with (st.spinner("Выполняется расчет...")):
            df = module.process_exams(st.session_state.exams)
            if not df.empty:
                st.success("Расчет завершен!")
                df["Модальность"] = df["Модальность"].apply(get_ru)

                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Детальная таблица", "Сводка по пациентам", "Графики", "Анализ одного пациента"])

                with tab1:
                    st.subheader("Детальная таблица по органам")


                    @st.fragment
                    def update_table(df_in):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            filter_patient = st.multiselect("Фильтр по пациенту", df_in["Пациент"].unique(),
                                                            placeholder="Нажмите здесь для выбора...",
                                                            key="patient")

                        with col2:
                            filter_modality = st.multiselect("Фильтр по модальности", df_in["Модальность"].unique(),
                                                             placeholder="Нажмите здесь для выбора...",
                                                             key="modality")
                        with col3:
                            filter_category = st.multiselect("Фильтр по зоне исследования", df_in["Зона"].unique(),
                                                             placeholder="Нажмите здесь для выбора...",
                                                             key="category")
                        filtered_df = df_in.copy()
                        if filter_patient:
                            filtered_df = filtered_df[filtered_df["Пациент"].isin(st.session_state.patient)]
                        if filter_modality:
                            filtered_df = filtered_df[filtered_df["Модальность"].isin(st.session_state.modality)]
                        if filter_category:
                            filtered_df = filtered_df[filtered_df["Зона"].isin(st.session_state.category)]

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

                        st.dataframe(styled_df, use_container_width=True, hide_index=True,
                                     column_config={
                                         "Прямая доза (мЗв)": st.column_config.NumberColumn(format="%.3f"),
                                         "Рассеянная доза (мЗв)": st.column_config.NumberColumn(format="%.3f"),
                                         "Суммарная доза (мЗв)": st.column_config.NumberColumn(format="%.3f")
                                     })

                        csv = filtered_df.to_csv(index=False, encoding="utf-8-sig")
                        st.download_button("Скачать CSV", csv, "radiation_results.csv", "text/csv")


                    update_table(df)

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
                        styled_summary = summary.style.map(color_dose, subset=["Общая доза (мЗв)"])
                    except AttributeError:
                        styled_summary = summary.style.applymap(color_dose, subset=["Общая доза (мЗв)"])

                    st.dataframe(styled_summary, use_container_width=True, hide_index=True,
                                 column_config={
                                     "Общая доза (мЗв)": st.column_config.NumberColumn(format="%.3f"),
                                 })

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

                        bars = ax.bar(summary["Пациент"], summary["Общая доза (мЗв)"], color=colors)
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
                        cat_data = df.groupby("Зона")["Суммарная доза (мЗв)"].sum()
                        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(cat_data)))
                        ax2.pie(cat_data.values, labels=cat_data.index, autopct="%1.1f%%",
                                colors=colors_pie, startangle=90)
                        ax2.set_title("Распределение по зонам органов")
                        plt.tight_layout()
                        st.pyplot(fig2)

                    st.subheader("Тепловая карта по органам")
                    pivot = df.pivot_table(index="Пациент", columns="Орган",
                                           values="Суммарная доза (мЗв)", fill_value=0)
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


                    @st.fragment
                    def update_fragment(df):
                        selected_patient = st.selectbox("Выберите пациента", df["Пациент"].unique())
                        patient_df = df[df["Пациент"] == selected_patient].copy()

                        if not patient_df.empty:
                            patient_info = patient_df.iloc[0]
                            st.markdown(
                                f"**Пациент:** {selected_patient} | **Исследование:** {patient_info['Исследование']} | **Модальность:** {patient_info['Модальность']} | **Зона:** {patient_info['Зона']}")

                            col1, col2 = st.columns([2, 1])
                            # Агрегация по органам
                            agg_df = (
                                patient_df
                                .groupby("Орган", as_index=False)
                                .agg({
                                    "Прямая доза (мЗв)": "sum",
                                    "Рассеянная доза (мЗв)": "sum",
                                    "Суммарная доза (мЗв)": "sum",
                                    "Тип воздействия": lambda x: "Прямое" if "Прямое" in x.values else "Рассеянное"
                                })
                            )
                            with col1:
                                target_organs = (
                                    patient_df.groupby("Орган")["Тип воздействия"]
                                    .apply(lambda x: "Прямое" if "Прямое" in x.values else "Рассеянное")
                                    .reset_index()
                                )

                                fig, ax = plt.subplots(figsize=(12, 7))

                                organs = agg_df["Орган"].tolist()
                                direct_doses = agg_df["Прямая доза (мЗв)"].tolist()
                                scatter_doses = agg_df["Рассеянная доза (мЗв)"].tolist()
                                total_doses = agg_df["Суммарная доза (мЗв)"].tolist()
                                is_target = agg_df["Тип воздействия"].tolist()

                                y_pos = np.arange(len(organs))
                                bar_height = 0.6

                                bars1 = ax.barh(y_pos, direct_doses, bar_height, label="Прямая доза",
                                                color="#d62728", alpha=0.9, edgecolor="black", linewidth=0.5)

                                bars2 = ax.barh(y_pos, scatter_doses, bar_height, left=direct_doses,
                                                label="Рассеянная доза", color="#9467bd", alpha=0.7,
                                                edgecolor="black", linewidth=0.5)

                                for i, (d, s, total, organ, target) in enumerate(
                                        zip(direct_doses, scatter_doses, total_doses, organs, is_target)
                                ):
                                    label = " [ЦЕЛЕВОЙ]" if target == "Прямое" else " [соседний]"
                                    ax.text(total + 0.02, i, f"{total:.3f} мЗв", va="center",
                                            ha="left", fontsize=10, fontweight="bold")
                                    ax.text(0.02, i, f"{organ}{label}", va="center",
                                            ha="left", fontsize=9, color="white", fontweight="bold")

                                ax.set_yticks(y_pos)
                                ax.set_yticklabels([""] * len(organs))
                                ax.set_xlabel("Накопленная доза, мЗв", fontsize=12)
                                ax.set_title(
                                    f"Распределение накопленной лучевой нагрузки по органам\n{selected_patient}",
                                    fontsize=14,
                                    fontweight="bold",
                                )

                                ax.legend(loc="lower right")
                                ax.grid(axis="x", linestyle="--", alpha=0.4)
                                ax.set_xlim(0, max(total_doses) * 1.3)

                                plt.tight_layout()
                                st.pyplot(fig)

                            with col2:
                                st.subheader("Статистика")

                                target_organ = agg_df[agg_df["Тип воздействия"] == "Прямое"].iloc[0] if len(
                                    agg_df[agg_df["Тип воздействия"] == "Прямое"]) > 0 else None
                                adjacent_organs = agg_df[agg_df["Тип воздействия"] == "Рассеянное"]

                                if target_organ is not None:
                                    st.markdown(f"**Целевой орган:** {target_organ['Орган']}")
                                    st.markdown(
                                        f"**Доза на целевой орган:** {target_organ['Суммарная доза (мЗв)']:.3f} мЗв")

                                st.markdown(f"**Количество соседних органов:** {len(adjacent_organs)}")

                                total_scatter = adjacent_organs["Суммарная доза (мЗв)"].sum()
                                st.markdown(f"**Суммарная рассеянная доза:** {total_scatter:.3f} мЗв")

                                if target_organ is not None:
                                    percent_scatter = (total_scatter / (
                                            target_organ['Суммарная доза (мЗв)'] + total_scatter)) * 100
                                    st.markdown(f"**Доля рассеяния:** {percent_scatter:.1f}%")

                                st.markdown("---")
                                st.markdown("**Топ-3 органа по дозе:**")
                                top3 = agg_df.nlargest(3, "Суммарная доза (мЗв)")[
                                    ["Орган", "Суммарная доза (мЗв)", "Тип воздействия"]]
                                for _, row in top3.iterrows():
                                    st.markdown(
                                        f"{row['Орган'].capitalize()}: **{row['Суммарная доза (мЗв)']:.3f} мЗв**")

                            st.markdown("---")
                            st.subheader("Сравнительная диаграмма: целевой и соседние органы")

                            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                            doze_values = []
                            if target_organ is not None:
                                labels_pie = [target_organ['Орган']] + adjacent_organs["Орган"].tolist()
                                doze_values = [target_organ['Суммарная доза (мЗв)']] + adjacent_organs[
                                    "Суммарная доза (мЗв)"].tolist()

                                adj_values = adjacent_organs["Суммарная доза (мЗв)"]
                                norm = mcolors.Normalize(
                                    vmin=min(doze_values),
                                    vmax=max(doze_values)
                                )
                                colors_pie = plt.cm.Reds(norm(doze_values))
                                explode = [0.05] + [0] * len(adjacent_organs)
                                ax1.pie(doze_values, labels=labels_pie, autopct="%1.1f%%", colors=colors_pie,
                                        explode=explode, startangle=90, shadow=True)
                                ax1.set_title("Доля дозы по органам", fontsize=12, fontweight="bold")

                            scatter_sorted = adjacent_organs.sort_values("Суммарная доза (мЗв)", ascending=True)
                            # colors_bar = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(scatter_sorted)))

                            values = scatter_sorted["Суммарная доза (мЗв)"]

                            # Нормализация значений в диапазон [0, 1]
                            norm = mcolors.Normalize(
                                vmin=min(doze_values),
                                vmax=max(doze_values)
                            )

                            # Зелёный → Жёлтый → Красный
                            cmap = plt.cm.Reds

                            colors_bar = cmap(norm(values))

                            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                            sm.set_array([])

                            cbar = plt.colorbar(sm, ax=ax2)
                            cbar.set_label("Доза, мЗв")

                            bars = ax2.barh(scatter_sorted["Орган"], scatter_sorted["Суммарная доза (мЗв)"],
                                            color=colors_bar,
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


                    update_fragment(df)

            else:
                st.warning("Нет данных для расчета")
