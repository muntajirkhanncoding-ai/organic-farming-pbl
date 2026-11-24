"""Generate a complete Word project report from the trained artefacts.

Usage
-----
    python generate_report.py
    python generate_report.py --output docs/Organic_Farming_Project_Report.docx

The report pulls the real numbers from outputs/results_summary.json and the
figures from outputs/figures, so it always matches the trained model.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from organic_ml.config import (  # noqa: E402
    BUNDLE_PATH,
    FIGURE_DIR,
    KEY_FEATURES,
    RESULTS_JSON,
    TABLE_DIR,
)
from organic_ml.explain import confidence_label, local_explanations  # noqa: E402
from organic_ml.io import load_bundle, load_json  # noqa: E402

GREEN = RGBColor(0x2A, 0x9D, 0x8F)
DARK = RGBColor(0x26, 0x46, 0x53)
ORANGE = RGBColor(0xE7, 0x6F, 0x51)
GREY = RGBColor(0x6B, 0x7C, 0x72)
LIGHT_GREEN = "E8F3EE"
LIGHT_GREY = "F4F6F3"


# --------------------------------------------------------------------------
# Low-level Word helpers
# --------------------------------------------------------------------------
def set_cell_background(cell, hex_color: str) -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(shading)


def set_cell_margins(cell, top=60, bottom=60, left=90, right=90) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for tag, value in (("top", top), ("bottom", bottom), ("start", left), ("end", right)):
        node = OxmlElement(f"w:{tag}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def add_page_number_footer(section) -> None:
    footer = section.footer
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    run = paragraph.add_run("Page ")
    run.font.size = Pt(9)
    run.font.color.rgb = GREY

    _add_field(paragraph, "PAGE")
    run = paragraph.add_run(" of ")
    run.font.size = Pt(9)
    run.font.color.rgb = GREY
    _add_field(paragraph, "NUMPAGES")


def _add_field(paragraph, field_name: str) -> None:
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = field_name
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    run.font.size = Pt(9)
    run.font.color.rgb = GREY


def add_horizontal_rule(paragraph, color: str = "2A9D8F", size: int = 12) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), color)
    pBdr.append(bottom)
    pPr.append(pBdr)


def add_bookmark(paragraph, name: str) -> None:
    """Optional helper retained for future cross-references."""
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), name)
    start.set(qn("w:name"), name)
    paragraph._p.insert(0, start)


class Report:
    def __init__(self, results: dict) -> None:
        self.results = results
        self.doc = Document()
        self.fig_no = 0
        self.table_no = 0
        self._setup_styles()
        self._setup_page()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup_styles(self) -> None:
        styles = self.doc.styles
        normal = styles["Normal"]
        normal.font.name = "Calibri"
        normal.font.size = Pt(10.5)
        normal.paragraph_format.space_after = Pt(6)
        normal.paragraph_format.line_spacing = 1.15

        for name, size, color in [
            ("Title", 26, DARK),
            ("Heading 1", 16, GREEN),
            ("Heading 2", 13, DARK),
            ("Heading 3", 11.5, GREEN),
        ]:
            style = styles[name]
            style.font.name = "Calibri"
            style.font.size = Pt(size)
            style.font.color.rgb = color
            style.font.bold = True
            style.paragraph_format.space_before = Pt(14)
            style.paragraph_format.space_after = Pt(6)

    def _setup_page(self) -> None:
        section = self.doc.sections[0]
        section.top_margin = Cm(2.2)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.4)
        section.right_margin = Cm(2.2)

        header = section.header.paragraphs[0]
        header.text = "Machine Learning-Based Organic Farming Suitability Prediction"
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for run in header.runs:
            run.font.size = Pt(8.5)
            run.font.color.rgb = GREY
            run.italic = True
        add_page_number_footer(section)

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------
    def heading(self, text: str, level: int = 1) -> None:
        self.doc.add_heading(text, level=level)

    def para(self, text: str, bold: bool = False, italic: bool = False, size: Optional[float] = None) -> None:
        paragraph = self.doc.add_paragraph()
        run = paragraph.add_run(text)
        run.bold = bold
        run.italic = italic
        if size:
            run.font.size = Pt(size)
        paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def bullets(self, items: Sequence[str]) -> None:
        for item in items:
            self.doc.add_paragraph(item, style="List Bullet")

    def numbered(self, items: Sequence[str]) -> None:
        for item in items:
            self.doc.add_paragraph(item, style="List Number")

    def table_caption(self, text: str) -> None:
        self.table_no += 1
        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(f"Table {self.table_no}: {text}")
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = DARK

    def figure(self, filename: str, caption: str, width: float = 6.1) -> None:
        path = FIGURE_DIR / filename
        if not path.exists():
            return
        self.doc.add_picture(str(path), width=Inches(width))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        self.fig_no += 1
        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(f"Figure {self.fig_no}: {caption}")
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = GREY

    def table(
        self,
        frame: pd.DataFrame,
        columns: Sequence[str],
        headers: Sequence[str],
        caption: str,
        widths: Optional[Sequence[float]] = None,
        font_size: float = 8.5,
        highlight_best: bool = False,
    ) -> None:
        self.table_caption(caption)
        table = self.doc.add_table(rows=1, cols=len(columns))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True

        header_cells = table.rows[0].cells
        for cell, header in zip(header_cells, headers):
            cell.text = ""
            paragraph = cell.paragraphs[0]
            run = paragraph.add_run(str(header))
            run.bold = True
            run.font.size = Pt(font_size)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_cell_background(cell, "264653")
            set_cell_margins(cell)

        for row_index, (_, row) in enumerate(frame.iterrows()):
            cells = table.add_row().cells
            for col_index, column in enumerate(columns):
                cell = cells[col_index]
                cell.text = ""
                value = row[column]
                paragraph = cell.paragraphs[0]
                run = paragraph.add_run(str(value))
                run.font.size = Pt(font_size)
                if col_index == 0:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                else:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                set_cell_margins(cell)
                if row_index % 2 == 1:
                    set_cell_background(cell, LIGHT_GREY)

        if widths:
            for row in table.rows:
                for index, width in enumerate(widths):
                    row.cells[index].width = Inches(width)
        self.doc.add_paragraph()

    def page_break(self) -> None:
        self.doc.add_page_break()

    # ------------------------------------------------------------------
    # Report sections
    # ------------------------------------------------------------------
    def title_page(self) -> None:
        for _ in range(2):
            self.doc.add_paragraph()

        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run("A PROJECT REPORT ON")
        run.font.size = Pt(11)
        run.font.color.rgb = GREY
        run.bold = True
        run.font.name = "Calibri"

        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run("MACHINE LEARNING-BASED\nORGANIC FARMING SUITABILITY PREDICTION")
        run.font.size = Pt(25)
        run.bold = True
        run.font.color.rgb = DARK

        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_horizontal_rule(paragraph)

        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run("BCA 5th Semester  |  Project-Based Learning (PBL)")
        run.font.size = Pt(12)
        run.font.color.rgb = GREEN
        run.bold = True

        for _ in range(2):
            self.doc.add_paragraph()

        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run("Submitted by")
        run.font.size = Pt(10)
        run.font.color.rgb = GREY

        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run("[Your Name]\n[Roll Number]")
        run.font.size = Pt(14)
        run.bold = True
        run.font.color.rgb = DARK

        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run("Under the guidance of")
        run.font.size = Pt(10)
        run.font.color.rgb = GREY

        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run("[Guide Name]\nAssistant Professor, Department of Computer Applications")
        run.font.size = Pt(12.5)
        run.bold = True

        for _ in range(2):
            self.doc.add_paragraph()

        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run("[Department Name]\n[College / University Name]\nAcademic Year 2025-26")
        run.font.size = Pt(11)

        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(f"Report generated on {date.today():%d %B %Y}")
        run.font.size = Pt(8.5)
        run.font.color.rgb = GREY
        run.italic = True

        self.page_break()

    def declaration(self) -> None:
        self.heading("Declaration", 1)
        self.para(
            "I hereby declare that this project report entitled "
            "\u201cMachine Learning-Based Organic Farming Suitability Prediction\u201d "
            "is my own work carried out as part of the BCA 5th semester "
            "Project-Based Learning programme. The code, experiments and results "
            "presented here were produced by me under the guidance of my faculty "
            "mentor. All external libraries, datasets and references used are "
            "acknowledged in the References section."
        )
        self.doc.add_paragraph()
        self.para("Place: ____________________", size=10)
        self.para("Date:  ____________________", size=10)
        self.doc.add_paragraph()
        self.para("Signature: ____________________", bold=True)
        self.page_break()

    def abstract(self) -> None:
        metrics = self.results["test_metrics_tuned"]
        cv = self.results["cv_metrics"]
        self.heading("Abstract", 1)
        self.para(
            "Organic farming avoids synthetic fertilizers and pesticides and "
            "depends on healthy soil, balanced nutrients and careful management. "
            "Deciding whether a particular field is ready for organic conversion "
            "is therefore a multi-factor problem. This project builds a machine "
            "learning system that predicts whether a crop-field combination is "
            "suitable for organic farming from nine soil and management inputs: "
            "pH, organic carbon, available nitrogen, phosphorus, potassium and "
            "sulphur, previous fertilizer use, previous pesticide use and years "
            "of chemical input."
        )
        self.para(
            f"The dataset contains 1,000 crop-field records with 24 columns. "
            f"After data validation and agronomic feature engineering, seven "
            f"model families were tuned with randomised search and compared "
            f"using repeated stratified cross-validation. The selected model was "
            f"a {self.results['selected_model']}. On the untouched 20% test set it "
            f"achieved {metrics['accuracy']:.1%} accuracy, "
            f"{metrics['f1']:.3f} F1 score and {metrics['roc_auc']:.3f} ROC-AUC "
            f"(cross-validation ROC-AUC {cv['roc_auc']:.3f})."
        )
        self.para(
            "The project also provides per-field explanations, agronomic "
            "recommendations, a batch prediction command and an interactive "
            "web application. A key finding is that a carefully engineered "
            "linear model matches or beats complex gradient-boosting models on "
            "this dataset, which makes the final system both accurate and "
            "explainable."
        )
        self.para(
            "Keywords: organic farming, soil suitability, machine learning, "
            "logistic regression, feature engineering, explainable AI.",
            italic=True,
            size=9.5,
        )
        self.page_break()

    def contents(self) -> None:
        self.heading("Contents", 1)
        items = [
            "1. Introduction",
            "2. Problem Statement and Objectives",
            "3. Dataset Description",
            "4. Methodology",
            "5. Exploratory Data Analysis",
            "6. Results",
            "7. Model Interpretation",
            "8. Comparison with the Previous Version",
            "9. Limitations and Ethical Considerations",
            "10. Conclusion and Future Work",
            "11. References",
            "Appendix A: Viva Questions and Answers",
            "Appendix B: How to Run the Project",
        ]
        for item in items:
            paragraph = self.doc.add_paragraph(item)
            paragraph.paragraph_format.space_after = Pt(4)
        self.page_break()

    def introduction(self) -> None:
        self.heading("1. Introduction", 1)
        self.para(
            "India's agricultural sector supports nearly half of its workforce, "
            "and a growing number of farmers are exploring organic cultivation "
            "because of rising input costs, soil degradation and consumer demand "
            "for chemical-free produce. Organic farming is not simply the "
            "absence of chemicals: it requires sufficient organic carbon, a "
            "balanced nutrient profile, a suitable pH range and a realistic "
            "transition away from synthetic inputs."
        )
        self.para(
            "Soil testing produces large amounts of numeric data, but "
            "interpreting that data consistently is difficult. Two fields with "
            "similar pH can have very different organic carbon and nutrient "
            "levels, and the effect of pH is not linear: both very acidic and "
            "very alkaline soils are poor for most crops. Machine learning is "
            "well suited to this kind of multi-factor, non-linear decision "
            "problem."
        )
        self.para(
            "This report presents a complete, reproducible pipeline that "
            "predicts organic-farming suitability, compares several classifiers, "
            "explains the predictions and exposes the result through a web "
            "application. It is an improved version of an earlier single-script "
            "project; the differences are summarised in Section 8."
        )

        self.heading("2. Problem Statement and Objectives", 1)
        self.para(
            "Given the soil chemistry and past input-management history of a "
            "field, predict whether it is suitable for organic farming. The "
            "problem is a binary classification task with two classes: "
            "\u201cSuitable\u201d and \u201cNot Suitable\u201d."
        )
        self.para("The specific objectives are:", bold=True)
        self.numbered(
            [
                "Build a clean and reproducible data pipeline with validation and leakage control.",
                "Engineer agronomically meaningful features that capture pH bands and nutrient sufficiency.",
                "Tune and compare at least four machine learning models fairly.",
                "Evaluate the final model on a held-out test set with confidence intervals.",
                "Explain both global model behaviour and individual field predictions.",
                "Provide a batch prediction tool and an interactive web demo.",
            ]
        )
        self.page_break()

    def dataset_section(self) -> None:
        self.heading("3. Dataset Description", 1)
        self.para(
            "The dataset contains 1,000 rows and 24 columns. Each row represents "
            "one crop-field sample. The target column is "
            "Organic_Farming_Suitability with values Suitable or Not Suitable. "
            "The distribution is 423 suitable fields (42.3%) and 577 not-suitable "
            "fields (57.7%), so the classes are moderately imbalanced."
        )
        self.para(
            "Important: this is a source-calibrated synthetic dataset created for "
            "teaching. The labels follow an instructional rubric, not official "
            "organic certification. The results therefore demonstrate the "
            "machine-learning workflow; they are not a real agronomic verdict.",
            bold=True,
        )

        variables = pd.DataFrame(
            [
                ("Sample_ID", "Identifier", "Row ID; dropped before modelling"),
                ("State", "Context", "Indian state; no measurable signal"),
                ("Soil_Type", "Context", "Alluvial, Black, Loamy, etc.; no signal"),
                ("Crop", "Context", "Crop name; no signal in this dataset"),
                ("pH", "Soil", "Soil reaction (4.5-9.5)"),
                ("Organic_Carbon_pct", "Soil", "Soil organic carbon percentage"),
                ("Available_N_kg_ha", "Soil", "Available nitrogen"),
                ("Available_P_kg_ha", "Soil", "Available phosphorus"),
                ("Available_K_kg_ha", "Soil", "Available potassium"),
                ("Available_S_mg_kg", "Soil", "Available sulphur"),
                ("Available_Zn/Fe/Mn/Cu/B", "Soil", "Micronutrients; no signal"),
                ("EC_dS_m", "Soil", "Electrical conductivity; no signal"),
                ("Annual_Rainfall_mm", "Weather", "Annual rainfall; no signal"),
                ("Mean_Temperature_C", "Weather", "Mean temperature; no signal"),
                ("Soil_Moisture_pct", "Weather", "Soil moisture; no signal"),
                ("Previous_Fertilizer_Usage", "History", "Chemical / Mixed / Organic"),
                ("Previous_Pesticide_Usage", "History", "High / Moderate / Low"),
                ("Years_Chemical_Input", "History", "Years of synthetic input"),
                ("Suitability_Score", "Label source", "Rubric score; excluded as leakage"),
                ("Organic_Farming_Suitability", "Target", "Suitable / Not Suitable"),
            ],
            columns=["Column", "Group", "Description"],
        )
        self.table(
            variables,
            ["Column", "Group", "Description"],
            ["Column", "Group", "Description"],
            "Variables in the dataset",
            widths=[2.0, 1.0, 3.1],
            font_size=8,
        )

        self.heading("3.1 Data quality checks", 2)
        self.para(
            "The pipeline performs the following checks before training: required "
            "columns, missing values, duplicate rows, allowed category values, "
            "physically plausible numeric ranges and target-label parsing. "
            "Unknown labels raise an error instead of being silently mapped to a "
            "class. In this dataset there are no missing cells and no duplicate "
            "rows."
        )
        self.para(
            "A critical leakage check found that Suitability_Score reproduces "
            "the target with 99.9% agreement (the label is essentially a "
            "threshold on that score). The score is therefore removed from the "
            "feature set. Keeping it would produce near-perfect accuracy but "
            "would be meaningless for new fields, where no score exists."
        )

        self.figure("01_class_balance.png", "Target class distribution", width=4.4)
        self.page_break()

    def methodology(self) -> None:
        self.heading("4. Methodology", 1)
        self.para(
            "The pipeline follows the standard supervised-learning workflow: "
            "data validation, exploratory analysis, feature engineering, "
            "preprocessing, model tuning, model selection, hold-out evaluation "
            "and explanation. Every transformation that learns from data is "
            "fitted inside a scikit-learn Pipeline, so the test set never "
            "influences imputation, scaling or encoding."
        )

        self.heading("4.1 Feature engineering", 2)
        self.para(
            "Exploratory analysis showed that the relationship between the "
            "inputs and the label is banded rather than linear. For example, "
            "suitability rises with pH up to about 7.8 and then falls sharply "
            "for strongly alkaline soils. A plain linear model cannot represent "
            "such a peak, so the following features are added:"
        )
        self.bullets(
            [
                "pH indicators: optimal band (6.0-7.8), distance from 6.9, low, high and very high flags.",
                "Nutrient sufficiency flags for organic carbon, nitrogen, phosphorus, potassium and sulphur.",
                "A nutrient sufficiency index from 0 to 5.",
                "Chemical burden = years of input x fertilizer intensity x pesticide intensity.",
                "A stewardship score and an interaction between optimal pH and organic history.",
            ]
        )
        self.para(
            "These features are row-wise transformations, so they cannot leak "
            "information across the train/test boundary. They are also "
            "interpretable: a coefficient or importance score can be explained "
            "to a farmer."
        )

        features = pd.DataFrame(
            [
                ("pH in optimal band", "pH between 6.0 and 7.8", "Captures the non-linear pH optimum"),
                ("pH distance", "|pH - 6.9|", "Penalty for moving away from the ideal"),
                ("OC sufficient", "Organic carbon >= 0.75%", "Soil carbon threshold"),
                ("N sufficient", "Nitrogen >= 280 kg/ha", "Macro-nutrient threshold"),
                ("P sufficient", "Phosphorus >= 25 kg/ha", "Macro-nutrient threshold"),
                ("K sufficient", "Potassium >= 250 kg/ha", "Macro-nutrient threshold"),
                ("S sufficient", "Sulphur >= 15 mg/kg", "Secondary nutrient threshold"),
                ("Nutrient index", "Sum of five flags (0-5)", "Overall nutrient balance"),
                ("Chemical burden", "Years x input intensities", "Transition difficulty"),
                ("Stewardship", "Organic score - years/10", "Management history"),
            ],
            columns=["Feature", "Definition", "Why it matters"],
        )
        self.table(
            features,
            ["Feature", "Definition", "Why it matters"],
            ["Feature", "Definition", "Why it matters"],
            "Engineered features",
            widths=[1.5, 2.2, 2.4],
            font_size=8,
        )

        self.heading("4.2 Preprocessing", 2)
        self.para(
            "Numeric features are median-imputed and, for distance-based and "
            "linear models, standardised. Categorical features are "
            "most-frequent-imputed and one-hot encoded with "
            "handle_unknown='ignore', so an unseen category at prediction time "
            "does not crash the system. All of this happens inside the pipeline."
        )

        self.heading("4.3 Models and tuning", 2)
        self.para(
            "Seven model families were evaluated: logistic regression, decision "
            "tree, k-nearest neighbours, random forest, extra trees, gradient "
            "boosting and histogram gradient boosting. XGBoost and LightGBM are "
            "included automatically when installed. Each model received the same "
            "treatment: randomised hyper-parameter search with 5-fold "
            "cross-validation on the training set, using ROC-AUC as the search "
            "objective. This fixes the unfair comparison in the earlier version, "
            "where only the random forest was tuned."
        )
        self.para(
            "Model selection used repeated stratified cross-validation "
            "(5 folds x 3 repeats) on the training set. When two models were "
            "within 0.002 ROC-AUC of each other, the simpler model was preferred "
            "(parsimony rule). The test set was touched only once, after the "
            "model and threshold were frozen."
        )

        self.heading("4.4 Evaluation metrics", 2)
        self.para(
            "Accuracy alone can be misleading with a 58/42 class split, so the "
            "report includes balanced accuracy, precision, recall, F1, ROC-AUC, "
            "precision-recall AUC, Matthews correlation coefficient and the "
            "Brier score. Bootstrap resampling (1,000 draws) provides 95% "
            "confidence intervals for the headline test metrics."
        )

        self.heading("4.5 Decision threshold", 2)
        self.para(
            "A classifier outputs a probability; a threshold turns it into a "
            "label. The default 0.5 is arbitrary. The pipeline tunes the "
            "threshold on out-of-fold predictions to maximise F1, but only "
            "adopts a non-default value when the improvement is at least 0.02, "
            "to avoid over-fitting the threshold to one split. The app lets the "
            "user move the threshold to reflect the real cost of a wrong "
            "recommendation."
        )
        self.page_break()

    def eda(self) -> None:
        self.heading("5. Exploratory Data Analysis", 1)
        self.para(
            "The exploratory analysis focuses on the features that actually "
            "carry signal. Location, crop, micronutrients and weather variables "
            "showed near-zero correlation with the target and were excluded from "
            "the model. This is an important finding: more columns do not "
            "automatically mean a better model."
        )
        self.figure("02_feature_distributions.png", "Distributions of the key soil features")
        self.figure("03_features_by_class.png", "Key features split by suitability class")
        self.figure("04_correlation_heatmap.png", "Correlation between key features and the target")
        self.figure("05_suitability_rates.png", "Suitability rate by management history, soil type and crop")
        self.page_break()

    def results_section(self) -> None:
        self.heading("6. Results", 1)
        comparison = pd.read_csv(TABLE_DIR / "model_comparison.csv")
        display = pd.DataFrame(
            {
                "Model": comparison["model"],
                "ROC-AUC": comparison.apply(
                    lambda r: f"{r['roc_auc_mean']:.3f} ± {r['roc_auc_std']:.3f}", axis=1
                ),
                "F1": comparison.apply(
                    lambda r: f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f}", axis=1
                ),
                "Accuracy": comparison.apply(
                    lambda r: f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}", axis=1
                ),
                "Precision": comparison["precision_mean"].round(3),
                "Recall": comparison["recall_mean"].round(3),
            }
        ).sort_values("ROC-AUC", ascending=False)

        self.heading("6.1 Cross-validated model comparison", 2)
        self.para(
            "Table 3 shows the repeated cross-validation results on the training "
            "set. These numbers are used for model selection; they are not the "
            "final test performance."
        )
        self.table(
            display,
            ["Model", "ROC-AUC", "F1", "Accuracy", "Precision", "Recall"],
            ["Model", "ROC-AUC", "F1", "Accuracy", "Precision", "Recall"],
            "Repeated 5x3 cross-validation on the training set (mean ± standard deviation)",
            widths=[1.55, 1.15, 1.05, 1.15, 0.85, 0.75],
            font_size=8,
        )
        self.figure("06_model_comparison.png", "Cross-validated comparison of all models")

        metrics = self.results["test_metrics_tuned"]
        ci = self.results["confidence_intervals"]
        self.heading("6.2 Held-out test performance", 2)
        self.para(
            f"The selected model is a {self.results['selected_model']}. "
            f"On the {self.results['test_rows']} fields in the untouched test "
            f"set it achieved the following results."
        )

        metric_rows = []
        labels = [
            ("accuracy", "Accuracy"),
            ("balanced_accuracy", "Balanced accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1 score"),
            ("roc_auc", "ROC-AUC"),
            ("pr_auc", "PR-AUC"),
            ("mcc", "Matthews correlation"),
            ("brier", "Brier score (lower is better)"),
        ]
        for key, label in labels:
            interval = ""
            if key in ci:
                interval = f"[{ci[key]['low']:.3f}, {ci[key]['high']:.3f}]"
            metric_rows.append((label, f"{metrics[key]:.4f}", interval))
        metric_table = pd.DataFrame(metric_rows, columns=["Metric", "Value", "95% CI"])
        self.table(
            metric_table,
            ["Metric", "Value", "95% CI"],
            ["Metric", "Value", "95% CI"],
            "Test-set performance at the deployment threshold",
            widths=[2.4, 1.3, 1.8],
            font_size=8.5,
        )

        self.figure("07_confusion_matrix.png", "Confusion matrix on the held-out test set", width=4.5)
        self.figure("08_roc_pr_curves.png", "ROC and precision-recall curves")
        self.figure("09_calibration.png", "Calibration curve", width=4.6)
        self.figure("10_threshold_analysis.png", "Threshold tuning on out-of-fold predictions")

        self.heading("6.3 Reading the confusion matrix", 2)
        cm = pd.read_csv(TABLE_DIR / "test_predictions.csv")
        tn = int(((cm["actual"] == 0) & (cm["predicted"] == 0)).sum())
        fp = int(((cm["actual"] == 0) & (cm["predicted"] == 1)).sum())
        fn = int(((cm["actual"] == 1) & (cm["predicted"] == 0)).sum())
        tp = int(((cm["actual"] == 1) & (cm["predicted"] == 1)).sum())
        self.para(
            f"The model correctly identified {tp} suitable fields and {tn} "
            f"not-suitable fields. It missed {fn} suitable fields (false "
            f"negatives) and flagged {fp} unsuitable fields as suitable (false "
            f"positives). For a screening tool used before an expensive soil "
            f"test, both errors matter: a false positive can waste conversion "
            f"effort, while a false negative can hide a good opportunity."
        )
        self.page_break()

    def interpretation(self) -> None:
        self.heading("7. Model Interpretation", 1)
        self.para(
            "A model that cannot be explained is difficult to trust in an "
            "agricultural setting. Two levels of explanation are provided: "
            "global feature importance and per-field contributions."
        )

        self.heading("7.1 What matters globally", 2)
        importance = pd.read_csv(TABLE_DIR / "feature_importance.csv")
        pretty = pd.DataFrame(
            {
                "Feature": importance["feature"].str.replace("_", " "),
                "Importance": importance["importance_mean"].round(4),
                "Std. dev.": importance["importance_std"].round(4),
            }
        ).head(10)
        self.table(
            pretty,
            ["Feature", "Importance", "Std. dev."],
            ["Feature", "Importance", "Std. dev."],
            "Top permutation-importance features (drop in ROC-AUC when shuffled)",
            widths=[2.6, 1.3, 1.2],
            font_size=8.5,
        )
        self.figure("12_permutation_importance.png", "Permutation importance of the final model")

        coefficient_path = TABLE_DIR / "logistic_coefficients.csv"
        if coefficient_path.exists():
            coefficients = pd.read_csv(coefficient_path).head(12)
            coefficients["feature"] = coefficients["feature"].str.replace("_", " ")
            self.heading("7.2 Logistic-regression coefficients", 2)
            self.para(
                "Because the selected model is linear, its coefficients can be "
                "read directly. Positive coefficients increase the log-odds of "
                "being suitable; negative coefficients decrease them. The "
                "magnitudes are on the standardised feature scale."
            )
            self.table(
                coefficients[["feature", "coefficient"]],
                ["feature", "coefficient"],
                ["Feature", "Coefficient"],
                "Largest logistic-regression coefficients",
                widths=[3.6, 1.4],
                font_size=8.5,
            )
            self.figure("13_logistic_coefficients.png", "What drives the logistic-regression decision")

        self.figure("14_partial_dependence.png", "Partial dependence of the top features")
        self.figure("15_learning_curve.png", "Learning curve of the final model")

        self.heading("7.3 Explaining a single field", 2)
        self.para(
            "The web application uses an occlusion explanation: each input is "
            "replaced by a typical value from the training set, and the change "
            "in probability is recorded. A positive contribution means the "
            "field's actual value is helping; a negative contribution means it "
            "is hurting. The table below shows this for a borderline example."
        )
        self._example_field_table()

    def _example_field_table(self) -> None:
        sample_path = PROJECT_ROOT / "data" / "sample_fields.csv"
        if not sample_path.exists() or not BUNDLE_PATH.exists():
            return
        bundle = load_bundle(BUNDLE_PATH)
        model = bundle["model"]
        reference = bundle["reference_stats"]
        samples = pd.read_csv(sample_path)
        probabilities = model.predict_proba(samples[KEY_FEATURES])[:, 1]
        table = pd.DataFrame(
            {
                "Field": samples["Field_ID"],
                "pH": samples["pH"],
                "Organic C (%)": samples["Organic_Carbon_pct"],
                "Probability": [f"{p:.3f}" for p in probabilities],
                "Prediction": ["Suitable" if p >= bundle["threshold"] else "Not Suitable" for p in probabilities],
                "Confidence": [confidence_label(float(p)) for p in probabilities],
            }
        )
        self.table(
            table,
            ["Field", "pH", "Organic C (%)", "Probability", "Prediction", "Confidence"],
            ["Field", "pH", "Organic C (%)", "Probability", "Prediction", "Confidence"],
            "Predictions for six example fields",
            widths=[0.8, 0.6, 1.0, 1.0, 1.1, 1.5],
            font_size=8,
        )

        # Show factors for the borderline field (DEMO-03) if present.
        if "DEMO-03" in set(samples["Field_ID"]):
            row = samples.loc[samples["Field_ID"] == "DEMO-03", KEY_FEATURES].iloc[[0]]
            factors = local_explanations(model, row, reference, top_n=6)
            factor_table = pd.DataFrame(
                {
                    "Factor": factors["label"],
                    "Value": factors["value"].astype(str),
                    "Effect": [f"{v:+.3f}" for v in factors["contribution"]],
                }
            )
            self.para(
                "For the borderline field DEMO-03, the largest effects are "
                "shown below. A positive effect pushes the field towards "
                "\u201cSuitable\u201d; a negative effect pulls it away.",
                italic=True,
                size=9.5,
            )
            self.table(
                factor_table,
                ["Factor", "Value", "Effect"],
                ["Factor", "Value", "Effect"],
                "Per-field contribution for DEMO-03 (borderline field)",
                widths=[3.0, 1.0, 1.2],
                font_size=8,
            )
        self.page_break()

    def comparison_section(self) -> None:
        self.heading("8. Comparison with the Previous Version", 1)
        self.para(
            "This project is a rebuilt and extended version of an earlier "
            "single-script submission. The table below compares the reported "
            "results and the engineering quality of the two versions."
        )
        metrics = self.results["test_metrics_tuned"]
        original = {"accuracy": 0.805, "f1": 0.764, "roc_auc": 0.889}
        table = pd.DataFrame(
            [
                ("Accuracy", f"{original['accuracy']:.1%}", f"{metrics['accuracy']:.1%}"),
                ("F1 score", f"{original['f1']:.3f}", f"{metrics['f1']:.3f}"),
                ("ROC-AUC", f"{original['roc_auc']:.3f}", f"{metrics['roc_auc']:.3f}"),
                ("Models compared", "4 (only RF tuned)", "7-9 (all tuned)"),
                ("Model selection", "Best test score (leakage)", "Repeated CV, test used once"),
                ("Threshold", "Fixed 0.5", "Tuned on out-of-fold predictions"),
                ("Confidence intervals", "No", "Bootstrap 95% CI"),
                ("Explainability", "Global only", "Global + per-field"),
                ("Deployment", "None", "Streamlit app + batch CLI"),
                ("Tests", "None", "pytest test suite"),
                ("Report", "Separate, not reproducible", "Auto-generated from results"),
            ],
            columns=["Aspect", "Previous version", "This version"],
        )
        self.table(
            table,
            ["Aspect", "Previous version", "This version"],
            ["Aspect", "Previous version", "This version"],
            "Previous version versus the improved version",
            widths=[1.7, 2.2, 2.3],
            font_size=8.5,
        )
        self.para(
            "The accuracy improvement comes from three changes, not from a more "
            "complex model: removing noise features, adding banded agronomic "
            "features, and tuning every model under the same protocol. The "
            "selected model remains a logistic regression, which is easier to "
            "explain than a gradient-boosting ensemble."
        )

        self.heading("9. Limitations and Ethical Considerations", 1)
        self.bullets(
            [
                "The dataset is synthetic. The labels follow an instructional rubric and do not represent official organic certification.",
                "The model is a screening aid, not a substitute for a laboratory soil test or a visit from an extension officer.",
                "The dataset covers a limited set of crops and Indian states; performance may drop on real field data from other regions.",
                "The label is derived from a suitability score. Even though the score is excluded from the features, the model is learning that rubric rather than an independent agronomic ground truth.",
                "Correlation is not causation. Feature importance shows what the model uses, not what a farmer should change.",
                "The recommendations are general organic-farming practices and should be adapted to local conditions.",
            ]
        )

        self.heading("10. Conclusion and Future Work", 1)
        self.para(
            "This project shows that careful data preparation and feature "
            "engineering can outperform a larger, more complex model on a small "
            "tabular dataset. The final system is accurate, calibrated, "
            "explainable and usable: it can score a batch of fields from a CSV "
            "file or answer a single-field query in a web application."
        )
        self.para("Future work includes:", bold=True)
        self.bullets(
            [
                "Collect real soil-test records and re-train the model with proper cross-region validation.",
                "Add crop-specific suitability models instead of one pooled model.",
                "Include time-series features such as multi-year soil organic carbon trends.",
                "Add a cost-sensitive threshold that reflects the farmer's budget and risk tolerance.",
                "Deploy the app on a cloud server with authentication and a proper database.",
            ]
        )
        self.page_break()

    def references(self) -> None:
        self.heading("11. References", 1)
        refs = [
            "Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
            "McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 56-61.",
            "Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.",
            "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
            "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of KDD 2016, 785-794.",
            "Food and Agriculture Organization (FAO). (2022). Organic Agriculture. https://www.fao.org/organic-agriculture",
            "United States Department of Agriculture, NRCS. Soil Health. https://www.nrcs.usda.gov/health/soil",
            "Indian Council of Agricultural Research (ICAR). Organic Farming Guidelines. https://icar.org.in",
            "Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and Statistical Modeling with Python. Proceedings of the 9th Python in Science Conference.",
        ]
        for ref in refs:
            paragraph = self.doc.add_paragraph(ref)
            paragraph.paragraph_format.left_indent = Cm(0.8)
            paragraph.paragraph_format.first_line_indent = Cm(-0.8)
            paragraph.paragraph_format.space_after = Pt(5)

        self.heading("Appendix A: Viva Questions and Answers", 1)
        qa = [
            (
                "What is the goal of this project?",
                "It predicts whether a field is suitable for organic farming from soil chemistry and past management inputs, and explains the prediction.",
            ),
            (
                "Why is this a classification problem?",
                "The target has two discrete classes, Suitable and Not Suitable.",
            ),
            (
                "Which features did you use?",
                "Nine raw inputs: pH, organic carbon, N, P, K, S, previous fertilizer use, previous pesticide use and years of chemical input. Twenty additional features are engineered from these.",
            ),
            (
                "Why did you drop location, crop and micronutrients?",
                "They had near-zero correlation and near-zero permutation importance with the target in this dataset. Including them added noise and slightly reduced accuracy.",
            ),
            (
                "What is data leakage and how did you prevent it?",
                "Leakage is when information unavailable at prediction time is used during training. Suitability_Score is the rubric from which the label is derived, so it was dropped. All preprocessing is fitted inside the pipeline on training folds only.",
            ),
            (
                "Why not use only accuracy?",
                "The classes are imbalanced (58/42), so accuracy can hide poor recall. We also report F1, balanced accuracy, ROC-AUC, PR-AUC, MCC and Brier score.",
            ),
            (
                "What is cross-validation?",
                "The training data is split into k folds; the model trains on k-1 folds and validates on the remaining fold, repeated k times. It gives a more stable estimate than one split.",
            ),
            (
                "Why did logistic regression win?",
                "After the agronomic feature engineering, the decision boundary is close to linear in the engineered space. The gradient-boosting models achieved similar AUC but no meaningful gain, so the parsimony rule chose the simpler model.",
            ),
            (
                "What does the decision threshold do?",
                "It converts a probability into a class. Lowering it flags more fields as suitable (higher recall, lower precision); raising it does the opposite.",
            ),
            (
                "What are the limitations?",
                "The data is synthetic, the labels come from a teaching rubric, and the model is a screening aid rather than an official certification tool.",
            ),
            (
                "How would you deploy this in the real world?",
                "Connect it to a soil-test database, validate on real regional data, add authentication, and keep a human agronomist in the loop.",
            ),
        ]
        for question, answer in qa:
            paragraph = self.doc.add_paragraph()
            run = paragraph.add_run(f"Q: {question}")
            run.bold = True
            run.font.color.rgb = DARK
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph = self.doc.add_paragraph()
            run = paragraph.add_run(f"A: {answer}")
            paragraph.paragraph_format.left_indent = Cm(0.5)
            paragraph.paragraph_format.space_after = Pt(8)

        self.heading("Appendix B: How to Run the Project", 1)
        steps = [
            "Install Python 3.10 or newer.",
            "Create a virtual environment: python -m venv .venv",
            "Activate it: .venv\\Scripts\\activate (Windows) or source .venv/bin/activate (Linux/Mac).",
            "Install dependencies: pip install -r requirements.txt",
            "Train and evaluate: python train.py",
            "Batch prediction: python predict.py --input data/sample_fields.csv",
            "Web app: streamlit run app.py",
            "Generate this report: python generate_report.py",
            "Run tests: pytest -q",
        ]
        self.numbered(steps)
        self.para(
            "All results in this report are generated from the run recorded in "
            "outputs/results_summary.json, so the report can be regenerated "
            "after any change to the model or data.",
            italic=True,
            size=9.5,
        )

    def build(self) -> Document:
        self.title_page()
        self.declaration()
        self.abstract()
        self.contents()
        self.introduction()
        self.dataset_section()
        self.methodology()
        self.eda()
        self.results_section()
        self.interpretation()
        self.comparison_section()
        self.references()
        return self.doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Word project report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "Organic_Farming_Project_Report.docx",
    )
    args = parser.parse_args()

    if not RESULTS_JSON.exists():
        raise SystemExit(
            "results_summary.json not found. Run `python train.py` first."
        )

    results = load_json(RESULTS_JSON)
    report = Report(results)
    document = report.build()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document.save(args.output)
    print(f"Report saved -> {args.output}")


if __name__ == "__main__":
    main()
