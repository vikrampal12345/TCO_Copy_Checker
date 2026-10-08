from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
import re

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from app.schemas.marking_scheme import (
    MarkingCriterion,
    MarkingScheme,
    QuestionMarkingScheme,
)
from app.services.checked_copy_renderer import CheckedCopyRenderer
from app.services.grading_model import GradingModel
from app.services.in_memory_answer_extraction_service import (
    InMemoryAnswerExtractionService,
)
from app.services.in_memory_grading_workflow import (
    InMemoryGradingWorkflow,
)
from app.services.in_memory_page_processor import (
    InMemoryPageProcessor,
)
from app.services.submission_aggregation_service import (
    SubmissionAggregationService,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

JOB_ROOT = (
    BASE_DIR
    / "data"
    / "api_jobs"
)

JOB_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="TCO Copy Checker - Live Test API",
    description=(
        "Temporary API for testing Question Paper + "
        "Marking Scheme + Handwritten Answer Copy."
    ),
    version="0.1.0",
)


# ============================================================
# HTML TEST UI
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TCO Copy Checker</title>

    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f7fb;
            color: #172033;
        }

        .container {
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
        }

        .header {
            margin-bottom: 24px;
        }

        .header h1 {
            margin-bottom: 8px;
        }

        .header p {
            color: #637083;
        }

        .card {
            background: white;
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 18px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        }

        .file-box {
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 16px;
        }

        .file-box label {
            display: block;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .file-box small {
            display: block;
            color: #64748b;
            margin-bottom: 10px;
        }

        input[type="file"] {
            width: 100%;
        }

        select,
        button {
            width: 100%;
            padding: 13px;
            border-radius: 9px;
            border: 1px solid #cbd5e1;
            font-size: 15px;
        }

        button {
            margin-top: 18px;
            border: none;
            background: #1769e0;
            color: white;
            font-weight: 700;
            cursor: pointer;
        }

        button:disabled {
            background: #94a3b8;
            cursor: wait;
        }

        #status {
            margin-top: 14px;
            font-weight: 600;
        }

        pre {
            background: #0f172a;
            color: #e2e8f0;
            padding: 18px;
            border-radius: 10px;
            overflow-x: auto;
            white-space: pre-wrap;
        }

        .download {
            display: inline-block;
            margin-top: 16px;
            padding: 12px 18px;
            border-radius: 8px;
            background: #16a34a;
            color: white;
            text-decoration: none;
            font-weight: 700;
        }
    </style>
</head>

<body>

<div class="container">

    <div class="header">
        <h1>TCO Copy Checker</h1>
        <p>
            Upload question paper, marking scheme and handwritten
            answer copy to test the live grading pipeline.
        </p>
    </div>

    <div class="card">

        <div class="file-box">
            <label>Question Paper</label>
            <small>TXT question paper</small>
            <input
                id="questionPaper"
                type="file"
                accept=".txt"
                required
            >
        </div>

        <div class="file-box">
            <label>Marking Scheme / Answer Key</label>
            <small>
                JSON or TXT containing the TCO MarkingScheme structure.
            </small>
            <input
                id="markingScheme"
                type="file"
                accept=".json,.txt,.pdf"
                required
            >
        </div>

        <div class="file-box">
            <label>Student Answer Copy</label>
            <small>Handwritten PDF</small>
            <input
                id="answerCopy"
                type="file"
                accept=".pdf"
                required
            >
        </div>

        <div class="file-box">
            <label>Pages to Process</label>
            <small>
                Start small because handwritten extraction uses GPU.
            </small>

            <select id="pages">
                <option value="1">1 page</option>
                <option value="2" selected>2 pages</option>
                <option value="3">3 pages</option>
                <option value="4">4 pages</option>
                <option value="10">10 pages</option>
            </select>
        </div>

        <button id="runButton" onclick="runGrading()">
            Upload & Run Grading
        </button>

        <div id="status"></div>

    </div>

    <div class="card">
        <h2>Result</h2>
        <pre id="result">No grading run yet.</pre>
        <a
            id="downloadLink"
            class="download"
            href="#"
            target="_blank"
            style="display:none;"
        >
            View Checked Copy
        </a>
    </div>

</div>


<script>

async function runGrading() {

    const questionPaper =
        document.getElementById("questionPaper").files[0];

    const markingScheme =
        document.getElementById("markingScheme").files[0];

    const answerCopy =
        document.getElementById("answerCopy").files[0];

    const pages =
        document.getElementById("pages").value;

    const button =
        document.getElementById("runButton");

    const status =
        document.getElementById("status");

    const result =
        document.getElementById("result");

    const downloadLink =
        document.getElementById("downloadLink");


    if (!questionPaper || !markingScheme || !answerCopy) {
        alert(
            "Please upload Question Paper, Marking Scheme and Answer Copy."
        );
        return;
    }


    const formData = new FormData();

    formData.append(
        "question_paper",
        questionPaper
    );

    formData.append(
        "marking_scheme",
        markingScheme
    );

    formData.append(
        "answer_copy",
        answerCopy
    );


    button.disabled = true;
    button.innerText = "Processing...";

    status.innerText =
        "Running extraction and grading. Please wait...";

    result.textContent =
        "Processing...";

    downloadLink.style.display = "none";


    try {

        const response = await fetch(
            "/api/test/grade?pages=" + pages,
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok) {
            throw new Error(
                data.detail || "Grading failed."
            );
        }


        result.textContent =
            JSON.stringify(
                data,
                null,
                2
            );


        if (data.checked_copy_url) {

            downloadLink.href =
                data.checked_copy_url;

            downloadLink.style.display =
                "inline-block";
        }


        status.innerText =
            "Grading completed successfully.";

    }
    catch (error) {

        result.textContent =
            error.message;

        status.innerText =
            "Grading failed.";

    }
    finally {

        button.disabled = false;

        button.innerText =
            "Upload & Run Grading";
    }
}

</script>

</body>
</html>
"""


# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def home() -> str:
    return HTML_PAGE


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "tco-copy-checker-test-api",
    }


# ============================================================
# SAVE UPLOAD
# ============================================================

async def save_upload(
    upload: UploadFile,
    destination: Path,
) -> None:

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
        "wb"
    ) as file:

        while True:

            chunk = await upload.read(
                1024 * 1024
            )

            if not chunk:
                break

            file.write(chunk)

    await upload.close()


# ============================================================
# LOAD MARKING SCHEME
# ============================================================



def load_marking_scheme(
    path: Path,
) -> MarkingScheme:
    """
    Load marking scheme from the current TCO TXT format.

    Example:

    ASSESSMENT_ID: ML-TEST-001
    STATUS: locked

    MARKS:
    MCQ: 1
    SHORT_ANSWER: 2
    TOTAL_MARKS: 50

    Question_Number,Type,Correct_Answer
    1,MCQ,B
    2,MCQ,C
    ...
    21,Short_Answer,"reference answer"
    """

    import csv
    from io import StringIO

    if path.suffix.lower() != ".txt":
        raise HTTPException(
            status_code=400,
            detail="Marking scheme must be a .txt file.",
        )

    raw_text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not raw_text:
        raise HTTPException(
            status_code=400,
            detail="Marking scheme TXT file is empty.",
        )

    lines = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip()
        and not line.strip().startswith("#")
    ]

    assessment_id = None
    status = None

    mcq_marks = None
    short_answer_marks = None
    declared_total_marks = None

    data_start_index = None

    # --------------------------------------------------------
    # Read metadata and marks section
    # --------------------------------------------------------

    for index, line in enumerate(lines):

        upper = line.upper()

        if upper.startswith("ASSESSMENT_ID:"):

            assessment_id = line.split(
                ":",
                1,
            )[1].strip()

            continue

        if upper.startswith("STATUS:"):

            status = line.split(
                ":",
                1,
            )[1].strip().lower()

            continue

        if upper.startswith("MCQ:"):

            value = line.split(
                ":",
                1,
            )[1].strip()

            try:
                mcq_marks = float(value)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid MCQ marks value.",
                ) from exc

            continue

        if upper.startswith("SHORT_ANSWER:"):

            value = line.split(
                ":",
                1,
            )[1].strip()

            try:
                short_answer_marks = float(value)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid SHORT_ANSWER marks value."
                    ),
                ) from exc

            continue

        if upper.startswith("TOTAL_MARKS:"):

            value = line.split(
                ":",
                1,
            )[1].strip()

            try:
                declared_total_marks = float(value)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid TOTAL_MARKS value.",
                ) from exc

            continue

        # ----------------------------------------------------
        # CSV header
        # ----------------------------------------------------

        if (
            "QUESTION_NUMBER" in upper
            and "TYPE" in upper
            and "CORRECT_ANSWER" in upper
        ):

            data_start_index = index + 1
            break


    # --------------------------------------------------------
    # Validate metadata
    # --------------------------------------------------------

    if not assessment_id:

        raise HTTPException(
            status_code=400,
            detail="ASSESSMENT_ID is required.",
        )

    if status != "locked":

        raise HTTPException(
            status_code=400,
            detail="STATUS must be 'locked' before grading.",
        )

    if mcq_marks is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "MCQ marks are required. "
                "Add 'MCQ: 1'."
            ),
        )

    if short_answer_marks is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "SHORT_ANSWER marks are required. "
                "Add 'SHORT_ANSWER: 2'."
            ),
        )

    if data_start_index is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not find answer-key CSV header: "
                "Question_Number,Type,Correct_Answer"
            ),
        )


    # --------------------------------------------------------
    # Parse CSV rows
    # --------------------------------------------------------

    # Include the CSV header because DictReader
    # uses it as the field-name row.
    csv_text = "\n".join(
        lines[data_start_index - 1:]
    )

    reader = csv.DictReader(
        StringIO(csv_text)
    )

    questions = []


    for row_number, row in enumerate(
        reader,
        start=1,
    ):

        question_number_raw = (
            row.get("Question_Number")
            or row.get("question_number")
        )

        question_type_raw = (
            row.get("Type")
            or row.get("type")
        )

        correct_answer_raw = (
            row.get("Correct_Answer")
            or row.get("correct_answer")
            or ""
        )

        if not question_number_raw:
            continue

        try:
            question_no = int(
                str(
                    question_number_raw
                ).strip()
            )

        except ValueError as exc:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid Question_Number "
                    f"at CSV row {row_number}."
                ),
            ) from exc


        question_type_text = (
            str(
                question_type_raw or ""
            )
            .strip()
            .lower()
        )

        answer = (
            str(
                correct_answer_raw
            )
            .strip()
            .strip('"')
        )


        # ----------------------------------------------------
        # Type mapping
        # ----------------------------------------------------

        if question_type_text == "mcq":

            question_type = "mcq"
            max_marks = mcq_marks

            if not answer:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Q{question_no}: "
                        "MCQ requires a correct answer."
                    ),
                )

            correct_answer = answer
            guidance = (
                "Use the approved MCQ answer key."
            )


        elif question_type_text in {
            "short_answer",
            "short-answer",
            "short answer",
        }:

            question_type = "subjective"
            max_marks = short_answer_marks
            correct_answer = None
            guidance = answer or None


        else:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Q{question_no}: unsupported "
                    f"question type '{question_type_raw}'."
                ),
            )


        questions.append(
            QuestionMarkingScheme(
                question_no=question_no,
                max_marks=max_marks,
                question_type=question_type,
                correct_answer=correct_answer,
                criteria=[],
                evaluation_guidance=guidance,
            )
        )


    # --------------------------------------------------------
    # Validate questions
    # --------------------------------------------------------

    if not questions:

        raise HTTPException(
            status_code=400,
            detail=(
                "No questions found in marking scheme TXT."
            ),
        )


    question_numbers = [
        question.question_no
        for question in questions
    ]

    duplicates = {
        number
        for number in question_numbers
        if question_numbers.count(number) > 1
    }

    if duplicates:

        raise HTTPException(
            status_code=400,
            detail=(
                "Duplicate question number(s): "
                + ", ".join(
                    str(number)
                    for number in sorted(
                        duplicates
                    )
                )
            ),
        )


    # --------------------------------------------------------
    # Calculate total marks
    # --------------------------------------------------------

    calculated_total = sum(
        question.max_marks
        for question in questions
    )


    if declared_total_marks is not None:

        if abs(
            calculated_total
            - declared_total_marks
        ) > 1e-6:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"TOTAL_MARKS={declared_total_marks} "
                    f"does not match calculated total="
                    f"{calculated_total}."
                ),
            )


    return MarkingScheme(
        assessment_id=assessment_id,
        version=1,
        questions=questions,
        total_marks=calculated_total,
        status="locked",
        generated_by="teacher",
    )


# ============================================================
# GRADE API
# ============================================================

@app.post("/api/test/grade")
async def grade_uploaded_copy(
    question_paper: UploadFile = File(...),
    marking_scheme: UploadFile = File(...),
    answer_copy: UploadFile = File(...),
    pages: int = Query(
        default=2,
        ge=1,
        le=50,
    ),
) -> dict:

    job_id = (
        "API-"
        + uuid.uuid4().hex[:12].upper()
    )


    job_dir = (
        JOB_ROOT
        / job_id
    )

    input_dir = (
        job_dir
        / "input"
    )

    output_dir = (
        job_dir
        / "output"
    )

    input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    question_paper_path = (
        input_dir
        / "question_paper.txt"
    )

    marking_scheme_suffix = (
        Path(
            marking_scheme.filename or "marking_scheme.txt"
        ).suffix.lower()
        or ".txt"
    )

    marking_scheme_path = (
        input_dir
        / f"marking_scheme{marking_scheme_suffix}"
    )

    answer_copy_suffix = Path(
        answer_copy.filename or "answer.pdf"
    ).suffix or ".pdf"

    answer_copy_path = (
        input_dir
        / f"answer_copy{answer_copy_suffix}"
    )


    # --------------------------------------------------------
    # Validate question paper
    # --------------------------------------------------------

    question_paper_suffix = Path(
        question_paper.filename or ""
    ).suffix.lower()

    if question_paper_suffix != ".txt":
        raise HTTPException(
            status_code=400,
            detail="Question paper must be a .txt file.",
        )

    # --------------------------------------------------------
    # Save all three uploaded files
    # --------------------------------------------------------

    await save_upload(
        question_paper,
        question_paper_path,
    )

    question_paper_text = (
        question_paper_path
        .read_text(encoding="utf-8")
        .strip()
    )

    if not question_paper_text:
        raise HTTPException(
            status_code=400,
            detail="Question paper TXT file is empty.",
        )

    await save_upload(
        marking_scheme,
        marking_scheme_path,
    )

    await save_upload(
        answer_copy,
        answer_copy_path,
    )


    # --------------------------------------------------------
    # Validate marking scheme
    # --------------------------------------------------------

    scheme = load_marking_scheme(
        marking_scheme_path
    )


    # --------------------------------------------------------
    # Student number
    # --------------------------------------------------------

    student_number = (
        Path(
            answer_copy.filename
            or "UNKNOWN"
        ).stem
    )


    # --------------------------------------------------------
    # Current live engine
    # --------------------------------------------------------

    processor = (
        InMemoryPageProcessor()
    )

    extractor = (
        InMemoryAnswerExtractionService()
    )

    grader = (
        GradingModel()
    )

    renderer = (
        CheckedCopyRenderer()
    )


    actual_page_count = (
        processor.get_page_count(
            str(answer_copy_path)
        )
    )

    if actual_page_count <= 0:

        raise HTTPException(
            status_code=400,
            detail="Answer copy contains no readable pages.",
        )


    page_limit = min(
        pages,
        actual_page_count,
    )


    # --------------------------------------------------------
    # Workflow callbacks
    # --------------------------------------------------------

    def page_count_fn(
        pdf_path: str,
    ) -> int:

        return min(
            processor.get_page_count(
                pdf_path
            ),
            page_limit,
        )


    def render_page_fn(
        pdf_path: str,
        page_no: int,
    ):

        return processor.render_page(
            pdf_path,
            page_no,
        )


    def extract_page_fn(
        image,
        page_no: int,
        expected_question_numbers: list[int],
        enable_recovery: bool,
        feedback: str | None,
    ):

        question_types = {
            int(question.question_no): (
                str(
                    getattr(
                        question,
                        "question_type",
                        "subjective",
                    )
                )
                .strip()
                .lower()
            )
            for question in scheme.questions
        }

        return extractor.extract_page_in_memory(
            image=image,
            page_no=page_no,
            expected_question_numbers=(
                expected_question_numbers
            ),
            enable_recovery=False,
            feedback=None,
            extraction_mode="general",
            question_types=question_types,
        )


    workflow = (
        InMemoryGradingWorkflow(
            page_count_fn=page_count_fn,
            render_page_fn=render_page_fn,
            extract_page_fn=extract_page_fn,
            aggregation_service=(
                SubmissionAggregationService()
            ),
            grading_model=grader,
            checked_copy_renderer=renderer,
            max_pages=page_limit,
        )
    )


    # --------------------------------------------------------
    # Run actual grading pipeline
    # --------------------------------------------------------

    try:

        result, submission, checked_copy_path = (
            workflow.run(
                pdf_path=str(
                    answer_copy_path
                ),
                marking_scheme=scheme,
                job_id=job_id,
                student_number=student_number,
                assessment_id=scheme.assessment_id,
                expected_question_numbers=[
                    int(question.question_no)
                    for question in scheme.questions
                ],
                enable_recovery=False,
                metadata={
                    "api_test": True,
                    "question_paper_uploaded": True,
                    "marking_scheme_uploaded": True,
                    "answer_copy_uploaded": True,
                    "verification": "disabled",
                    "grading_review": "disabled",
                    "page_limit": page_limit,
                },
            )
        )

    except Exception as exc:
        error_text = str(exc)

        if "requires review before grading" in error_text.lower():
            return {
                "status": "review_required",
                "review_required": True,
                "grading_started": False,
                "message": (
                    "Extraction is incomplete. "
                    "Teacher review is required before grading."
                ),
                "detail": (
                    "Please review or retry extraction "
                    "before grading."
                ),
            }

        raise HTTPException(
            status_code=500,
            detail=(
                f"Grading pipeline failed: {exc}"
            ),
        ) from exc


    # --------------------------------------------------------
    # Copy checked PDF into API job output
    # --------------------------------------------------------

    source_checked_copy = Path(
        checked_copy_path
    )

    if not source_checked_copy.is_absolute():

        source_checked_copy = (
            BASE_DIR
            / source_checked_copy
        )


    source_checked_copy = (
        source_checked_copy.resolve()
    )


    if not source_checked_copy.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "Grading completed but the checked-copy "
                "PDF could not be found."
            ),
        )


    final_checked_copy = (
        output_dir
        / f"{job_id}_checked.pdf"
    )


    shutil.copy2(
        source_checked_copy,
        final_checked_copy,
    )


    # --------------------------------------------------------
    # Question-wise result
    # --------------------------------------------------------

    question_wise = []

    for page in result.pages:

        for question in page["questions"]:

            question_wise.append(
                {
                    "question_no": question[
                        "question_no"
                    ],
                    "marks_awarded": question[
                        "marks_awarded"
                    ],
                    "max_marks": question[
                        "max_marks"
                    ],
                    "reason": question[
                        "reason"
                    ],
                    "confidence": question[
                        "confidence"
                    ],
                    "page": page[
                        "page_number"
                    ],
                }
            )


    return {
        "status": "success",
        "job_id": job_id,
        "student_number": result.student_id,
        "assessment_id": result.assessment_id,

        "input": {
            "question_paper": (
                question_paper.filename
            ),
            "marking_scheme": (
                marking_scheme.filename
            ),
            "answer_copy": (
                answer_copy.filename
            ),
        },

        "processing": {
            "total_pdf_pages": actual_page_count,
            "pages_processed": result.pages_processed,
            "questions_extracted": len(
                submission.answers
            ),
        },

        "grading": {
            "total_marks": result.total_marks,
            "obtained_marks": result.obtained_marks,
            "percentage": result.percentage,
        },

        "question_wise": question_wise,

        "checked_copy_url": (
            f"/api/jobs/"
            f"{job_id}"
            f"/checked-copy"
        ),
    }


# ============================================================
# CHECKED COPY DOWNLOAD
# ============================================================

@app.get(
    "/api/jobs/{job_id}/checked-copy"
)
def get_checked_copy(
    job_id: str,
):

    if not job_id.startswith("API-"):

        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )


    path = (
        JOB_ROOT
        / job_id
        / "output"
        / f"{job_id}_checked.pdf"
    )


    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="Checked copy not found.",
        )


    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=(
            f"{job_id}_checked.pdf"
        ),
    )
