from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.agent.answer_sheet_evaluation_agent import (
    AnswerSheetEvaluationAgent,
)
from app.core.config import settings


router = APIRouter(
    prefix="/api/v1",
    tags=["copy-checker"],
)


# ============================================================
# AGENT
# ============================================================

agent = AnswerSheetEvaluationAgent()


# ============================================================
# CREATE EVALUATION
# ============================================================


@router.post("/evaluations")
async def create_evaluation(
    file: UploadFile = File(...),
    assessment_id: str | None = Form(default=None),
):
    """
    Create a handwritten answer-sheet evaluation job.

    Student number is taken from the ORIGINAL uploaded
    filename.

    Example:

        STU047.pdf
            ↓
        student_number = STU047

    The uploaded file is then stored internally with a
    generated safe filename, but student identity continues
    to use the original student number.
    """

    # --------------------------------------------------------
    # Validate original filename
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File name is required.",
        )

    original_path = Path(file.filename)

    # Use only the filename itself, not any client-supplied
    # directory information.
    original_name = original_path.name

    extension = Path(original_name).suffix.lower()

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    if extension != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported in this version.",
        )

    # --------------------------------------------------------
    # Extract student number from original filename
    # --------------------------------------------------------

    student_number = Path(original_name).stem.strip()

    if not student_number:
        raise HTTPException(
            status_code=400,
            detail=(
                "Student number must be present in the "
                "answer-sheet filename."
            ),
        )

    # --------------------------------------------------------
    # Upload directory
    # --------------------------------------------------------

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Safe internal storage name
    # --------------------------------------------------------
    #
    # We do NOT use this generated name as student identity.
    # Student identity remains the original student number.
    #

    safe_name = (
        f"{uuid4().hex}_{original_name}"
    )

    pdf_path = upload_dir / safe_name

    # --------------------------------------------------------
    # Save uploaded PDF
    # --------------------------------------------------------

    try:
        content = await file.read()

        pdf_path.write_bytes(content)

        # ----------------------------------------------------
        # Create evaluation job
        # ----------------------------------------------------

        result = agent.create_job(
            pdf_path=str(pdf_path),
            student_number=student_number,
            assessment_id=assessment_id,
        )

        return result.model_dump()

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to process the PDF.",
        ) from exc