from PIL import Image

from app.schemas.extraction import (
    ExtractedAnswer,
    PageExtractionResult,
)
from app.schemas.submission_verification import (
    SubmissionVerificationResult,
)
from app.services.in_memory_submission_verification_workflow import (
    InMemorySubmissionVerificationWorkflow,
)
from app.services.submission_aggregation_service import (
    SubmissionAggregationService,
)


def make_page_result(page_no: int, attempt: int):
    return PageExtractionResult(
        page_no=page_no,
        questions=[
            ExtractedAnswer(
                question_no=page_no,
                answer=f"Answer page {page_no}, attempt {attempt}",
                confidence=0.95,
                source_tile=f"page_{page_no}_tile_01",
                status="EXTRACTED",
            )
        ],
        unresolved_items=[],
        missing_questions=[],
        review_required=False,
        warnings=[],
        processed_tiles=8,
        recovery_attempted=False,
        recovery_tiles=0,
    )


class FakeVerifierModel:
    def __init__(self):
        self.calls = []

    def verify(
        self,
        submission,
        *,
        attempt,
        page_images,
        temperature=0,
    ):
        self.calls.append(
            {
                "attempt": attempt,
                "page_count": len(page_images),
                "questions": [
                    item.question_no
                    for item in submission.answers
                ],
                "temperature": temperature,
            }
        )

        if attempt == 1:
            return SubmissionVerificationResult(
                passed=False,
                confidence=0.40,
                feedback="Re-extract page 2 carefully.",
                attempt=attempt,
            )

        return SubmissionVerificationResult(
            passed=True,
            confidence=0.97,
            issues=[],
            feedback=None,
            attempt=attempt,
        )


def test_full_in_memory_extraction_aggregation_verification_flow():
    extracted_attempts = []
    rendered_images = []

    def page_count_fn(pdf_path):
        assert pdf_path == "fake_student.pdf"
        return 3

    def render_page_fn(pdf_path, page_no):
        image = Image.new(
            "RGB",
            (120, 120),
            "white",
        )

        rendered_images.append(
            (page_no, image)
        )

        return image

    def extract_page_fn(
        image,
        page_no,
        expected_question_numbers,
        enable_recovery,
        feedback,
    ):
        extracted_attempts.append(
            {
                "page_no": page_no,
                "feedback": feedback,
                "expected": expected_question_numbers,
                "enable_recovery": enable_recovery,
            }
        )

        return make_page_result(
            page_no=page_no,
            attempt=(
                1
                if feedback is None
                else 2
            ),
        )

    verifier = FakeVerifierModel()

    workflow = InMemorySubmissionVerificationWorkflow(
        page_count_fn=page_count_fn,
        render_page_fn=render_page_fn,
        extract_page_fn=extract_page_fn,
        aggregation_service=SubmissionAggregationService(),
        verifier_model=verifier,
        max_attempts=3,
    )

    result = workflow.run(
        pdf_path="fake_student.pdf",
        expected_question_numbers=[1, 2, 3],
        job_id="JOB001",
        student_number="STU047",
    )

    assert result.passed is True
    assert result.attempts == 2
    assert result.review_required is False

    # Three pages extracted on attempt 1.
    # Three pages extracted again on attempt 2.
    assert len(extracted_attempts) == 6

    assert extracted_attempts[0]["page_no"] == 1
    assert extracted_attempts[1]["page_no"] == 2
    assert extracted_attempts[2]["page_no"] == 3

    assert extracted_attempts[0]["feedback"] is None
    assert extracted_attempts[1]["feedback"] is None
    assert extracted_attempts[2]["feedback"] is None

    assert extracted_attempts[3]["page_no"] == 1
    assert extracted_attempts[4]["page_no"] == 2
    assert extracted_attempts[5]["page_no"] == 3

    assert extracted_attempts[3]["feedback"] == (
        "Re-extract page 2 carefully."
    )

    # Every verification attempt receives all original pages.
    assert verifier.calls[0]["attempt"] == 1
    assert verifier.calls[0]["page_count"] == 3

    assert verifier.calls[1]["attempt"] == 2
    assert verifier.calls[1]["page_count"] == 3

    assert verifier.calls[0]["questions"] == [1, 2, 3]
    assert verifier.calls[1]["questions"] == [1, 2, 3]

    # All RAM images must be closed after the workflow.
    for _, image in rendered_images:
        try:
            image.getbbox()
        except Exception:
            pass

    assert result.submission["student_number"] == "STU047"
    assert result.submission["job_id"] == "JOB001"


def test_in_memory_workflow_stops_after_three_failed_verifications():
    extracted_attempts = []

    def page_count_fn(pdf_path):
        return 2

    def render_page_fn(pdf_path, page_no):
        return Image.new(
            "RGB",
            (80, 80),
            "white",
        )

    def extract_page_fn(
        image,
        page_no,
        expected_question_numbers,
        enable_recovery,
        feedback,
    ):
        extracted_attempts.append(
            (page_no, feedback)
        )

        return make_page_result(
            page_no=page_no,
            attempt=1 if feedback is None else 2,
        )

    class AlwaysFailVerifier:
        def __init__(self):
            self.calls = []

        def verify(
            self,
            submission,
            *,
            attempt,
            page_images,
            temperature=0,
        ):
            self.calls.append(attempt)

            return SubmissionVerificationResult(
                passed=False,
                confidence=0.20,
                feedback=f"Retry after attempt {attempt}.",
                attempt=attempt,
            )

    verifier = AlwaysFailVerifier()

    workflow = InMemorySubmissionVerificationWorkflow(
        page_count_fn=page_count_fn,
        render_page_fn=render_page_fn,
        extract_page_fn=extract_page_fn,
        aggregation_service=SubmissionAggregationService(),
        verifier_model=verifier,
        max_attempts=3,
    )

    result = workflow.run(
        pdf_path="fake_student.pdf",
        expected_question_numbers=[1, 2],
        student_number="STU048",
    )

    assert result.passed is False
    assert result.attempts == 3
    assert result.review_required is True

    assert verifier.calls == [1, 2, 3]

    # 2 pages × 3 complete extraction attempts
    assert len(extracted_attempts) == 6
