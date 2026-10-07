from app.schemas.submission import (
    AggregatedAnswer,
    AggregatedSubmission,
)
from app.services.in_memory_page_processor import (
    InMemoryPageProcessor,
)
from app.services.submission_verifier_model import (
    SubmissionVerifierModel,
)
from app.services.submission_verification_service import (
    SubmissionVerificationService,
)


PDF_PATH = r"data\test_copies\Student_23.pdf"


def test_live_submission_verifier_page_1():
    print("\n" + "=" * 90)
    print("TCO COPY CHECKER")
    print("LIVE COMBINED SUBMISSION VERIFIER TEST")
    print("=" * 90)

    processor = InMemoryPageProcessor()

    print(f"PDF: {PDF_PATH}")

    page_count = processor.get_page_count(PDF_PATH)

    print(f"Total pages: {page_count}")

    image = processor.render_page(
        PDF_PATH,
        page_number=1,
    )

    print(
        f"Page 1 image in RAM: "
        f"{image.width} x {image.height}"
    )

    submission = AggregatedSubmission(
        job_id="LIVE-VERIFY-001",
        student_number="STU023",
        total_pages=1,
        processed_pages=1,
        answers=[
            AggregatedAnswer(
                question_no=1,
                answer="A",
                confidence=0.95,
                status="EXTRACTED",
            ),
            AggregatedAnswer(
                question_no=2,
                answer="C",
                confidence=0.90,
                status="EXTRACTED",
            ),
            AggregatedAnswer(
                question_no=3,
                answer="A",
                confidence=0.90,
                status="EXTRACTED",
            ),
            AggregatedAnswer(
                question_no=4,
                answer="C",
                confidence=0.90,
                status="EXTRACTED",
            ),
        ],
        expected_questions=[1, 2, 3, 4],
        missing_questions=[],
        repeated_questions=[],
        conflicting_questions=[],
        unresolved_items=0,
        review_required=False,
    )

    verifier_model = SubmissionVerifierModel()

    print(f"Verifier model: {verifier_model.model}")
    print("Sending original page + aggregated extraction...")
    print("This may take some time with the local model.\n")

    verification_service = SubmissionVerificationService(
        lambda current_submission, attempt:
            verifier_model.verify(
                current_submission,
                attempt=attempt,
                page_images=[
                    (1, image),
                ],
                temperature=0,
            )
    )

    try:
        raw_result = verification_service.verify(
            submission,
            attempt=1,
        )

        result = verification_service.normalize_result(
            raw_result,
            attempt=1,
        )
    finally:
        image.close()

    print("=" * 90)
    print("VERIFIER RESULT")
    print("=" * 90)

    print(result.model_dump_json(indent=2))

    print("\n" + "=" * 90)

    assert result.attempt == 1
    assert 0 <= result.confidence <= 1
    assert isinstance(result.passed, bool)

    print("LIVE SUBMISSION VERIFIER TEST COMPLETED")
    print("=" * 90)
