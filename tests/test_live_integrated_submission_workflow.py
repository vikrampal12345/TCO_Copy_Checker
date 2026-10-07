from PIL import Image

from app.schemas.submission import AggregatedSubmission
from app.services.in_memory_answer_extraction_service import (
    InMemoryAnswerExtractionService,
)
from app.services.in_memory_page_processor import (
    InMemoryPageProcessor,
)
from app.services.in_memory_submission_verification_workflow import (
    InMemorySubmissionVerificationWorkflow,
)
from app.services.submission_aggregation_service import (
    SubmissionAggregationService,
)
from app.services.submission_verifier_model import (
    SubmissionVerifierModel,
)


PDF_PATH = r"data\test_copies\Student_23.pdf"


def test_live_page_extraction_aggregation_verification():
    print("\n" + "=" * 100)
    print("TCO COPY CHECKER")
    print("LIVE PAGE EXTRACTION -> AGGREGATION -> VERIFICATION")
    print("=" * 100)
    print(f"PDF: {PDF_PATH}")

    processor = InMemoryPageProcessor()
    extractor = InMemoryAnswerExtractionService()
    verifier = SubmissionVerifierModel()

    page_count = processor.get_page_count(PDF_PATH)

    print(f"PDF total pages: {page_count}")
    print(f"Extraction model: {extractor.vision_service.get_model_name()}")
    print(f"Verifier model: {verifier.model}")
    print("Live test scope: Page 1 only")
    print("Maximum verification attempts: 2")
    print()

    def page_count_fn(pdf_path: str) -> int:
        return 1

    def render_page_fn(pdf_path: str, page_no: int) -> Image.Image:
        return processor.render_page(
            pdf_path,
            page_no,
        )

    def extract_page_fn(
        image: Image.Image,
        page_no: int,
        expected_question_numbers: list[int],
        enable_recovery: bool,
        feedback: str | None,
    ):
        print("=" * 100)
        print(f"EXTRACTION PAGE {page_no}")
        print("=" * 100)

        if feedback:
            print(f"Verifier feedback supplied to extractor: {feedback}")
        else:
            print("No verifier feedback. Initial extraction.")

        return extractor.extract_page_in_memory(
            image=image,
            page_no=page_no,
            expected_question_numbers=expected_question_numbers,
            enable_recovery=enable_recovery,
            feedback=feedback,
        )

    workflow = InMemorySubmissionVerificationWorkflow(
        page_count_fn=page_count_fn,
        render_page_fn=render_page_fn,
        extract_page_fn=extract_page_fn,
        aggregation_service=SubmissionAggregationService(),
        verifier_model=verifier,
        max_attempts=2,
    )

    result = workflow.run(
        pdf_path=PDF_PATH,
        expected_question_numbers=[],
        job_id="LIVE-INTEGRATED-001",
        student_number="STU023",
        enable_recovery=True,
        metadata={
            "test": True,
            "scope": "page_1_live",
        },
    )

    print("\n" + "=" * 100)
    print("FINAL INTEGRATED RESULT")
    print("=" * 100)
    print(result.model_dump_json(indent=2))

    assert 1 <= result.attempts <= 2
    assert isinstance(result.passed, bool)
    assert len(result.submission["answers"]) > 0
    assert 0 <= result.verification.confidence <= 1

    submission = AggregatedSubmission.model_validate(
        result.submission
    )

    assert submission.job_id == "LIVE-INTEGRATED-001"
    assert submission.student_number == "STU023"
    assert submission.total_pages == 1
    assert submission.processed_pages == 1

    print("\n" + "=" * 100)
    print("LIVE INTEGRATED TEST COMPLETED")
    print("=" * 100)



