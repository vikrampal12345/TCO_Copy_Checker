from app.schemas.extraction import ExtractedAnswer, PageExtractionResult
from app.schemas.submission_verification import (
    SubmissionVerificationResult,
)
from app.services.submission_aggregation_service import (
    SubmissionAggregationService,
)
from app.services.submission_verification_service import (
    SubmissionVerificationService,
)
from app.services.submission_verification_workflow import (
    SubmissionVerificationWorkflow,
)


def make_page(answer: str) -> PageExtractionResult:
    return PageExtractionResult(
        page_no=1,
        questions=[
            ExtractedAnswer(
                question_no=1,
                answer=answer,
                confidence=0.95,
                source_tile="tile_01",
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


def build_workflow(verifier_fn):
    aggregation_service = SubmissionAggregationService()

    verification_service = SubmissionVerificationService(
        verifier_fn
    )

    return SubmissionVerificationWorkflow(
        aggregation_service=aggregation_service,
        verification_service=verification_service,
        max_attempts=3,
    )


def test_passes_on_first_attempt():
    extractor_calls = []

    def extract_fn(attempt, feedback):
        extractor_calls.append((attempt, feedback))
        return [make_page("Correct answer")]

    def verifier_fn(submission, attempt):
        return SubmissionVerificationResult(
            passed=True,
            confidence=0.98,
            attempt=attempt,
        )

    workflow = build_workflow(verifier_fn)

    result = workflow.run(
        extract_fn=extract_fn,
        job_id="JOB001",
        student_number="STU047",
        expected_question_numbers=[1],
    )

    assert result.passed is True
    assert result.attempts == 1
    assert result.review_required is False
    assert len(extractor_calls) == 1
    assert extractor_calls[0] == (1, None)


def test_fail_then_pass_on_second_attempt():
    extractor_calls = []

    def extract_fn(attempt, feedback):
        extractor_calls.append((attempt, feedback))

        if attempt == 1:
            return [make_page("Bad extraction")]

        return [make_page("Corrected extraction")]

    def verifier_fn(submission, attempt):
        if attempt == 1:
            return SubmissionVerificationResult(
                passed=False,
                confidence=0.40,
                feedback="Re-extract Q1 carefully.",
                attempt=attempt,
            )

        return SubmissionVerificationResult(
            passed=True,
            confidence=0.97,
            attempt=attempt,
        )

    workflow = build_workflow(verifier_fn)

    result = workflow.run(
        extract_fn=extract_fn,
        job_id="JOB002",
        student_number="STU048",
        expected_question_numbers=[1],
    )

    assert result.passed is True
    assert result.attempts == 2
    assert result.review_required is False

    assert len(extractor_calls) == 2
    assert extractor_calls[0] == (1, None)
    assert extractor_calls[1][0] == 2
    assert extractor_calls[1][1] == "Re-extract Q1 carefully."


def test_three_failures_require_review():
    extractor_calls = []

    def extract_fn(attempt, feedback):
        extractor_calls.append((attempt, feedback))
        return [make_page(f"Attempt {attempt} extraction")]

    def verifier_fn(submission, attempt):
        return SubmissionVerificationResult(
            passed=False,
            confidence=0.30,
            feedback=f"Retry required after attempt {attempt}.",
            attempt=attempt,
        )

    workflow = build_workflow(verifier_fn)

    result = workflow.run(
        extract_fn=extract_fn,
        job_id="JOB003",
        student_number="STU049",
        expected_question_numbers=[1],
    )

    assert result.passed is False
    assert result.attempts == 3
    assert result.review_required is True

    assert len(extractor_calls) == 3
    assert extractor_calls[0] == (1, None)
    assert extractor_calls[1][0] == 2
    assert extractor_calls[1][1] == "Retry required after attempt 1."
    assert extractor_calls[2][0] == 3
    assert extractor_calls[2][1] == "Retry required after attempt 2."


def test_custom_max_attempts():
    calls = []

    def extract_fn(attempt, feedback):
        calls.append(attempt)
        return [make_page("Uncertain")]

    def verifier_fn(submission, attempt):
        return SubmissionVerificationResult(
            passed=False,
            confidence=0.20,
            feedback="Still uncertain.",
            attempt=attempt,
        )

    workflow = SubmissionVerificationWorkflow(
        aggregation_service=SubmissionAggregationService(),
        verification_service=SubmissionVerificationService(
            verifier_fn
        ),
        max_attempts=2,
    )

    result = workflow.run(
        extract_fn=extract_fn,
        student_number="STU050",
    )

    assert result.passed is False
    assert result.attempts == 2
    assert result.review_required is True
    assert calls == [1, 2]
