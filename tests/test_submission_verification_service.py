from app.schemas.submission import AggregatedAnswer, AggregatedSubmission
from app.schemas.submission_verification import (
    SubmissionVerificationIssue,
    SubmissionVerificationResult,
)
from app.services.submission_verification_service import (
    SubmissionVerificationService,
)


def make_submission():
    return AggregatedSubmission(
        job_id="JOB001",
        student_number="STU047",
        total_pages=2,
        processed_pages=2,
        answers=[
            AggregatedAnswer(
                question_no=1,
                answer="Photosynthesis is the process by which plants make food.",
                confidence=0.92,
                status="EXTRACTED",
            )
        ],
        expected_questions=[1],
        missing_questions=[],
        repeated_questions=[],
        conflicting_questions=[],
        unresolved_items=0,
        review_required=False,
    )


def test_verification_pass():
    calls = []

    def fake_verifier(submission, attempt):
        calls.append(attempt)

        return SubmissionVerificationResult(
            passed=True,
            confidence=0.96,
            attempt=attempt,
        )

    service = SubmissionVerificationService(fake_verifier)

    result = service.verify(
        make_submission(),
        attempt=1,
    )

    assert result.passed is True
    assert result.confidence == 0.96
    assert result.attempt == 1
    assert calls == [1]


def test_verification_fail_builds_feedback():
    def fake_verifier(submission, attempt):
        return SubmissionVerificationResult(
            passed=False,
            confidence=0.55,
            issues=[
                SubmissionVerificationIssue(
                    question_no=1,
                    issue_type="possible_extraction_error",
                    severity="high",
                    message="Answer appears incomplete.",
                )
            ],
            attempt=attempt,
        )

    service = SubmissionVerificationService(fake_verifier)

    result = service.verify(
        make_submission(),
        attempt=1,
    )

    result = service.normalize_result(
        result,
        attempt=1,
    )

    assert result.passed is False
    assert result.attempt == 1
    assert result.feedback is not None
    assert "Q1" in result.feedback
    assert "incomplete" in result.feedback


def test_custom_feedback_is_preserved():
    def fake_verifier(submission, attempt):
        return SubmissionVerificationResult(
            passed=False,
            confidence=0.40,
            feedback="Re-check Q3 and Q5 extraction.",
            attempt=attempt,
        )

    service = SubmissionVerificationService(fake_verifier)

    result = service.verify(
        make_submission(),
        attempt=2,
    )

    result = service.normalize_result(
        result,
        attempt=2,
    )

    assert result.attempt == 2
    assert result.feedback == "Re-check Q3 and Q5 extraction."


def test_submission_summary():
    submission = make_submission()

    summary = SubmissionVerificationService.submission_summary(
        submission
    )

    assert summary["student_number"] == "STU047"
    assert summary["job_id"] == "JOB001"
    assert summary["total_pages"] == 2
    assert summary["processed_pages"] == 2
    assert summary["answer_count"] == 1
    assert summary["review_required"] is False


def test_high_severity_issue_forces_failure():
    result = SubmissionVerificationResult(
        passed=True,
        confidence=0.95,
        issues=[
            SubmissionVerificationIssue(
                question_no=3,
                issue_type="extraction_error",
                severity="high",
                message="The extracted answer does not match the handwriting.",
            )
        ],
        feedback="Re-extract Q3.",
        attempt=1,
    )

    normalized = SubmissionVerificationService.normalize_result(
        result,
        attempt=1,
    )

    assert normalized.passed is False
    assert normalized.attempt == 1
    assert normalized.feedback == "Re-extract Q3."
