from __future__ import annotations

from PIL import Image

from app.schemas.extraction import (
    ExtractedAnswer,
    PageExtractionResult,
)
from app.schemas.extraction_verification import (
    ExtractionVerificationIssue,
    ExtractionVerificationResult,
)
from app.services.extraction_verification_service import (
    ExtractionVerificationService,
)
from app.services.extraction_verification_workflow import (
    ExtractionVerificationWorkflow,
)


def build_extraction(
    page_no: int,
    answer: str,
) -> PageExtractionResult:
    return PageExtractionResult(
        page_no=page_no,
        questions=[
            ExtractedAnswer(
                question_no=1,
                answer=answer,
                confidence=0.90,
                status="EXTRACTED",
            )
        ],
    )


def test_verification_passes_on_first_attempt() -> None:
    image = Image.new(
        "RGB",
        (100, 100),
        "white",
    )

    calls: list[int] = []

    def fake_extractor(
        image: Image.Image,
        page_no: int,
        feedback: str | None,
    ) -> PageExtractionResult:
        calls.append(1)

        return build_extraction(
            page_no=page_no,
            answer="A",
        )

    def fake_verifier(
        image: Image.Image,
        extraction: PageExtractionResult,
        attempt: int,
    ) -> ExtractionVerificationResult:
        return ExtractionVerificationResult(
            page_no=extraction.page_no,
            passed=True,
            confidence=0.98,
            attempt=attempt,
        )

    verifier = ExtractionVerificationService(
        verify_fn=fake_verifier,
    )

    workflow = ExtractionVerificationWorkflow(
        verification_service=verifier,
        max_attempts=3,
    )

    result = workflow.run(
        image=image,
        page_no=1,
        extract_fn=fake_extractor,
    )

    assert result.passed is True
    assert result.attempts == 1
    assert result.review_required is False
    assert result.extraction["questions"][0]["answer"] == "A"
    assert len(calls) == 1


def test_failed_verification_retries_with_feedback() -> None:
    image = Image.new(
        "RGB",
        (100, 100),
        "white",
    )

    extractor_feedback: list[str | None] = []

    def fake_extractor(
        image: Image.Image,
        page_no: int,
        feedback: str | None,
    ) -> PageExtractionResult:
        extractor_feedback.append(feedback)

        if feedback is None:
            return build_extraction(
                page_no=page_no,
                answer="B",
            )

        return build_extraction(
            page_no=page_no,
            answer="A",
        )

    def fake_verifier(
        image: Image.Image,
        extraction: PageExtractionResult,
        attempt: int,
    ) -> ExtractionVerificationResult:

        if attempt == 1:
            return ExtractionVerificationResult(
                page_no=extraction.page_no,
                passed=False,
                confidence=0.92,
                issues=[
                    ExtractionVerificationIssue(
                        question_no=1,
                        issue_type="wrong_answer",
                        severity="high",
                        message=(
                            "The extracted answer does not "
                            "match the handwriting on the page."
                        ),
                    )
                ],
                feedback=(
                    "Re-check Question 1 carefully."
                ),
                attempt=attempt,
            )

        return ExtractionVerificationResult(
            page_no=extraction.page_no,
            passed=True,
            confidence=0.97,
            attempt=attempt,
        )

    verifier = ExtractionVerificationService(
        verify_fn=fake_verifier,
    )

    workflow = ExtractionVerificationWorkflow(
        verification_service=verifier,
        max_attempts=3,
    )

    result = workflow.run(
        image=image,
        page_no=1,
        extract_fn=fake_extractor,
    )

    assert result.passed is True
    assert result.attempts == 2
    assert result.review_required is False

    assert len(extractor_feedback) == 2

    assert extractor_feedback[0] is None

    assert extractor_feedback[1] is not None
    assert "Question 1" in extractor_feedback[1]

    assert (
        result.extraction["questions"][0]["answer"]
        == "A"
    )


def test_failed_verification_marks_review_after_max_attempts() -> None:
    image = Image.new(
        "RGB",
        (100, 100),
        "white",
    )

    extractor_calls = 0
    verifier_calls = 0

    def fake_extractor(
        image: Image.Image,
        page_no: int,
        feedback: str | None,
    ) -> PageExtractionResult:
        nonlocal extractor_calls

        extractor_calls += 1

        return build_extraction(
            page_no=page_no,
            answer="B",
        )

    def fake_verifier(
        image: Image.Image,
        extraction: PageExtractionResult,
        attempt: int,
    ) -> ExtractionVerificationResult:
        nonlocal verifier_calls

        verifier_calls += 1

        return ExtractionVerificationResult(
            page_no=extraction.page_no,
            passed=False,
            confidence=0.60,
            issues=[
                ExtractionVerificationIssue(
                    question_no=1,
                    issue_type="unresolved",
                    severity="high",
                    message=(
                        "Question 1 remains unclear."
                    ),
                )
            ],
            feedback=(
                "Question 1 could not be verified."
            ),
            attempt=attempt,
        )

    verifier = ExtractionVerificationService(
        verify_fn=fake_verifier,
    )

    workflow = ExtractionVerificationWorkflow(
        verification_service=verifier,
        max_attempts=3,
    )

    result = workflow.run(
        image=image,
        page_no=1,
        extract_fn=fake_extractor,
    )

    assert result.passed is False
    assert result.attempts == 3
    assert result.review_required is True

    assert extractor_calls == 3
    assert verifier_calls == 3

    assert any(
        "Maximum extraction verification attempts"
        in warning
        for warning in result.warnings
    )


def test_feedback_builder() -> None:
    result = ExtractionVerificationResult(
        page_no=2,
        passed=False,
        confidence=0.50,
        issues=[
            ExtractionVerificationIssue(
                question_no=5,
                issue_type="wrong_question_number",
                severity="high",
                message=(
                    "Question number appears to be 6, "
                    "not 5."
                ),
            ),
            ExtractionVerificationIssue(
                question_no=6,
                issue_type="missing_content",
                severity="medium",
                message=(
                    "The answer appears incomplete."
                ),
            ),
        ],
        feedback=(
            "Reinspect the question boundaries."
        ),
        attempt=1,
    )

    feedback = (
        ExtractionVerificationService.build_feedback(
            result
        )
    )

    assert "Reinspect the question boundaries." in feedback
    assert "Q5:" in feedback
    assert "Q6:" in feedback
    assert "incomplete" in feedback


def test_verifier_does_not_modify_extraction() -> None:
    image = Image.new(
        "RGB",
        (100, 100),
        "white",
    )

    extraction = build_extraction(
        page_no=1,
        answer="B",
    )

    original_answer = (
        extraction.questions[0].answer
    )

    def fake_verifier(
        image: Image.Image,
        extraction: PageExtractionResult,
        attempt: int,
    ) -> ExtractionVerificationResult:
        return ExtractionVerificationResult(
            page_no=1,
            passed=False,
            confidence=0.40,
            issues=[
                ExtractionVerificationIssue(
                    question_no=1,
                    issue_type="wrong_answer",
                    severity="high",
                    message="Expected A.",
                )
            ],
            feedback="Re-extract Question 1.",
            attempt=attempt,
        )

    verifier = ExtractionVerificationService(
        verify_fn=fake_verifier,
    )

    result = verifier.verify(
        image=image,
        extraction=extraction,
        attempt=1,
    )

    assert result.passed is False

    assert (
        extraction.questions[0].answer
        == original_answer
        == "B"
    )