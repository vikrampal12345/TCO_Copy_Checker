from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from app.schemas.extraction import PageExtractionResult
from app.schemas.submission import (
    AggregatedAnswer,
    AggregatedSubmission,
    AnswerObservation,
)


class SubmissionAggregationService:
    """
    Combines page-level extraction results into one student submission.

    Responsibilities:
    - Collect observations from every processed page.
    - Group observations by question number.
    - Detect missing questions when expected question numbers are known.
    - Detect repeated question numbers.
    - Detect conflicting extracted answers.
    - Select the best available answer.
    - Mark the submission for review when extraction is ambiguous.
    """

    def __init__(self, conflict_similarity_threshold: float = 0.85) -> None:
        if not 0 <= conflict_similarity_threshold <= 1:
            raise ValueError(
                "conflict_similarity_threshold must be between 0 and 1"
            )

        self.conflict_similarity_threshold = conflict_similarity_threshold

    @staticmethod
    def _normalize_text(text: str | None) -> str:
        if text is None:
            return ""

        value = " ".join(str(text).strip().split())

        punctuation = ".,;:!?()[]{}\"'`"
        value = value.translate(str.maketrans("", "", punctuation))

        return value.lower().strip()

    @classmethod
    def _is_unreadable(cls, answer: str | None, status: str | None) -> bool:
        normalized = cls._normalize_text(answer)

        if not normalized:
            return True

        unresolved_tokens = {
            "[unclear]",
            "unclear",
            "[unreadable]",
            "unreadable",
            "?",
            "????",
        }

        if normalized in unresolved_tokens:
            return True

        if status and status.upper() in {
            "REVIEW_REQUIRED",
            "UNRESOLVED",
            "FAILED",
        }:
            return True

        return False

    @staticmethod
    def _text_similarity(left: str, right: str) -> float:
        """
        Lightweight token-based similarity.

        This is intentionally deterministic and does not use another model.
        """
        left_tokens = set(left.split())
        right_tokens = set(right.split())

        if not left_tokens and not right_tokens:
            return 1.0

        if not left_tokens or not right_tokens:
            return 0.0

        intersection = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)

        return intersection / union

    def _answers_conflict(
        self,
        observations: list[AnswerObservation],
    ) -> bool:
        readable = [
            self._normalize_text(obs.answer)
            for obs in observations
            if not self._is_unreadable(obs.answer, obs.status)
        ]

        readable = [answer for answer in readable if answer]

        if len(readable) <= 1:
            return False

        reference = readable[0]

        for answer in readable[1:]:
            if answer == reference:
                continue

            similarity = self._text_similarity(reference, answer)

            if similarity < self.conflict_similarity_threshold:
                return True

        return False

    @staticmethod
    def _best_observation(
        observations: list[AnswerObservation],
    ) -> AnswerObservation | None:
        readable = [
            obs
            for obs in observations
            if not SubmissionAggregationService._is_unreadable(
                obs.answer,
                obs.status,
            )
        ]

        if not readable:
            return None

        return max(
            readable,
            key=lambda obs: (
                obs.confidence,
                len(obs.answer.strip()),
            ),
        )

    def aggregate(
        self,
        page_results: Iterable[PageExtractionResult],
        *,
        job_id: str | None = None,
        student_number: str | None = None,
        expected_question_numbers: list[int] | None = None,
        metadata: dict | None = None,
    ) -> AggregatedSubmission:
        page_results = list(page_results)

        observations_by_question: dict[int, list[AnswerObservation]] = (
            defaultdict(list)
        )

        warnings: list[str] = []

        total_pages = len(page_results)
        processed_pages = 0
        unresolved_items = 0
        review_required = False

        for page_result in page_results:
            if page_result.page_no >= 1:
                processed_pages += 1

            if page_result.review_required:
                review_required = True

            unresolved_items += len(page_result.unresolved_items)

            warnings.extend(page_result.warnings)

            for item in page_result.questions:
                if item.question_no is None:
                    unresolved_items += 1
                    review_required = True
                    continue

                observation = AnswerObservation(
                    page_no=page_result.page_no,
                    question_no=item.question_no,
                    answer=item.answer,
                    confidence=item.confidence,
                    source_tile=item.source_tile,
                    status=item.status,
                    warning=item.warning,
                )

                observations_by_question[item.question_no].append(
                    observation
                )

        expected = sorted(
            set(expected_question_numbers or [])
        )

        missing_questions: list[int] = []

        if expected:
            actual_questions = set(observations_by_question.keys())
            missing_questions = sorted(actual_questions ^ actual_questions & set())
            missing_questions = sorted(
                set(expected) - actual_questions
            )

            if missing_questions:
                review_required = True
                warnings.append(
                    "Missing expected questions: "
                    + ", ".join(map(str, missing_questions))
                )

        aggregated_answers: list[AggregatedAnswer] = []
        repeated_questions: list[int] = []
        conflicting_questions: list[int] = []

        for question_no in sorted(observations_by_question):
            observations = observations_by_question[question_no]

            if len(observations) > 1:
                repeated_questions.append(question_no)

            conflict = self._answers_conflict(observations)

            if conflict:
                conflicting_questions.append(question_no)
                review_required = True

            best = self._best_observation(observations)

            question_review_required = conflict

            if best is None:
                question_review_required = True
                review_required = True

                warning = (
                    "No readable observation was available for "
                    f"question {question_no}."
                )
                warnings.append(warning)

                aggregated_answers.append(
                    AggregatedAnswer(
                        question_no=question_no,
                        answer=None,
                        confidence=None,
                        observations=observations,
                        status="REVIEW_REQUIRED",
                        review_required=True,
                        warning=warning,
                    )
                )
                continue

            question_warning: str | None = None

            if conflict:
                question_warning = (
                    "Multiple extracted observations for this question "
                    "are inconsistent."
                )
            elif best.warning:
                question_warning = best.warning

            aggregated_answers.append(
                AggregatedAnswer(
                    question_no=question_no,
                    answer=best.answer,
                    confidence=best.confidence,
                    observations=observations,
                    status=(
                        "REVIEW_REQUIRED"
                        if question_review_required
                        else "EXTRACTED"
                    ),
                    review_required=question_review_required,
                    warning=question_warning,
                )
            )

        if repeated_questions:
            warnings.append(
                "Repeated question numbers detected: "
                + ", ".join(map(str, repeated_questions))
            )

        if conflicting_questions:
            warnings.append(
                "Conflicting observations detected for questions: "
                + ", ".join(map(str, conflicting_questions))
            )

        if unresolved_items > 0:
            review_required = True
            warnings.append(
                f"{unresolved_items} unresolved extraction item(s) detected."
            )

        return AggregatedSubmission(
            job_id=job_id,
            student_number=student_number,
            total_pages=total_pages,
            processed_pages=processed_pages,
            answers=aggregated_answers,
            expected_questions=expected,
            missing_questions=missing_questions,
            repeated_questions=repeated_questions,
            conflicting_questions=conflicting_questions,
            unresolved_items=unresolved_items,
            review_required=review_required,
            warnings=warnings,
            metadata=metadata or {},
        )
