from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

from app.core.model_provider import (
    ModelProvider,
    get_model_provider,
)
from app.schemas.evaluation_result import EvaluationResult
from app.schemas.marking_scheme import MarkingScheme
from app.schemas.page_result import QuestionEvaluation, PageResult
from app.schemas.submission import AggregatedSubmission
from app.services.marking_scheme_service import MarkingSchemeService


class GradingModelError(ValueError):
    """Raised when grading cannot be completed safely."""


class GradingModel:
    """
    Grading model for handwritten answer-sheet evaluation.

    Input:
        - Aggregated student submission
        - Teacher-approved and locked marking scheme

    Output:
        - Question-wise marks
        - Reasons
        - Confidence
        - Total obtained marks
        - Percentage

    The model is intentionally provider-agnostic.
    Current provider/model comes from ModelProvider.
    """

    def __init__(
        self,
        provider: ModelProvider | None = None,
    ) -> None:
        self.provider = provider or get_model_provider()
        self.client = self.provider.get_client()
        self.model = self.provider.get_text_model()

    # ========================================================
    # PUBLIC API
    # ========================================================

    def grade_submission(
        self,
        submission: AggregatedSubmission,
        marking_scheme: MarkingScheme,
        *,
        job_id: str | None = None,
        student_number: str | None = None,
        assessment_id: str | None = None,
        temperature: float = 0.0,
    ) -> EvaluationResult:

        # ----------------------------------------------------
        # Marking scheme validation
        # ----------------------------------------------------

        MarkingSchemeService.validate(marking_scheme)

        if marking_scheme.status != "locked":
            raise GradingModelError(
                "Grading requires a locked marking scheme."
            )

        # ----------------------------------------------------
        # Objective answer-key validation
        # ----------------------------------------------------

        for question in marking_scheme.questions:
            if (
                question.question_type == "mcq"
                and not question.correct_answer
            ):
                raise GradingModelError(
                    f"Q{question.question_no}: MCQ requires "
                    "a correct_answer in the locked marking scheme."
                )

        # ----------------------------------------------------
        # Submission validation
        # ----------------------------------------------------

        if submission.review_required:
            raise GradingModelError(
                "Submission requires review before grading. "
                "Resolve extraction conflicts or unresolved "
                "answers first."
            )

        resolved_job_id = (
            job_id
            or submission.job_id
            or "UNASSIGNED"
        )

        resolved_student_number = (
            student_number
            or submission.student_number
            or "UNKNOWN"
        )

        # ----------------------------------------------------
        # Build grading input
        # ----------------------------------------------------

        grading_input = self._build_grading_input(
            submission=submission,
            marking_scheme=marking_scheme,
        )

        prompt = self._build_prompt(
            grading_input=grading_input,
            marking_scheme=marking_scheme,
        )

        # ----------------------------------------------------
        # Model call
        # ----------------------------------------------------

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict academic grading model. "
                        "Evaluate student answers only according "
                        "to the supplied marking scheme. "
                        "Do not invent missing answers. "
                        "Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=temperature,
            response_format={
                "type": "text",
            },
        )

        raw_content = response.choices[0].message.content

        if not raw_content:
            raise GradingModelError(
                "Grading model returned empty content."
            )

        parsed = self._parse_json(raw_content)

        # ----------------------------------------------------
        # Validate model output
        # ----------------------------------------------------

        graded_questions = self._validate_model_output(
            parsed=parsed,
            marking_scheme=marking_scheme,
        )

        # ----------------------------------------------------
        # Deterministic MCQ scoring
        # ----------------------------------------------------

        self._apply_deterministic_mcq_scoring(
            graded_questions=graded_questions,
            submission=submission,
            marking_scheme=marking_scheme,
        )

        # ----------------------------------------------------
        # Deterministic totals
        # ----------------------------------------------------

        obtained_marks = sum(
            float(item["marks_awarded"])
            for item in graded_questions
        )

        total_marks = float(
            marking_scheme.total_marks
        )

        if total_marks <= 0:
            raise GradingModelError(
                "Marking scheme total marks must be greater "
                "than zero."
            )

        percentage = (
            obtained_marks / total_marks
        ) * 100.0

        # ----------------------------------------------------
        # Build question evaluations
        # ----------------------------------------------------

        answers_by_question = {
            answer.question_no: answer
            for answer in submission.answers
        }

        question_evaluations: list[QuestionEvaluation] = []

        for item in graded_questions:

            question_no = int(
                item["question_no"]
            )

            scheme_question = next(
                question
                for question in marking_scheme.questions
                if question.question_no == question_no
            )

            aggregated_answer = answers_by_question.get(
                question_no
            )

            extracted_answer = (
                aggregated_answer.answer
                if aggregated_answer
                else None
            )

            question_evaluations.append(
                QuestionEvaluation(
                    question_no=str(question_no),
                    max_marks=float(
                        scheme_question.max_marks
                    ),
                    marks_awarded=float(
                        item["marks_awarded"]
                    ),
                    extracted_answer=extracted_answer,
                    reason=str(
                        item.get("reason", "")
                    ),
                    confidence=float(
                        item.get("confidence", 0.0)
                    ),
                )
            )

        # ----------------------------------------------------
        # Page grouping
        # ----------------------------------------------------

        page_results = self._build_page_results(
            job_id=resolved_job_id,
            question_evaluations=question_evaluations,
            submission=submission,
        )

        # ----------------------------------------------------
        # Final evaluation result
        # ----------------------------------------------------

        return EvaluationResult(
            job_id=resolved_job_id,
            status="graded",
            student_id=resolved_student_number,
            assessment_id=(
                assessment_id
                or marking_scheme.assessment_id
            ),
            pages_processed=submission.processed_pages,
            total_pages=submission.total_pages,
            total_marks=total_marks,
            obtained_marks=obtained_marks,
            percentage=percentage,
            pages=[
                page.model_dump()
                for page in page_results
            ],
            warnings=[],
        )

    # ========================================================
    # INPUT BUILDING
    # ========================================================

    @staticmethod
    def _build_grading_input(
        submission: AggregatedSubmission,
        marking_scheme: MarkingScheme,
    ) -> dict[str, Any]:

        answers_by_question = {
            answer.question_no: answer
            for answer in submission.answers
        }

        questions: list[dict[str, Any]] = []

        for scheme_question in marking_scheme.questions:

            aggregated_answer = answers_by_question.get(
                scheme_question.question_no
            )

            if aggregated_answer is None:
                answer_text = "[NO ANSWER]"
            else:
                answer_text = (
                    aggregated_answer.answer
                    or "[NO ANSWER]"
                )

            criteria = [
                {
                    "criterion_id": criterion.criterion_id,
                    "description": criterion.description,
                    "marks": criterion.marks,
                    "accepted_points": criterion.accepted_points,
                }
                for criterion in scheme_question.criteria
            ]

            questions.append(
                {
                    "question_no": scheme_question.question_no,
                    "max_marks": scheme_question.max_marks,
                    "question_type": scheme_question.question_type,
                    "correct_answer": scheme_question.correct_answer,
                    "evaluation_guidance": (
                        scheme_question.evaluation_guidance
                    ),
                    "criteria": criteria,
                    "student_answer": answer_text,
                }
            )

        return {
            "assessment_id": marking_scheme.assessment_id,
            "total_marks": marking_scheme.total_marks,
            "questions": questions,
        }

    # ========================================================
    # PROMPT
    # ========================================================

    @staticmethod
    def _build_prompt(
        grading_input: dict[str, Any],
        marking_scheme: MarkingScheme,
    ) -> str:

        return f"""
Grade the student answer sheet strictly according to the
provided marking scheme.

IMPORTANT RULES:
1. Grade every question in the marking scheme.
2. Never award more than the question maximum marks.
3. Award 0 when there is no answer or no valid credit.
4. Use the question_type field to decide how the answer should be evaluated.
5. MCQ, true_false, and fill_in:
   - Award the full question marks when the student's answer is correct according to the marking scheme.
   - Award 0 when the answer is incorrect.
   - Do not give partial marks unless the marking scheme explicitly allows them.
6. MCQ grading is authoritative:
   - The "correct_answer" field is the official approved answer key.
   - Do not guess or infer a different correct option.
   - Normalize case and simple formatting differences such as
     "B", "b", "(B)", "Option B", and "B.".
   - If the student's normalized answer matches the normalized
     correct_answer, award the full question marks.
   - If the student's answer does not match the correct_answer,
     award 0 marks.
   - Do not award partial marks for MCQ unless the marking scheme
     explicitly defines partial credit.

7. Subjective questions:
   - Evaluate the meaning, concepts, facts, reasoning, and key points in the student's answer.
   - Do not require exact wording from the reference answer.
   - Accept semantically equivalent wording when it satisfies the marking criteria.
   - Give partial marks when the student demonstrates partial understanding.
   - For a 2-mark subjective question, use the available evidence to distinguish between approximately:
     * 2.0 marks: complete and correct answer.
     * 1.5 marks: mostly correct with a minor omission or weakness.
     * 1.0 mark: core concept is correct but an important part is missing or incomplete.
     * 0.5 marks: limited but relevant understanding.
     * 0 marks: incorrect, irrelevant, contradictory, or unanswered.
   - These are grading guidelines, not mandatory fixed scores; follow the marking criteria when they specify different partial-credit rules.
   - Do not penalize spelling or grammar unless the marking scheme explicitly requires it.
8. Numerical, formula, diagram, and mixed questions must be graded according to their specific marking criteria and evaluation guidance.
9. The marking scheme is authoritative. Do not invent new requirements, change criteria, or use outside information to alter the grading.
10. Give a short reason explaining why the marks were awarded.
11. Confidence must be between 0 and 1.
12. Return ONLY JSON.

MARKING SCHEME:
{json.dumps(grading_input, ensure_ascii=False, indent=2)}

REQUIRED OUTPUT FORMAT:
{{
  "questions": [
    {{
      "question_no": 1,
      "marks_awarded": 0,
      "reason": "Short grading reason",
      "confidence": 0.95
    }}
  ]
}}
"""

    # ========================================================
    # JSON PARSING
    # ========================================================

    @staticmethod
    def _parse_json(
        raw_content: str,
    ) -> dict[str, Any]:

        content = raw_content.strip()

        if content.startswith("```"):
            lines = content.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            content = "\n".join(lines).strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise GradingModelError(
                "Grading model returned invalid JSON."
            ) from exc

        if not isinstance(parsed, dict):
            raise GradingModelError(
                "Grading model output must be a JSON object."
            )

        return parsed

    # ========================================================
    # MODEL OUTPUT VALIDATION
    # ========================================================

    @staticmethod
    def _validate_model_output(
        parsed: dict[str, Any],
        marking_scheme: MarkingScheme,
    ) -> list[dict[str, Any]]:

        raw_questions = parsed.get("questions")

        if not isinstance(raw_questions, list):
            raise GradingModelError(
                "Grading output does not contain a valid "
                "'questions' list."
            )

        expected_numbers = {
            question.question_no
            for question in marking_scheme.questions
        }

        actual_numbers: list[int] = []
        validated_questions: list[dict[str, Any]] = []

        for item in raw_questions:

            if not isinstance(item, dict):
                raise GradingModelError(
                    "Each graded question must be a JSON object."
                )

            try:
                question_no = int(
                    item["question_no"]
                )
                marks_awarded = float(
                    item["marks_awarded"]
                )
                confidence = float(
                    item.get("confidence", 0.0)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise GradingModelError(
                    "Invalid question grading entry."
                ) from exc

            if question_no in actual_numbers:
                raise GradingModelError(
                    f"Duplicate grading result for Q{question_no}."
                )

            if question_no not in expected_numbers:
                raise GradingModelError(
                    f"Unexpected question number Q{question_no} "
                    "returned by grading model."
                )

            scheme_question = next(
                question
                for question in marking_scheme.questions
                if question.question_no == question_no
            )

            if marks_awarded < 0:
                raise GradingModelError(
                    f"Q{question_no}: awarded marks cannot "
                    "be negative."
                )

            if marks_awarded > scheme_question.max_marks:
                raise GradingModelError(
                    f"Q{question_no}: awarded marks "
                    f"{marks_awarded} exceed maximum "
                    f"{scheme_question.max_marks}."
                )

            if not 0 <= confidence <= 1:
                raise GradingModelError(
                    f"Q{question_no}: confidence must be "
                    "between 0 and 1."
                )

            reason = str(
                item.get("reason", "")
            ).strip()

            if not reason:
                raise GradingModelError(
                    f"Q{question_no}: grading reason is required."
                )

            validated_questions.append(
                {
                    "question_no": question_no,
                    "marks_awarded": marks_awarded,
                    "reason": reason,
                    "confidence": confidence,
                }
            )

            actual_numbers.append(question_no)

        actual_set = set(actual_numbers)

        missing_questions = (
            expected_numbers - actual_set
        )

        if missing_questions:
            formatted = ", ".join(
                f"Q{number}"
                for number in sorted(missing_questions)
            )

            raise GradingModelError(
                "Grading model did not return results for: "
                + formatted
            )

        return sorted(
            validated_questions,
            key=lambda item: item["question_no"],
        )

    # ========================================================
    # DETERMINISTIC MCQ SCORING
    # ========================================================

    @staticmethod
    def _normalize_mcq_answer(value: str | None) -> str:
        """
        Normalize an MCQ answer for deterministic comparison.

        Supported examples:
            B
            b
            (B)
            B.
            Option B
            Option B.
            B. Booth's algorithm
            (B) Booth's algorithm
        """

        text = str(value or "").strip().lower()

        if not text:
            return ""

        # Normalize common Unicode punctuation.
        text = (
            text
            .replace("?", "'")
            .replace("?", "'")
            .replace("?", '"')
            .replace("?", '"')
        )

        # Normalize whitespace.
        text = re.sub(r"\s+", " ", text).strip()

        # Pure option marker:
        # B
        # (B)
        # [B]
        # Option B
        marker_match = re.fullmatch(
            r"(?:option\s*)?[\(\[\{]?\s*([a-z])\s*[\)\]\}\.]?",
            text,
        )

        if marker_match:
            return marker_match.group(1)

        # Remove option marker from answers such as:
        # B. Booth's algorithm
        # (B) Booth's algorithm
        # Option B - Booth's algorithm
        text = re.sub(
            r"^(?:option\s*)?[\(\[\{]?\s*[a-z]\s*[\)\]\}\.\:\-]\s*",
            "",
            text,
        )

        # Final whitespace cleanup.
        text = re.sub(r"\s+", " ", text).strip()

        # Ignore harmless trailing punctuation.
        text = text.strip(" .,:;")

        return text


    @classmethod
    def _apply_deterministic_mcq_scoring(
        cls,
        graded_questions: list[dict[str, Any]],
        submission: AggregatedSubmission,
        marking_scheme: MarkingScheme,
    ) -> None:
        """
        Override LLM-produced marks for MCQs using the
        teacher-approved answer key.

        MCQ grading is deterministic:
            correct -> full marks
            incorrect -> 0
            unanswered -> 0
        """

        answers_by_question = {
            answer.question_no: answer
            for answer in submission.answers
        }

        for item in graded_questions:
            question_no = int(item["question_no"])

            scheme_question = next(
                question
                for question in marking_scheme.questions
                if question.question_no == question_no
            )

            if scheme_question.question_type != "mcq":
                continue

            correct_answer = scheme_question.correct_answer

            if not correct_answer:
                raise GradingModelError(
                    f"Q{question_no}: MCQ requires "
                    "a correct_answer in the locked marking scheme."
                )

            aggregated_answer = answers_by_question.get(
                question_no
            )

            student_answer = (
                aggregated_answer.answer
                if aggregated_answer
                else None
            )

            student_normalized = cls._normalize_mcq_answer(
                student_answer
            )

            correct_normalized = cls._normalize_mcq_answer(
                correct_answer
            )

            # ------------------------------------------------
            # No student answer
            # ------------------------------------------------

            if not student_normalized:
                item["marks_awarded"] = 0.0
                item["reason"] = (
                    "No student answer was extracted."
                )
                item["confidence"] = 1.0
                continue

            # ------------------------------------------------
            # Exact normalized answer-key comparison
            # ------------------------------------------------

            if student_normalized == correct_normalized:
                item["marks_awarded"] = float(
                    scheme_question.max_marks
                )
                item["reason"] = (
                    "Student answer matches the approved "
                    "MCQ answer key."
                )
            else:
                item["marks_awarded"] = 0.0
                item["reason"] = (
                    "Student answer does not match the approved "
                    "MCQ answer key."
                )

            # The mark decision itself is deterministic.
            item["confidence"] = 1.0


    # ========================================================
    # PAGE RESULTS
    # ========================================================

    @staticmethod
    def _build_page_results(
        job_id: str,
        question_evaluations: list[QuestionEvaluation],
        submission: AggregatedSubmission,
    ) -> list[PageResult]:

        answer_map = {
            answer.question_no: answer
            for answer in submission.answers
        }

        pages: dict[int, list[QuestionEvaluation]] = {}

        for evaluation in question_evaluations:

            question_no = int(
                evaluation.question_no
            )

            aggregated_answer = answer_map.get(
                question_no
            )

            if not aggregated_answer:
                continue

            if aggregated_answer.observations:
                source_page = min(
                    observation.page_no
                    for observation
                    in aggregated_answer.observations
                )
            else:
                source_page = 1

            pages.setdefault(
                source_page,
                [],
            ).append(evaluation)

        results: list[PageResult] = []

        for page_number, questions in sorted(
            pages.items()
        ):

            page_max_marks = sum(
                question.max_marks or 0.0
                for question in questions
            )

            page_marks_awarded = sum(
                question.marks_awarded or 0.0
                for question in questions
            )

            confidence = (
                sum(
                    question.confidence
                    for question in questions
                )
                / len(questions)
                if questions
                else 0.0
            )

            results.append(
                PageResult(
                    job_id=job_id,
                    page_number=page_number,
                    status="graded",
                    questions=questions,
                    page_max_marks=page_max_marks,
                    page_marks_awarded=page_marks_awarded,
                    confidence=confidence,
                    warnings=[],
                    metadata={
                        "grading": True,
                    },
                )
            )

        return results


# ======================================================================
# FUTURE GRADING REVIEW MODEL
# ======================================================================
#
# KEEP THIS DISABLED FOR NOW.
#
# Later, when a better independent model is available, this reviewer
# will receive:
#     - original student answers
#     - marking scheme
#     - grading result
#
# It will independently verify whether the awarded marks are justified.
#
# Example future structure:
#
#
# class GradingReviewModel:
#
#     def __init__(
#         self,
#         provider: ModelProvider | None = None,
#     ) -> None:
#         self.provider = provider or get_model_provider()
#         self.client = self.provider.get_client()
#         self.model = self.provider.get_text_model()
#
#     def review(
#         self,
#         evaluation_result: EvaluationResult,
#         marking_scheme: MarkingScheme,
#         submission: AggregatedSubmission,
#     ) -> dict[str, Any]:
#         ...
#
#
# The workflow will later become:
#
#     GRADING MODEL
#          ↓
#     GRADING REVIEW MODEL
#          ↓
#     PASS → Teacher Review
#       OR
#     FAIL → re-grade affected questions
#
# ======================================================================




