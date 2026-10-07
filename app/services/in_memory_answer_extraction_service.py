from __future__ import annotations

import json
from typing import Any

from PIL import Image

from app.schemas.extraction import (
    ExtractedAnswer,
    PageExtractionResult,
)
from app.services.answer_extraction_service import (
    AnswerExtractionService,
)
from app.services.in_memory_vision_service import (
    InMemoryVisionService,
)


class InMemoryAnswerExtractionService(
    AnswerExtractionService
):
    """
    Whole-page handwritten answer extraction.

    IMPORTANT:
    - No page tiling.
    - Complete page is sent as one image.
    - Model returns a document-style content structure.
    - Deterministic code converts answer content into
      the existing ExtractedAnswer contract.
    """

    WHOLE_PAGE_PROMPT = r"""
You are extracting ALL handwritten content from ONE COMPLETE page of a
student answer sheet.

Your ONLY job is transcription/extraction.
DO NOT grade.
DO NOT evaluate correctness.
DO NOT solve questions.
DO NOT summarize.
DO NOT shorten.
DO NOT rewrite the student's answer.

CRITICAL RULE:
EVERY handwritten student answer must be extracted, regardless of length.

An answer may be:
- one letter such as A/B/C/D
- True/False
- a word or short phrase
- a sentence
- a paragraph
- a long subjective answer
- multiple paragraphs
- bullet points
- numbered steps
- calculations
- numerical work
- formulas/equations
- tables
- diagram labels or written diagram information
- mixed handwritten content

VERY IMPORTANT:
A long paragraph is still an ANSWER.
Never skip a paragraph just because it is long.
Never extract only MCQs or short answers.
Continue reading the COMPLETE PAGE from top to bottom.

QUESTION NUMBER:
The question number may appear on the left, above the answer, on the same
line, circled, underlined, separated from the answer, or in another position.
Do not assume one fixed location.

For each recognizable student answer:
1. identify its question number if visible/readable;
2. copy the student's actual written content as closely as possible;
3. preserve the complete answer;
4. preserve bullets, steps, numbers, formulas and equations;
5. preserve line meaning and paragraph content;
6. do not replace a long answer with a summary.

If handwriting is genuinely unreadable:
use "[UNCLEAR]" and a lower confidence.

DO NOT invent questions or answers.

Return ONLY valid JSON.

Use exactly this structure:

{
  "page_number": 1,
  "page_type": "handwritten_answer_sheet",
  "content": [
    {
      "question_no": 1,
      "content_type": "answer",
      "text": "complete student-written answer",
      "confidence": 0.95
    }
  ],
  "other_content": [],
  "review_required": false,
  "warnings": []
}

IMPORTANT:
- Put ALL handwritten student answers inside "content".
- Use "content_type": "answer" for both short answers and long paragraphs.
- Do NOT put student answers inside "other_content".
- Do NOT omit a question because its answer is long.
- Do NOT stop after finding a few answers.
- Read the whole page before returning the JSON.
"""

    RECOVERY_PROMPT = r"""
You are performing a second careful reading of ONE COMPLETE PAGE
of a student's handwritten answer sheet.

Previous extraction verification reported this feedback:

{feedback}

Target question numbers:

{targets}

Re-read the COMPLETE ORIGINAL PAGE.

Focus carefully on the target questions while using the complete page
to understand the question-answer boundaries.

Do NOT grade.
Do NOT solve.
Do NOT invent.
Do NOT rewrite the student's answer.

Return the student's actual written answer as closely as possible.

If it is still genuinely unreadable, return "[UNCLEAR]".

Return ONLY valid JSON using exactly this structure:

{
  "page_number": 1,
  "page_type": "handwritten_answer_sheet",
  "content": [
    {
      "question_no": 4,
      "content_type": "answer",
      "text": "student answer",
      "confidence": 0.85
    }
  ],
  "other_content": [],
  "review_required": false,
  "warnings": []
}
"""

    def __init__(
        self,
        vision_service: InMemoryVisionService | None = None,
    ) -> None:

        self.vision_service = (
            vision_service
            or InMemoryVisionService()
        )

    # ========================================================
    # MODEL CALL
    # ========================================================

    def _call_model(
        self,
        image: Image.Image,
        prompt: str,
    ) -> dict[str, Any]:

        result = (
            self.vision_service
            .analyze_pil_image(
                image=image,
                prompt=prompt,
                max_dimension=2048,
                jpeg_quality=95,
                temperature=0,
            )
        )

        parsed = result.get(
            "parsed",
            {},
        )

        if not isinstance(
            parsed,
            dict,
        ):
            raise ValueError(
                "Vision model returned an invalid "
                "JSON object."
            )

        return parsed

    # ========================================================
    # MODEL RESPONSE -> EXTRACTED ANSWERS
    # ========================================================

    def _parse_content(
        self,
        parsed: dict[str, Any],
        page_no: int,
    ) -> list[ExtractedAnswer]:

        content = parsed.get(
            "content",
            [],
        )

        # -----------------------------------------------
        # Compatibility with our temporary questions[]
        # schema if the model ever returns it.
        # -----------------------------------------------

        if not isinstance(
            content,
            list,
        ):

            content = []

        candidates = []

        for item in content:

            if not isinstance(
                item,
                dict,
            ):
                continue

            content_type = str(
                item.get(
                    "content_type",
                    "",
                )
            ).strip().lower()

            if content_type != "answer":
                continue

            question_no = item.get(
                "question_no"
            )

            text = item.get(
                "text",
                "[UNCLEAR]",
            )

            candidate = self.normalize_candidate(
                item={
                    "question_no": question_no,
                    "answer": text,
                    "confidence": item.get(
                        "confidence",
                        0.0,
                    ),
                },
                source_tile=(
                    f"page_{page_no:03d}"
                ),
            )

            candidates.append(
                candidate
            )

        # -----------------------------------------------
        # Fallback if model returns questions[]
        # -----------------------------------------------

        if not candidates:

            questions = parsed.get(
                "questions",
                [],
            )

            if isinstance(
                questions,
                list,
            ):

                for item in questions:

                    if not isinstance(
                        item,
                        dict,
                    ):
                        continue

                    candidate = self.normalize_candidate(
                        item=item,
                        source_tile=(
                            f"page_{page_no:03d}"
                        ),
                    )

                    candidates.append(
                        candidate
                    )

        return candidates

    # ========================================================
    # WHOLE PAGE EXTRACTION
    # ========================================================

    def _extract_whole_page(
        self,
        image: Image.Image,
        page_no: int,
        prompt: str,
    ) -> list[ExtractedAnswer]:

        parsed = self._call_model(
            image=image,
            prompt=prompt,
        )

        print(
            json.dumps(
                parsed,
                indent=2,
                ensure_ascii=False,
            )
        )

        return self._parse_content(
            parsed=parsed,
            page_no=page_no,
        )

    # ========================================================
    # WHOLE PAGE RECOVERY
    # ========================================================

    def run_recovery_in_memory(
        self,
        image: Image.Image,
        page_no: int,
        target_questions: list[int],
        feedback: str | None,
    ) -> list[ExtractedAnswer]:

        if not target_questions:
            return []

        prompt = (
            self.RECOVERY_PROMPT
            .replace(
                "{feedback}",
                feedback
                or "Some answers require careful re-reading.",
            )
            .replace(
                "{targets}",
                ", ".join(
                    f"Q{number}"
                    for number in target_questions
                ),
            )
        )

        print()
        print("=" * 70)
        print(
            "WHOLE-PAGE TARGETED RECOVERY"
        )
        print("=" * 70)
        print(
            "Target questions:",
            target_questions,
        )

        candidates = self._extract_whole_page(
            image=image,
            page_no=page_no,
            prompt=prompt,
        )

        target_set = set(
            target_questions
        )

        return [
            candidate
            for candidate in candidates
            if candidate.question_no
            in target_set
        ]

    # ========================================================
    # EXTRACT ONE COMPLETE PAGE
    # ========================================================

    def extract_page_in_memory(
        self,
        image: Image.Image,
        page_no: int,
        expected_question_numbers: list[int] | None = None,
        enable_recovery: bool = True,
        feedback: str | None = None,
    ) -> PageExtractionResult:

        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                "image must be a PIL.Image.Image"
            )

        expected_question_numbers = sorted(
            {
                int(number)
                for number in (
                    expected_question_numbers
                    or []
                )
            }
        )

        print()
        print("=" * 70)
        print(
            f"WHOLE-PAGE ANSWER EXTRACTION "
            f"FOR PAGE {page_no}"
        )
        print("=" * 70)

        if feedback:

            print(
                "Verifier feedback supplied "
                "for this extraction attempt."
            )

        else:

            print(
                "No verifier feedback. "
                "Initial full-page extraction."
            )

        # ----------------------------------------------------
        # INITIAL EXTRACTION
        # ----------------------------------------------------

        initial_candidates = (
            self._extract_whole_page(
                image=image,
                page_no=page_no,
                prompt=(
                    self.WHOLE_PAGE_PROMPT
                    + (
                        "\n\nPrevious verifier feedback:\n"
                        + feedback
                        if feedback
                        else ""
                    )
                ),
            )
        )

        (
            initial_questions,
            initial_unresolved,
            initial_warnings,
        ) = self.merge_candidates(
            initial_candidates
        )

        missing_questions = (
            self.find_missing_questions(
                initial_questions,
                expected_question_numbers,
            )
        )

        recovery_targets = (
            self.find_recovery_targets(
                initial_questions,
                missing_questions,
            )
        )

        # ----------------------------------------------------
        # EMPTY EXTRACTION IS NOT A PASS CONDITION
        # ----------------------------------------------------

        if not initial_questions:

            initial_warnings.append(
                "No question-answer pairs were "
                "extracted from the complete page."
            )

        print()
        print("=" * 70)
        print("INITIAL WHOLE-PAGE ANALYSIS")
        print("=" * 70)

        print(
            "Detected:",
            [
                item.question_no
                for item in initial_questions
                if item.question_no is not None
            ],
        )

        print(
            "Missing:",
            missing_questions,
        )

        print(
            "Recovery targets:",
            recovery_targets,
        )

        # ----------------------------------------------------
        # RECOVERY
        # ----------------------------------------------------

        recovery_candidates = []

        recovery_attempted = False

        if (
            enable_recovery
            and recovery_targets
        ):

            recovery_attempted = True

            recovery_candidates = (
                self.run_recovery_in_memory(
                    image=image,
                    page_no=page_no,
                    target_questions=recovery_targets,
                    feedback=(
                        feedback
                        or "Re-read uncertain answers carefully."
                    ),
                )
            )

        # ----------------------------------------------------
        # FINAL MERGE
        # ----------------------------------------------------

        combined_candidates = (
            initial_candidates
            + recovery_candidates
        )

        (
            final_questions,
            final_unresolved,
            final_warnings,
        ) = self.merge_candidates(
            combined_candidates
        )

        final_missing = (
            self.find_missing_questions(
                final_questions,
                expected_question_numbers,
            )
        )

        warnings = []

        for warning in (
            initial_warnings
            + final_warnings
        ):

            if warning not in warnings:
                warnings.append(
                    warning
                )

        if final_missing:

            warnings.append(
                "Missing expected question(s): "
                + ", ".join(
                    f"Q{number}"
                    for number in final_missing
                )
            )

        if recovery_attempted:

            warnings.append(
                "Whole-page targeted recovery "
                "was attempted."
            )

        review_required = bool(
            not final_questions
            or final_missing
            or final_unresolved
            or any(
                item.status
                == "REVIEW_REQUIRED"
                for item in final_questions
            )
        )

        print()
        print("=" * 70)
        print("FINAL WHOLE-PAGE EXTRACTION")
        print("=" * 70)

        for item in final_questions:

            print(
                f"Q{item.question_no}: "
                f"{item.answer} | "
                f"confidence={item.confidence:.2f} | "
                f"status={item.status}"
            )

        print()
        print(
            "Review required:",
            review_required,
        )

        return PageExtractionResult(
            page_no=page_no,
            questions=final_questions,
            unresolved_items=final_unresolved,
            missing_questions=final_missing,
            review_required=review_required,
            warnings=warnings,
            processed_tiles=1,
            recovery_attempted=recovery_attempted,
            recovery_tiles=(
                1
                if recovery_attempted
                else 0
            ),
        )

    # ========================================================
    # COMPLETE PDF
    # ========================================================

    def extract_pdf_in_memory(
        self,
        pdf_path: str,
        page_processor,
        expected_question_numbers: list[int] | None = None,
        enable_recovery: bool = True,
    ):

        results = []

        for page_no, image in page_processor.iter_pages(
            pdf_path
        ):

            try:

                result = (
                    self.extract_page_in_memory(
                        image=image,
                        page_no=page_no,
                        expected_question_numbers=(
                            expected_question_numbers
                        ),
                        enable_recovery=enable_recovery,
                    )
                )

                results.append(
                    result
                )

            finally:

                try:
                    image.close()
                except Exception:
                    pass

        return results

