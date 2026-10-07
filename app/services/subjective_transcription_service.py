from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.schemas.subjective import (
    SubjectiveAnswerBlock,
    SubjectiveDiagram,
    SubjectiveEquation,
    SubjectiveLine,
    SubjectivePageResult,
)
from app.services.vision_service import (
    VisionService,
)


class SubjectiveTranscriptionService:
    """
    Subjective handwritten-answer transcription service.

    Strategy:

        1. Whole-page vision attempt
                    ↓
        2. Validate structured result
                    ↓
        3. If insufficient -> segmented fallback
                    ↓
        4. Merge region results
                    ↓
        5. Return structured transcription

    The service does NOT:
        - assign marks
        - judge correctness
        - calculate scores
        - update ResultStore
        - call TCO Master Agent
        - depend on a specific model
    """

    # ========================================================
    # CONFIG
    # ========================================================

    FALLBACK_ROWS = 6

    FALLBACK_OVERLAP_RATIO = 0.15

    # Below this confidence, the result is considered
    # insufficient for whole-page-only processing.
    WHOLE_PAGE_MIN_CONFIDENCE = 0.60

    # Below this confidence, the specific block needs review.
    BLOCK_MIN_CONFIDENCE = 0.75

    # ========================================================
    # WHOLE-PAGE PROMPT
    # ========================================================

    WHOLE_PAGE_PROMPT = """
You are transcribing a student's handwritten answer sheet.

This is the COMPLETE PAGE.

Read the page as a document, preserving the spatial
relationship between questions and their handwritten answers.

Your task is ONLY transcription.

DO NOT:
- assign marks
- judge correctness
- summarize
- rewrite in better language
- correct spelling
- invent missing words
- explain the answer

PRESERVE:
- original wording as closely as possible
- sentence order
- steps
- numbers
- mathematical notation
- equations
- diagrams

Return ONLY valid JSON.

Required format:

{
  "blocks": [
    {
      "question_no": 3,
      "answer_text": "student's handwritten answer",
      "lines": [
        {
          "text": "first handwritten line",
          "confidence": 0.85
        }
      ],
      "equations": [
        {
          "text": "2 x 3 = 6",
          "confidence": 0.90
        }
      ],
      "diagrams": [
        {
          "description": "visible diagram",
          "confidence": 0.75
        }
      ],
      "confidence": 0.82
    }
  ],
  "overall_confidence": 0.80
}

RULES:

1. One block should represent one answer/question region.
2. question_no should be the visible question number when
   it can be determined.
3. If question number is not readable, use null.
4. answer_text must contain ONLY visible student writing.
5. Use [UNCLEAR] for unreadable text.
6. Do not invent missing words.
7. Do not assign marks.
8. Do not judge correctness.
9. Preserve handwritten wording.
10. Preserve equations separately.
11. Preserve diagrams separately.
12. confidence must be between 0 and 1.
13. overall_confidence must be between 0 and 1.
14. Return no explanation outside JSON.

If the page contains no readable student writing,
return:

{
  "blocks": [],
  "overall_confidence": 0.0
}
"""

    # ========================================================
    # FALLBACK PROMPT
    # ========================================================

    FALLBACK_PROMPT = """
You are transcribing handwritten student answers from ONE
REGION of an answer sheet.

This is NOT the complete page.

Read only the handwriting that is visible in this region.

Your task is ONLY transcription.

DO NOT:
- assign marks
- judge correctness
- summarize
- improve grammar
- correct spelling
- invent missing words

PRESERVE:
- original wording
- line order
- steps
- equations
- mathematical symbols
- visible diagrams

Return ONLY valid JSON.

Required format:

{
  "blocks": [
    {
      "question_no": 3,
      "answer_text": "visible handwritten answer",
      "lines": [
        {
          "text": "line one",
          "confidence": 0.85
        }
      ],
      "equations": [],
      "diagrams": [],
      "confidence": 0.80
    }
  ],
  "overall_confidence": 0.80
}

RULES:

1. Only report handwriting actually visible in this region.
2. question_no must be the visible question number.
3. If question number cannot be determined, use null.
4. Use [UNCLEAR] rather than guessing.
5. Do not assign marks.
6. Do not judge correctness.
7. Do not invent text.
8. Preserve original wording as closely as possible.
9. confidence must be between 0 and 1.
10. Return no explanation outside JSON.

If no readable handwritten answer exists in this region:

{
  "blocks": [],
  "overall_confidence": 0.0
}
"""

    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        vision_service: VisionService | None = None,
    ) -> None:

        self.vision_service = (
            vision_service
            or VisionService()
        )

    # ========================================================
    # PARSE RESPONSE
    # ========================================================

    @staticmethod
    def parse_response(
        raw: str,
    ) -> dict:

        if raw is None:
            raise ValueError(
                "Vision model returned no content."
            )

        cleaned = raw.strip()

        if not cleaned:
            raise ValueError(
                "Vision model returned empty content."
            )

        # ----------------------------------------------------
        # Remove markdown fences
        # ----------------------------------------------------

        if cleaned.startswith("```"):

            lines = cleaned.splitlines()

            if (
                lines
                and lines[0]
                .strip()
                .startswith("```")
            ):
                lines = lines[1:]

            if (
                lines
                and lines[-1]
                .strip()
                == "```"
            ):
                lines = lines[:-1]

            cleaned = "\n".join(
                lines
            ).strip()

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        try:

            data = json.loads(
                cleaned
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Invalid JSON from vision model.\n\n"
                f"RAW OUTPUT:\n{raw}\n\n"
                f"JSON ERROR:\n{exc}"
            ) from exc

        # ----------------------------------------------------
        # Normalize array
        # ----------------------------------------------------

        if isinstance(
            data,
            list,
        ):

            data = {
                "blocks": data,
                "overall_confidence": 0.0,
            }

        if not isinstance(
            data,
            dict,
        ):

            raise ValueError(
                "Vision response must be "
                "a JSON object or list."
            )

        if "blocks" not in data:

            data["blocks"] = []

        if not isinstance(
            data["blocks"],
            list,
        ):

            raise ValueError(
                "'blocks' must be a list."
            )

        return data

    # ========================================================
    # CONFIDENCE
    # ========================================================

    @staticmethod
    def clamp_confidence(
        value,
    ) -> float:

        try:
            value = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            value = 0.0

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    # ========================================================
    # NORMALIZE BLOCK
    # ========================================================

    def normalize_block(
        self,
        raw_block: dict,
        source_region: str,
    ) -> SubjectiveAnswerBlock:

        question_no = raw_block.get(
            "question_no"
        )

        if question_no is not None:

            try:
                question_no = int(
                    question_no
                )
            except (
                TypeError,
                ValueError,
            ):
                question_no = None

        answer_text = raw_block.get(
            "answer_text",
            "",
        )

        if answer_text is None:
            answer_text = ""

        answer_text = str(
            answer_text
        ).strip()

        confidence = (
            self.clamp_confidence(
                raw_block.get(
                    "confidence",
                    0.0,
                )
            )
        )

        # ----------------------------------------------------
        # Lines
        # ----------------------------------------------------

        lines = []

        raw_lines = raw_block.get(
            "lines",
            [],
        )

        if isinstance(
            raw_lines,
            list,
        ):

            for raw_line in raw_lines:

                if not isinstance(
                    raw_line,
                    dict,
                ):
                    continue

                text = raw_line.get(
                    "text",
                    "",
                )

                if text is None:
                    text = ""

                text = str(
                    text
                ).strip()

                if not text:
                    continue

                line_confidence = (
                    self.clamp_confidence(
                        raw_line.get(
                            "confidence",
                            confidence,
                        )
                    )
                )

                lines.append(
                    SubjectiveLine(
                        text=text,
                        confidence=line_confidence,
                    )
                )

        # ----------------------------------------------------
        # Equations
        # ----------------------------------------------------

        equations = []

        raw_equations = raw_block.get(
            "equations",
            [],
        )

        if isinstance(
            raw_equations,
            list,
        ):

            for raw_equation in raw_equations:

                if not isinstance(
                    raw_equation,
                    dict,
                ):
                    continue

                equation_text = (
                    raw_equation.get(
                        "text",
                        "",
                    )
                )

                if equation_text is None:
                    equation_text = ""

                equation_text = str(
                    equation_text
                ).strip()

                if not equation_text:
                    continue

                equation_confidence = (
                    self.clamp_confidence(
                        raw_equation.get(
                            "confidence",
                            confidence,
                        )
                    )
                )

                equations.append(
                    SubjectiveEquation(
                        text=equation_text,
                        confidence=equation_confidence,
                    )
                )

        # ----------------------------------------------------
        # Diagrams
        # ----------------------------------------------------

        diagrams = []

        raw_diagrams = raw_block.get(
            "diagrams",
            [],
        )

        if isinstance(
            raw_diagrams,
            list,
        ):

            for raw_diagram in raw_diagrams:

                if not isinstance(
                    raw_diagram,
                    dict,
                ):
                    continue

                description = (
                    raw_diagram.get(
                        "description",
                        "",
                    )
                )

                if description is None:
                    description = ""

                description = str(
                    description
                ).strip()

                if not description:
                    continue

                diagram_confidence = (
                    self.clamp_confidence(
                        raw_diagram.get(
                            "confidence",
                            confidence,
                        )
                    )
                )

                diagrams.append(
                    SubjectiveDiagram(
                        description=description,
                        confidence=diagram_confidence,
                    )
                )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        needs_review = (
            question_no is None
            or not answer_text
            or "[UNCLEAR]" in answer_text
            or confidence
            < self.BLOCK_MIN_CONFIDENCE
        )

        if needs_review:

            status = (
                "REVIEW_REQUIRED"
            )

            if question_no is None:

                warning = (
                    "Question number could not "
                    "be determined."
                )

            elif not answer_text:

                warning = (
                    "No readable answer text "
                    "was extracted."
                )

            elif "[UNCLEAR]" in answer_text:

                warning = (
                    "Answer contains unreadable "
                    "handwriting."
                )

            else:

                warning = (
                    "Low-confidence transcription."
                )

        else:

            status = "EXTRACTED"
            warning = None

        return SubjectiveAnswerBlock(
            question_no=question_no,
            answer_text=answer_text,
            lines=lines,
            equations=equations,
            diagrams=diagrams,
            confidence=confidence,
            status=status,
            source_region=source_region,
            warning=warning,
        )

    # ========================================================
    # NORMALIZE RESPONSE
    # ========================================================

    def normalize_response(
        self,
        response: dict,
        source_region: str,
    ) -> tuple[
        list[SubjectiveAnswerBlock],
        float,
    ]:

        blocks = []

        for raw_block in response.get(
            "blocks",
            [],
        ):

            if not isinstance(
                raw_block,
                dict,
            ):
                continue

            block = (
                self.normalize_block(
                    raw_block=raw_block,
                    source_region=source_region,
                )
            )

            # Ignore completely empty blocks.
            if (
                block.question_no is None
                and not block.answer_text
                and not block.lines
                and not block.equations
                and not block.diagrams
            ):
                continue

            blocks.append(
                block
            )

        overall_confidence = (
            self.clamp_confidence(
                response.get(
                    "overall_confidence",
                    0.0,
                )
            )
        )

        return (
            blocks,
            overall_confidence,
        )

    # ========================================================
    # CREATE FALLBACK STRIPS
    # ========================================================

    def create_fallback_strips(
        self,
        image_path: str | Path,
        output_dir: str | Path,
    ) -> list[tuple[str, Path]]:

        image_path = Path(
            image_path
        )

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        image = Image.open(
            image_path
        )

        image.load()

        image = image.convert(
            "RGB"
        )

        width, height = image.size

        base_height = (
            height
            // self.FALLBACK_ROWS
        )

        overlap = int(
            base_height
            * self.FALLBACK_OVERLAP_RATIO
        )

        results = []

        for index in range(
            self.FALLBACK_ROWS
        ):

            y1 = (
                index * base_height
                - (
                    overlap
                    if index > 0
                    else 0
                )
            )

            y2 = (
                (index + 1)
                * base_height
                + (
                    overlap
                    if index
                    < self.FALLBACK_ROWS - 1
                    else 0
                )
            )

            y1 = max(
                0,
                y1,
            )

            y2 = min(
                height,
                y2,
            )

            crop = image.crop(
                (
                    0,
                    y1,
                    width,
                    y2,
                )
            )

            name = (
                f"region_{index + 1:02d}"
            )

            output_path = (
                output_dir
                / f"{name}.jpg"
            )

            crop.save(
                output_path,
                format="JPEG",
                quality=95,
                optimize=True,
            )

            results.append(
                (
                    name,
                    output_path,
                )
            )

        return results

    # ========================================================
    # MERGE BLOCKS
    # ========================================================

    def merge_blocks(
        self,
        blocks: list[
            SubjectiveAnswerBlock
        ],
    ) -> list[
        SubjectiveAnswerBlock
    ]:
        """
        Merge duplicate question blocks caused by overlapping
        fallback regions.

        Important:
        We do not rewrite/transcribe text here.

        We simply select the strongest candidate when two
        candidates are effectively identical and preserve
        conflicts for human review.
        """

        grouped: dict[
            int,
            list[SubjectiveAnswerBlock],
        ] = {}

        unresolved = []

        for block in blocks:

            if block.question_no is None:

                unresolved.append(
                    block
                )

                continue

            grouped.setdefault(
                block.question_no,
                [],
            ).append(
                block
            )

        final_blocks = []

        # ----------------------------------------------------
        # Numbered blocks
        # ----------------------------------------------------

        for question_no, candidates in grouped.items():

            candidates.sort(
                key=lambda block: (
                    block.answer_text
                    != "[UNCLEAR]",
                    block.confidence,
                ),
                reverse=True,
            )

            best = candidates[0]

            readable = {
                item.answer_text
                for item in candidates
                if item.answer_text
                and "[UNCLEAR]"
                not in item.answer_text
            }

            # ------------------------------------------------
            # Conflict detection
            # ------------------------------------------------

            if len(readable) > 1:

                best.status = (
                    "REVIEW_REQUIRED"
                )

                best.warning = (
                    "Conflicting transcription "
                    "candidates detected across "
                    "regions."
                )

            final_blocks.append(
                best
            )

        # ----------------------------------------------------
        # Add unresolved blocks
        # ----------------------------------------------------

        final_blocks.extend(
            unresolved
        )

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        final_blocks.sort(
            key=lambda block: (
                block.question_no
                if block.question_no is not None
                else 10**9
            )
        )

        return final_blocks

    # ========================================================
    # WHOLE PAGE
    # ========================================================

    def try_whole_page(
        self,
        image_path: str | Path,
    ) -> tuple[
        list[SubjectiveAnswerBlock],
        float,
        bool,
    ]:
        """
        Attempt transcription using one full-page image.
        """

        try:

            response = (
                self.vision_service
                .analyze_image(
                    image_path=image_path,
                    prompt=self.WHOLE_PAGE_PROMPT,
                    max_dimension=2048,
                    jpeg_quality=95,
                    temperature=0.0,
                )
            )

        except Exception as exc:

            return (
                [],
                0.0,
                False,
            )

        try:

            blocks, confidence = (
                self.normalize_response(
                    response["parsed"],
                    source_region="whole_page",
                )
            )

        except Exception:

            return (
                [],
                0.0,
                False,
            )

        usable = bool(
            blocks
        )

        sufficient = (
            usable
            and confidence
            >= self.WHOLE_PAGE_MIN_CONFIDENCE
        )

        return (
            blocks,
            confidence,
            sufficient,
        )

    # ========================================================
    # FALLBACK
    # ========================================================

    def run_fallback(
        self,
        image_path: str | Path,
        output_dir: str | Path,
    ) -> list[
        SubjectiveAnswerBlock
    ]:

        regions = (
            self.create_fallback_strips(
                image_path=image_path,
                output_dir=output_dir,
            )
        )

        blocks = []

        for region_name, region_path in regions:

            print("\n")
            print("=" * 70)
            print(
                f"SUBJECTIVE FALLBACK: "
                f"{region_name}"
            )
            print("=" * 70)

            try:

                response = (
                    self.vision_service
                    .analyze_image(
                        image_path=region_path,
                        prompt=self.FALLBACK_PROMPT,
                        max_dimension=2048,
                        jpeg_quality=95,
                        temperature=0.0,
                    )
                )

                normalized_blocks, _ = (
                    self.normalize_response(
                        response["parsed"],
                        source_region=region_name,
                    )
                )

                blocks.extend(
                    normalized_blocks
                )

            except Exception as exc:

                print(
                    f"Region failed: {exc}"
                )

        return blocks

    # ========================================================
    # PUBLIC API
    # ========================================================

    def extract_page(
        self,
        image_path: str | Path,
        page_no: int,
        fallback_output_dir: str | Path,
        force_fallback: bool = False,
    ) -> SubjectivePageResult:
        """
        Public page-level transcription API.

        Production-friendly behavior:

            whole page
                ↓
            sufficient?
              ↙      ↘
            yes       no
             ↓         ↓
           return   fallback regions
        """

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Page image not found:\n"
                f"{image_path}"
            )

        # ----------------------------------------------------
        # Whole-page path
        # ----------------------------------------------------

        if not force_fallback:

            (
                whole_blocks,
                whole_confidence,
                sufficient,
            ) = self.try_whole_page(
                image_path
            )

            if sufficient:

                review_required = any(
                    block.status
                    == "REVIEW_REQUIRED"
                    for block in whole_blocks
                )

                warnings = []

                if review_required:

                    warnings.append(
                        "One or more subjective "
                        "answer blocks require review."
                    )

                return SubjectivePageResult(
                    page_no=page_no,
                    blocks=whole_blocks,
                    extraction_mode="whole_page",
                    review_required=review_required,
                    warnings=warnings,
                    regions_processed=1,
                    fallback_used=False,
                )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        fallback_blocks = (
            self.run_fallback(
                image_path=image_path,
                output_dir=fallback_output_dir,
            )
        )

        merged = (
            self.merge_blocks(
                fallback_blocks
            )
        )

        warnings = [
            "Whole-page transcription was "
            "insufficient or fallback was forced."
        ]

        if any(
            block.status
            == "REVIEW_REQUIRED"
            for block in merged
        ):

            warnings.append(
                "One or more subjective "
                "answer blocks require review."
            )

        review_required = bool(
            not merged
            or any(
                block.status
                == "REVIEW_REQUIRED"
                for block in merged
            )
        )

        return SubjectivePageResult(
            page_no=page_no,
            blocks=merged,
            extraction_mode="segmented_fallback",
            review_required=review_required,
            warnings=warnings,
            regions_processed=self.FALLBACK_ROWS,
            fallback_used=True,
        )