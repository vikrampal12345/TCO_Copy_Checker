import json
from pathlib import Path

from PIL import Image

from app.schemas.extraction import (
    ExtractedAnswer,
    PageExtractionResult,
)
from app.services.vision_service import (
    VisionService,
)


class AnswerExtractionService:
    """
    Extract handwritten answers from one page.

    Main pipeline:

        page image
            ↓
        8 overlapping tiles
            ↓
        initial vision extraction
            ↓
        deterministic merge
            ↓
        expected-question check
            ↓
        targeted recovery
            ↓
        final merge
            ↓
        PageExtractionResult

    This service does NOT:
        - assign marks
        - judge correctness
        - talk to ResultStore
        - orchestrate TCO Master Agent
        - use LangGraph
    """

    # ========================================================
    # TILE CONFIG
    # ========================================================

    ROWS = 4
    COLS = 2

    OVERLAP_X_RATIO = 0.10
    OVERLAP_Y_RATIO = 0.10

    # A readable answer below this threshold requires review.
    MIN_CONFIDENCE = 0.75

    # ========================================================
    # INITIAL EXTRACTION PROMPT
    # ========================================================

    EXTRACTION_PROMPT = """
You are extracting answers from a student's handwritten
answer sheet.

This image is one region of one page.

Read the region carefully from TOP TO BOTTOM.

Identify every visible question number and the student's
answer associated with that question.

Return ONLY valid JSON.

Preferred format:

{
  "questions": [
    {
      "question_no": 7,
      "answer": "A",
      "confidence": 0.90
    }
  ]
}

Rules:

1. One object per visible question.
2. question_no must be the visible question number.
3. answer must be the student's visible answer.
4. For MCQs use A, B, C, or D when clearly visible.
5. Never combine multiple questions.
6. Preserve top-to-bottom order.
7. Never invent a question number.
8. Never invent an answer.
9. If the question number is genuinely unreadable, use null.
10. If the answer is unreadable, use "[UNCLEAR]".
11. Do not assign marks.
12. Do not judge correctness.
13. Do not infer an answer from neighboring questions.
14. confidence must be between 0 and 1.
15. Ignore headings and general instructions.
16. Return no explanation outside JSON.

An uncertain result is better than a guessed result.
"""

    # ========================================================
    # RECOVERY PROMPT TEMPLATE
    # ========================================================

    RECOVERY_PROMPT_TEMPLATE = """
You are performing a SECOND, targeted inspection of a
student's handwritten answer sheet.

The initial extraction could not reliably resolve some
questions.

TARGET QUESTIONS:

{target_questions}

Inspect this image very carefully.

For each target question that is actually visible in this
region, identify:

- the question number
- the student's answer
- confidence

Return ONLY valid JSON:

{
  "questions": [
    {
      "question_no": 1,
      "answer": "A",
      "confidence": 0.90
    }
  ]
}

Rules:

1. ONLY return question numbers from the target list.
2. Do not return other questions.
3. Do not invent a target question.
4. Do not infer the answer from surrounding questions.
5. Use physical position to associate the answer with the
   question number.
6. For MCQs use A, B, C, or D when clearly visible.
7. If the question number is visible but the answer cannot
   be read, use "[UNCLEAR]".
8. If no target question is visible, return:
   {{"questions": []}}
9. Do not assign marks.
10. Do not judge correctness.
11. Do not invent missing text.
12. confidence must be between 0 and 1.
13. Return no explanation outside JSON.

An empty result is better than a guessed answer.
"""

    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        vision_service: VisionService | None = None,
    ):
        self.vision_service = (
            vision_service
            or VisionService()
        )

    # ========================================================
    # CREATE TILES
    # ========================================================

    def create_tiles(
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

        if not image_path.exists():
            raise FileNotFoundError(
                f"Page image not found: "
                f"{image_path}"
            )

        image = Image.open(
            image_path
        )

        image.load()

        image = image.convert(
            "RGB"
        )

        width, height = image.size

        tile_width = (
            width // self.COLS
        )

        tile_height = (
            height // self.ROWS
        )

        overlap_x = int(
            tile_width
            * self.OVERLAP_X_RATIO
        )

        overlap_y = int(
            tile_height
            * self.OVERLAP_Y_RATIO
        )

        tiles = []

        for row in range(
            self.ROWS
        ):

            for col in range(
                self.COLS
            ):

                x1 = (
                    col * tile_width
                    - (
                        overlap_x
                        if col > 0
                        else 0
                    )
                )

                y1 = (
                    row * tile_height
                    - (
                        overlap_y
                        if row > 0
                        else 0
                    )
                )

                x2 = (
                    (col + 1)
                    * tile_width
                    + (
                        overlap_x
                        if col < self.COLS - 1
                        else 0
                    )
                )

                y2 = (
                    (row + 1)
                    * tile_height
                    + (
                        overlap_y
                        if row < self.ROWS - 1
                        else 0
                    )
                )

                x1 = max(
                    0,
                    x1,
                )

                y1 = max(
                    0,
                    y1,
                )

                x2 = min(
                    width,
                    x2,
                )

                y2 = min(
                    height,
                    y2,
                )

                tile = image.crop(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                )

                tile_number = (
                    row * self.COLS
                    + col
                    + 1
                )

                tile_name = (
                    f"tile_{tile_number:02d}"
                )

                tile_path = (
                    output_dir
                    / f"{tile_name}.jpg"
                )

                tile.save(
                    tile_path,
                    format="JPEG",
                    quality=95,
                    optimize=True,
                )

                tiles.append(
                    (
                        tile_name,
                        tile_path,
                    )
                )

        return tiles

    # ========================================================
    # NORMALIZE MODEL CANDIDATE
    # ========================================================

    def normalize_candidate(
        self,
        item: dict,
        source_tile: str,
    ) -> ExtractedAnswer:

        question_no = item.get(
            "question_no"
        )

        answer = str(
            item.get(
                "answer",
                "[UNCLEAR]",
            )
        ).strip()

        if not answer:
            answer = "[UNCLEAR]"

        confidence_raw = item.get(
            "confidence",
            0.0,
        )

        try:

            confidence = float(
                confidence_raw
            )

        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
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

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if question_no is None:

            status = (
                "REVIEW_REQUIRED"
            )

            warning = (
                "Question number could not "
                "be determined."
            )

        elif answer == "[UNCLEAR]":

            status = (
                "REVIEW_REQUIRED"
            )

            warning = (
                "Answer could not be "
                "read reliably."
            )

        elif confidence < self.MIN_CONFIDENCE:

            status = (
                "REVIEW_REQUIRED"
            )

            warning = (
                "Low-confidence extraction."
            )

        else:

            status = "EXTRACTED"

            warning = None

        return ExtractedAnswer(
            question_no=question_no,
            answer=answer,
            confidence=confidence,
            source_tile=source_tile,
            status=status,
            warning=warning,
        )

    # ========================================================
    # IGNORE EMPTY NULL RESULTS
    # ========================================================

    @staticmethod
    def is_empty_unresolved(
        candidate: ExtractedAnswer,
    ) -> bool:
        """
        A tile may simply contain no recognizable question.

        Example:
            question_no=None
            answer="[UNCLEAR]"

        This should NOT become a review item by itself.
        """

        return (
            candidate.question_no is None
            and candidate.answer == "[UNCLEAR]"
        )

    # ========================================================
    # MERGE CANDIDATES
    # ========================================================

    def merge_candidates(
        self,
        candidates: list[ExtractedAnswer],
    ) -> tuple[
        list[ExtractedAnswer],
        list[ExtractedAnswer],
        list[str],
    ]:
        """
        Merge all numbered candidates.

        Same question:
            - same answer -> strongest candidate
            - conflicting answers -> REVIEW_REQUIRED

        Null + unclear candidates are ignored.
        """

        grouped: dict[
            int,
            list[ExtractedAnswer],
        ] = {}

        unresolved = []

        # ----------------------------------------------------
        # Group
        # ----------------------------------------------------

        for candidate in candidates:

            if candidate.question_no is None:

                if not self.is_empty_unresolved(
                    candidate
                ):
                    unresolved.append(
                        candidate
                    )

                continue

            grouped.setdefault(
                candidate.question_no,
                [],
            ).append(
                candidate
            )

        merged = []

        warnings = []

        # ----------------------------------------------------
        # Merge each question
        # ----------------------------------------------------

        for question_no, items in grouped.items():

            # Prefer readable answer, then confidence.
            items.sort(
                key=lambda item: (
                    item.answer != "[UNCLEAR]",
                    item.confidence,
                ),
                reverse=True,
            )

            best = items[0]

            # ------------------------------------------------
            # All readable answers
            # ------------------------------------------------

            readable_answers = {
                item.answer
                for item in items
                if item.answer != "[UNCLEAR]"
            }

            # ------------------------------------------------
            # Conflict
            # ------------------------------------------------

            if len(
                readable_answers
            ) > 1:

                best.status = (
                    "REVIEW_REQUIRED"
                )

                values = ", ".join(
                    sorted(
                        readable_answers
                    )
                )

                best.warning = (
                    "Conflicting answers detected "
                    f"across tiles: {values}"
                )

                warnings.append(
                    f"Q{question_no}: conflicting "
                    "answers detected."
                )

            # ------------------------------------------------
            # Unclear
            # ------------------------------------------------

            elif best.answer == "[UNCLEAR]":

                best.status = (
                    "REVIEW_REQUIRED"
                )

                best.warning = (
                    "Answer could not be "
                    "read reliably."
                )

            # ------------------------------------------------
            # Low confidence
            # ------------------------------------------------

            elif (
                best.confidence
                < self.MIN_CONFIDENCE
            ):

                best.status = (
                    "REVIEW_REQUIRED"
                )

                best.warning = (
                    "Low-confidence extraction."
                )

            # ------------------------------------------------
            # Otherwise
            # ------------------------------------------------

            else:

                best.status = (
                    "EXTRACTED"
                )

                best.warning = None

            merged.append(
                best
            )

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        merged.sort(
            key=lambda item: (
                item.question_no
                if item.question_no is not None
                else 10**9
            )
        )

        return (
            merged,
            unresolved,
            warnings,
        )

    # ========================================================
    # FIND MISSING QUESTIONS
    # ========================================================

    @staticmethod
    def find_missing_questions(
        questions: list[ExtractedAnswer],
        expected_question_numbers: list[int],
    ) -> list[int]:

        detected = {
            item.question_no
            for item in questions
            if item.question_no is not None
        }

        missing = [
            number
            for number in expected_question_numbers
            if number not in detected
        ]

        return sorted(
            set(missing)
        )

    # ========================================================
    # FIND QUESTIONS NEEDING RECOVERY
    # ========================================================

    @staticmethod
    def find_recovery_targets(
        questions: list[ExtractedAnswer],
        missing_questions: list[int],
    ) -> list[int]:

        targets = set(
            missing_questions
        )

        for item in questions:

            if item.question_no is None:
                continue

            if (
                item.status
                == "REVIEW_REQUIRED"
            ):

                targets.add(
                    item.question_no
                )

        return sorted(
            targets
        )

    # ========================================================
    # RECOVERY PROMPT
    # ========================================================
    def build_recovery_prompt(
        self,
        target_questions: list[int],
    ) -> str:
        """
        Build the targeted recovery prompt.

        We intentionally use string replacement instead of
        str.format() because the prompt itself contains JSON
        braces such as:

            {
            "questions": []
            }

        str.format() would interpret those braces as template
        placeholders and raise KeyError.
        """

        formatted = ", ".join(
            str(number)
            for number in target_questions
        )

        return self.RECOVERY_PROMPT_TEMPLATE.replace(
            "{target_questions}",
            formatted,
        )
    

    # ========================================================
    # TARGETED RECOVERY
    # ========================================================

    def run_recovery(
        self,
        tiles: list[tuple[str, Path]],
        target_questions: list[int],
    ) -> list[ExtractedAnswer]:

        if not target_questions:
            return []

        prompt = (
            self.build_recovery_prompt(
                target_questions
            )
        )

        recovery_candidates = []

        print("\n")
        print("=" * 70)
        print(
            "TARGETED RECOVERY"
        )
        print("=" * 70)

        print(
            "Target questions:",
            target_questions,
        )

        # ----------------------------------------------------
        # One recovery request per tile.
        #
        # We intentionally do NOT make one request per
        # question. A single recovery call can look for
        # all missing/uncertain questions in that tile.
        # ----------------------------------------------------

        for tile_name, tile_path in tiles:

            print("\n")
            print(
                f"RECOVERY TILE: "
                f"{tile_name}"
            )

            result = (
                self.vision_service
                .analyze_image(
                    image_path=tile_path,
                    prompt=prompt,
                )
            )

            parsed = result[
                "parsed"
            ]

            questions = parsed.get(
                "questions",
                [],
            )

            print(
                json.dumps(
                    parsed,
                    indent=2,
                    ensure_ascii=False,
                )
            )

            for item in questions:

                candidate = (
                    self.normalize_candidate(
                        item=item,
                        source_tile=(
                            f"{tile_name}:recovery"
                        ),
                    )
                )

                # Only keep actual target numbers.
                if (
                    candidate.question_no
                    in target_questions
                ):

                    recovery_candidates.append(
                        candidate
                    )

        return recovery_candidates

    # ========================================================
    # FINALIZE RESULTS
    # ========================================================

    def finalize(
        self,
        all_candidates: list[ExtractedAnswer],
        expected_question_numbers: list[int],
    ) -> tuple[
        list[ExtractedAnswer],
        list[ExtractedAnswer],
        list[int],
        list[str],
    ]:

        (
            merged,
            unresolved,
            warnings,
        ) = self.merge_candidates(
            all_candidates
        )

        missing = (
            self.find_missing_questions(
                merged,
                expected_question_numbers,
            )
        )

        # ----------------------------------------------------
        # Mark expected questions that remain missing
        # ----------------------------------------------------

        if missing:

            warnings.append(
                "Missing expected question(s): "
                + ", ".join(
                    f"Q{number}"
                    for number in missing
                )
            )

        # ----------------------------------------------------
        # Final review flag
        # ----------------------------------------------------

        return (
            merged,
            unresolved,
            missing,
            warnings,
        )

    # ========================================================
    # EXTRACT PAGE
    # ========================================================

    def extract_page(
        self,
        image_path: str | Path,
        page_no: int,
        tile_output_dir: str | Path,
        expected_question_numbers: list[int] | None = None,
        enable_recovery: bool = True,
    ) -> PageExtractionResult:

        image_path = Path(
            image_path
        )

        tile_output_dir = Path(
            tile_output_dir
        )

        # ----------------------------------------------------
        # Normalize expected list
        # ----------------------------------------------------

        if expected_question_numbers is None:

            expected_question_numbers = []

        expected_question_numbers = sorted(
            {
                int(number)
                for number in expected_question_numbers
            }
        )

        # ----------------------------------------------------
        # Create tiles
        # ----------------------------------------------------

        tiles = self.create_tiles(
            image_path=image_path,
            output_dir=tile_output_dir,
        )

        # ----------------------------------------------------
        # Initial extraction
        # ----------------------------------------------------

        all_candidates = []

        processed_tiles = 0

        print("\n")
        print("=" * 70)
        print(
            "INITIAL ANSWER EXTRACTION"
        )
        print("=" * 70)

        for tile_name, tile_path in tiles:

            print("\n")
            print(
                f"PROCESSING: "
                f"{tile_name}"
            )

            result = (
                self.vision_service
                .analyze_image(
                    image_path=tile_path,
                    prompt=self.EXTRACTION_PROMPT,
                )
            )

            parsed = result[
                "parsed"
            ]

            questions = parsed.get(
                "questions",
                [],
            )

            print(
                json.dumps(
                    parsed,
                    indent=2,
                    ensure_ascii=False,
                )
            )

            for item in questions:

                candidate = (
                    self.normalize_candidate(
                        item=item,
                        source_tile=tile_name,
                    )
                )

                all_candidates.append(
                    candidate
                )

            processed_tiles += 1

        # ----------------------------------------------------
        # First merge
        # ----------------------------------------------------

        (
            initial_merged,
            initial_unresolved,
            initial_warnings,
        ) = self.merge_candidates(
            all_candidates
        )

        # ----------------------------------------------------
        # Determine recovery targets
        # ----------------------------------------------------

        missing_questions = (
            self.find_missing_questions(
                initial_merged,
                expected_question_numbers,
            )
        )

        recovery_targets = (
            self.find_recovery_targets(
                initial_merged,
                missing_questions,
            )
        )

        print("\n")
        print("=" * 70)
        print(
            "INITIAL ANALYSIS"
        )
        print("=" * 70)

        print(
            "Detected:",
            [
                item.question_no
                for item in initial_merged
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
        # Recovery
        # ----------------------------------------------------

        recovery_candidates = []

        recovery_attempted = False

        recovery_tiles = 0

        if (
            enable_recovery
            and recovery_targets
        ):

            recovery_attempted = True

            recovery_candidates = (
                self.run_recovery(
                    tiles=tiles,
                    target_questions=recovery_targets,
                )
            )

            recovery_tiles = len(
                tiles
            )

        # ----------------------------------------------------
        # Combine initial + recovery
        # ----------------------------------------------------

        combined_candidates = (
            all_candidates
            + recovery_candidates
        )

        # ----------------------------------------------------
        # Final merge
        # ----------------------------------------------------

        (
            final_questions,
            final_unresolved,
            final_missing,
            final_warnings,
        ) = self.finalize(
            all_candidates=combined_candidates,
            expected_question_numbers=expected_question_numbers,
        )

        # ----------------------------------------------------
        # Combine warnings
        # ----------------------------------------------------

        warnings = []

        for warning in (
            initial_warnings
            + final_warnings
        ):

            if warning not in warnings:

                warnings.append(
                    warning
                )

        # ----------------------------------------------------
        # Recovery warning
        # ----------------------------------------------------

        if recovery_attempted:

            warnings.append(
                "Targeted recovery was attempted "
                "for missing or uncertain questions."
            )

        # ----------------------------------------------------
        # Remaining unresolved
        # ----------------------------------------------------

        if final_unresolved:

            warnings.append(
                f"{len(final_unresolved)} "
                "unresolved extraction item(s)."
            )

        # ----------------------------------------------------
        # Review flag
        # ----------------------------------------------------

        review_required = bool(
            final_missing
            or final_unresolved
            or any(
                item.status
                == "REVIEW_REQUIRED"
                for item in final_questions
            )
        )

        # ----------------------------------------------------
        # Create result
        # ----------------------------------------------------

        return PageExtractionResult(
            page_no=page_no,
            questions=final_questions,
            unresolved_items=final_unresolved,
            missing_questions=final_missing,
            review_required=review_required,
            warnings=warnings,
            processed_tiles=processed_tiles,
            recovery_attempted=recovery_attempted,
            recovery_tiles=recovery_tiles,
        )