from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from app.schemas.evaluation_result import EvaluationResult


class CheckedCopyRenderer:
    """
    Creates a teacher-style checked-copy PDF.

    Current version:
        Original handwritten page
        +
        teacher-style grading panel on the right.

    The panel shows:
        - Question number
        - Red check/cross
        - Marks awarded
        - Short grading reason
        - Overall result only on the last graded page

    Exact marks beside the handwritten answer are intentionally
    deferred until extraction provides answer bounding-box coordinates.
    """

    PANEL_WIDTH = 700
    MARGIN = 30

    CHECK_COLOR = "red"
    TEXT_COLOR = "black"

    def render(
        self,
        *,
        page_images: Iterable[tuple[int, Image.Image]],
        evaluation_result: EvaluationResult,
        output_dir: str | Path,
        file_name: str = "checked_copy.pdf",
    ) -> Path:

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pages_input = list(page_images)

        questions_by_page = {
            int(page["page_number"]): page.get(
                "questions",
                [],
            )
            for page in evaluation_result.pages
        }

        graded_page_numbers = sorted(
            page_number
            for page_number, questions
            in questions_by_page.items()
            if questions
        )

        last_graded_page = (
            graded_page_numbers[-1]
            if graded_page_numbers
            else None
        )

        rendered_pages: list[Image.Image] = []

        try:

            for page_no, original in pages_input:

                base = original.convert("RGB")

                canvas_height = max(
                    base.height,
                    900,
                )

                canvas = Image.new(
                    "RGB",
                    (
                        base.width + self.PANEL_WIDTH,
                        canvas_height,
                    ),
                    "white",
                )

                canvas.paste(
                    base,
                    (0, 0),
                )

                draw = ImageDraw.Draw(canvas)

                font = ImageFont.load_default()

                small_font = ImageFont.load_default()

                panel_left = base.width

                # ------------------------------------------------
                # Panel separator
                # ------------------------------------------------

                draw.line(
                    (
                        panel_left,
                        0,
                        panel_left,
                        canvas.height,
                    ),
                    fill="black",
                    width=3,
                )

                x = panel_left + self.MARGIN
                y = self.MARGIN

                # ------------------------------------------------
                # Header
                # ------------------------------------------------

                draw.text(
                    (x, y),
                    "TEACHER CHECKED COPY",
                    fill=self.TEXT_COLOR,
                    font=font,
                )

                y += 35

                draw.text(
                    (x, y),
                    f"Page {page_no}",
                    fill=self.TEXT_COLOR,
                    font=small_font,
                )

                y += 40

                questions = questions_by_page.get(
                    page_no,
                    [],
                )

                # ------------------------------------------------
                # Page information
                # ------------------------------------------------

                page_marks_awarded = sum(
                    (
                        float(
                            question.get(
                                "marks_awarded",
                                0.0,
                            )
                            or 0.0
                        )
                    )
                    for question in questions
                )

                page_max_marks = sum(
                    (
                        float(
                            question.get(
                                "max_marks",
                                0.0,
                            )
                            or 0.0
                        )
                    )
                    for question in questions
                )

                if questions:

                    draw.text(
                        (x, y),
                        (
                            f"Page Marks: "
                            f"{page_marks_awarded:g} / "
                            f"{page_max_marks:g}"
                        ),
                        fill=self.TEXT_COLOR,
                        font=small_font,
                    )

                    y += 35

                    draw.text(
                        (x, y),
                        "QUESTION-WISE MARKING",
                        fill=self.TEXT_COLOR,
                        font=font,
                    )

                    y += 45

                else:

                    draw.text(
                        (x, y),
                        "No graded questions on this page.",
                        fill=self.TEXT_COLOR,
                        font=small_font,
                    )

                    y += 40

                # ------------------------------------------------
                # Question-wise grading
                # ------------------------------------------------

                for question in questions:

                    question_no = question.get(
                        "question_no",
                        "?",
                    )

                    marks = float(
                        question.get(
                            "marks_awarded",
                            0.0,
                        )
                        or 0.0
                    )

                    max_marks = float(
                        question.get(
                            "max_marks",
                            0.0,
                        )
                        or 0.0
                    )

                    reason = str(
                        question.get(
                            "reason",
                        )
                        or ""
                    ).strip()

                    # Stop before the bottom of the page.
                    if y >= canvas.height - 130:
                        break

                    # --------------------------------------------
                    # Question row
                    # --------------------------------------------

                    question_text = (
                        f"Q{question_no}: "
                        f"{marks:g} / {max_marks:g}"
                    )

                    draw.text(
                        (x, y),
                        question_text,
                        fill=self.TEXT_COLOR,
                        font=font,
                    )

                    # --------------------------------------------
                    # Red teacher-style check / cross
                    # --------------------------------------------

                    icon_x = x + 150
                    icon_y = y + 4

                    if marks > 0:

                        # Check mark: two connected strokes.
                        draw.line(
                            (
                                icon_x,
                                icon_y + 7,
                                icon_x + 6,
                                icon_y + 14,
                            ),
                            fill=self.CHECK_COLOR,
                            width=3,
                        )

                        draw.line(
                            (
                                icon_x + 6,
                                icon_y + 14,
                                icon_x + 18,
                                icon_y,
                            ),
                            fill=self.CHECK_COLOR,
                            width=3,
                        )

                    else:

                        # Cross mark for zero.
                        draw.line(
                            (
                                icon_x,
                                icon_y,
                                icon_x + 14,
                                icon_y + 14,
                            ),
                            fill=self.CHECK_COLOR,
                            width=3,
                        )

                        draw.line(
                            (
                                icon_x + 14,
                                icon_y,
                                icon_x,
                                icon_y + 14,
                            ),
                            fill=self.CHECK_COLOR,
                            width=3,
                        )

                    # --------------------------------------------
                    # Reason
                    # --------------------------------------------

                    y += 28

                    for line in self._wrap_text(
                        reason,
                        width=55,
                    ):

                        if y >= canvas.height - 70:
                            break

                        draw.text(
                            (x + 18, y),
                            line,
                            fill=self.TEXT_COLOR,
                            font=small_font,
                        )

                        y += 22

                    y += 18

                # ------------------------------------------------
                # Overall summary ONLY on last graded page
                # ------------------------------------------------

                if (
                    last_graded_page is not None
                    and page_no == last_graded_page
                ):

                    obtained = (
                        evaluation_result.obtained_marks
                        or 0.0
                    )

                    total = (
                        evaluation_result.total_marks
                        or 0.0
                    )

                    percentage = (
                        evaluation_result.percentage
                        or 0.0
                    )

                    summary_y = max(
                        y + 10,
                        canvas.height - 115,
                    )

                    draw.line(
                        (
                            x,
                            summary_y - 8,
                            canvas.width - self.MARGIN,
                            summary_y - 8,
                        ),
                        fill="black",
                        width=1,
                    )

                    draw.text(
                        (x, summary_y),
                        (
                            f"OVERALL: "
                            f"{obtained:.2f} / "
                            f"{total:.2f}"
                        ),
                        fill=self.TEXT_COLOR,
                        font=font,
                    )

                    draw.text(
                        (x, summary_y + 28),
                        (
                            f"PERCENTAGE: "
                            f"{percentage:.2f}%"
                        ),
                        fill=self.TEXT_COLOR,
                        font=small_font,
                    )

                rendered_pages.append(canvas)

            if not rendered_pages:
                raise ValueError(
                    "No pages available for checked-copy rendering."
                )

            output_path = (
                output_dir / file_name
            )

            first = rendered_pages[0]

            rest = rendered_pages[1:]

            first.save(
                output_path,
                "PDF",
                resolution=150.0,
                save_all=True,
                append_images=rest,
            )

            return output_path

        finally:

            for page in rendered_pages:
                try:
                    page.close()
                except Exception:
                    pass

    @staticmethod
    def _wrap_text(
        text: str,
        *,
        width: int,
    ) -> list[str]:

        if not text:
            return [""]

        words = text.split()

        lines: list[str] = []

        current = words[0]

        for word in words[1:]:

            candidate = (
                f"{current} {word}"
            )

            if len(candidate) > width:

                lines.append(current)

                current = word

            else:

                current = candidate

        lines.append(current)

        return lines
