from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from app.schemas.evaluation_result import EvaluationResult


class CheckedCopyRenderer:
    """
    Creates a checked-copy PDF.

    Current version:
        Original handwritten page
        +
        grading panel on the right side.

    Exact marks beside handwritten answers will be added later when
    answer bounding-box coordinates are available from extraction.
    """

    PANEL_WIDTH = 760
    MARGIN = 30

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

        questions_by_page = {
            int(page["page_number"]): page.get(
                "questions",
                [],
            )
            for page in evaluation_result.pages
        }

        rendered_pages: list[Image.Image] = []

        try:

            for page_no, original in page_images:

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

                draw.text(
                    (x, y),
                    "AI CHECKED COPY",
                    fill="black",
                    font=font,
                )

                y += 35

                draw.text(
                    (x, y),
                    f"Page {page_no}",
                    fill="black",
                    font=small_font,
                )

                y += 35

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

                draw.text(
                    (x, y),
                    f"Overall: {obtained:.2f} / {total:.2f}",
                    fill="black",
                    font=small_font,
                )

                y += 30

                draw.text(
                    (x, y),
                    f"Percentage: {percentage:.2f}%",
                    fill="black",
                    font=small_font,
                )

                y += 45

                questions = questions_by_page.get(
                    page_no,
                    [],
                )

                for question in questions:

                    if y >= canvas.height - 100:
                        break

                    question_no = question.get(
                        "question_no",
                        "?",
                    )

                    marks = (
                        question.get(
                            "marks_awarded"
                        )
                        or 0.0
                    )

                    max_marks = (
                        question.get(
                            "max_marks"
                        )
                        or 0.0
                    )

                    reason = str(
                        question.get(
                            "reason"
                        )
                        or ""
                    ).strip()

                    draw.text(
                        (x, y),
                        (
                            f"Q{question_no}: "
                            f"{marks:g} / {max_marks:g}"
                        ),
                        fill="black",
                        font=font,
                    )

                    y += 30

                    for line in self._wrap_text(
                        reason,
                        width=42,
                    ):

                        if y >= canvas.height - 70:
                            break

                        draw.text(
                            (x + 20, y),
                            line,
                            fill="black",
                            font=small_font,
                        )

                        y += 25

                    y += 12

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
