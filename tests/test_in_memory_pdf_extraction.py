from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path

import pymupdf
from openai import OpenAI
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = Path("data/test_copies/Student_23.pdf")

MODEL_BASE_URL = os.getenv(
    "MODEL_BASE_URL",
    "http://127.0.0.1:1234/v1",
)

MODEL_API_KEY = os.getenv(
    "MODEL_API_KEY",
    "lm-studio",
)

VISION_MODEL = os.getenv(
    "VISION_MODEL_NAME",
    "qwen2.5-vl-3b-instruct",
)

# Rendering quality.
# 200 DPI keeps memory usage reasonable while providing
# enough resolution for handwritten content testing.
DPI = 200

# JPEG quality used only for the in-memory image sent to the model.
# The JPEG is NEVER written to disk.
JPEG_QUALITY = 90

# Maximum image dimension sent to the vision model.
MAX_IMAGE_DIMENSION = 1800


# ============================================================
# CLIENT
# ============================================================

client = OpenAI(
    base_url=MODEL_BASE_URL,
    api_key=MODEL_API_KEY,
)


# ============================================================
# PDF PAGE -> IN-MEMORY IMAGE
# ============================================================

def render_page_to_memory(
    page: pymupdf.Page,
    dpi: int = DPI,
) -> Image.Image:
    """
    Render one PDF page directly into RAM.

    No PNG/JPG file is created on disk.
    """

    zoom = dpi / 72.0

    matrix = pymupdf.Matrix(
        zoom,
        zoom,
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    # Pixmap bytes are kept in memory.
    image_bytes = pixmap.tobytes("png")

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    # Free pixmap as early as possible.
    del pixmap
    del image_bytes

    # Keep image reasonably sized for local vision model.
    image.thumbnail(
        (
            MAX_IMAGE_DIMENSION,
            MAX_IMAGE_DIMENSION,
        ),
        Image.Resampling.LANCZOS,
    )

    return image


# ============================================================
# IMAGE -> BASE64 DATA URI
# ============================================================

def image_to_data_uri(
    image: Image.Image,
) -> str:
    """
    Convert an in-memory PIL image to a base64 data URI.

    Again, nothing is saved to disk.
    """

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# ============================================================
# VISION EXTRACTION
# ============================================================

def extract_page_content(
    image: Image.Image,
    page_number: int,
) -> tuple[str, dict | None]:
    """
    Send one page image to the local vision model.

    The model is instructed to extract ALL visible content,
    not only multiple-choice answers.
    """

    image_data_uri = image_to_data_uri(image)

    prompt = f"""
You are a document and handwriting extraction system.

This is page {page_number} of a student's handwritten answer sheet.

Extract ALL visible academic content from this page.

IMPORTANT RULES:

1. Read the page visually.
2. Do NOT assume that questions are placed in any fixed location.
3. A question number may appear:
   - on the left side
   - above the answer
   - on the same line
   - inside a circle
   - underlined
   - near the answer
   - in another natural position
4. Detect question numbers wherever they appear.
5. Preserve the actual order of content on the page.
6. Extract handwritten answers as faithfully as possible.
7. Include:
   - MCQ answers
   - true/false
   - fill in the blanks
   - short answers
   - long/subjective answers
   - numerical calculations
   - formulas
   - equations
   - bullet points
   - steps
   - headings
   - tables
   - labels
   - visible diagram text
8. Do NOT invent missing text.
9. If something cannot be confidently read, write [UNCLEAR].
10. Do not silently correct spelling, grammar, mathematics, or handwriting.
11. Keep formulas/equations in readable text form.
12. If a diagram is visible but its exact content cannot be represented as text,
    describe only what is visibly identifiable.
13. If a part of the page is blank, do not invent content.
14. Return ONLY valid JSON.
15. Do not wrap the JSON in markdown code fences.

Use exactly this structure:

{{
  "page_number": {page_number},
  "page_type": "handwritten_answer_sheet",
  "content": [
    {{
      "question_no": 1,
      "content_type": "answer",
      "text": "actual extracted content",
      "confidence": 0.0
    }}
  ],
  "other_content": [
    {{
      "content_type": "heading|table|diagram|formula|note|other",
      "text": "actual visible content",
      "confidence": 0.0
    }}
  ],
  "review_required": false,
  "warnings": []
}}
"""

    response = client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise handwriting and document "
                    "extraction model. Never invent content."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_uri,
                        },
                    },
                ],
            },
        ],
    )

    raw_text = response.choices[0].message.content or ""

    raw_text = raw_text.strip()

    # Remove markdown fences if the model still returns them.
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw_text = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = None

    return raw_text, parsed


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_page_result(
    page_number: int,
    raw_text: str,
    parsed: dict | None,
) -> None:
    print()
    print("=" * 90)
    print(f"PAGE {page_number} - EXTRACTION RESULT")
    print("=" * 90)

    if parsed is not None:
        print(
            json.dumps(
                parsed,
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print("MODEL RETURNED NON-JSON OUTPUT:")
        print("-" * 90)
        print(raw_text)

    print("=" * 90)


# ============================================================
# MAIN TEST
# ============================================================

def test_student_23_in_memory_extraction() -> None:
    print()
    print("=" * 90)
    print("TCO COPY CHECKER")
    print("IN-MEMORY PDF -> PAGE -> VISION EXTRACTION TEST")
    print("=" * 90)

    # --------------------------------------------------------
    # Validate PDF
    # --------------------------------------------------------

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH.resolve()}"
        )

    if not PDF_PATH.is_file():
        raise ValueError(
            f"Path is not a file: {PDF_PATH.resolve()}"
        )

    print()
    print(f"PDF: {PDF_PATH.resolve()}")
    print(f"Vision model: {VISION_MODEL}")
    print(f"LM Studio: {MODEL_BASE_URL}")
    print(f"DPI: {DPI}")

    # --------------------------------------------------------
    # Open PDF
    # --------------------------------------------------------

    document = pymupdf.open(PDF_PATH)

    try:
        total_pages = len(document)

        print()
        print(f"Total pages: {total_pages}")
        print()
        print("PROCESSING FLOW:")
        print("PDF -> Page in RAM -> Vision Model -> Result -> Next Page")
        print("No page image is permanently written to disk.")

        # ----------------------------------------------------
        # Process one page at a time
        # ----------------------------------------------------

        for page_index in range(total_pages):

            page_number = page_index + 1

            print()
            print("#" * 90)
            print(f"PROCESSING PAGE {page_number}/{total_pages}")
            print("#" * 90)

            # -----------------------------------------------
            # Load only current page
            # -----------------------------------------------

            page = document.load_page(
                page_index
            )

            print(
                f"Page {page_number}: loaded into memory."
            )

            # -----------------------------------------------
            # Render into RAM
            # -----------------------------------------------

            image = render_page_to_memory(
                page=page,
                dpi=DPI,
            )

            print(
                f"Page {page_number}: rendered in RAM."
            )

            print(
                f"Image size in RAM: "
                f"{image.width} x {image.height}"
            )

            # -----------------------------------------------
            # Send to vision model
            # -----------------------------------------------

            print(
                f"Page {page_number}: sending to "
                f"{VISION_MODEL}..."
            )

            raw_text, parsed = extract_page_content(
                image=image,
                page_number=page_number,
            )

            # -----------------------------------------------
            # Display extracted content
            # -----------------------------------------------

            display_page_result(
                page_number=page_number,
                raw_text=raw_text,
                parsed=parsed,
            )

            # -----------------------------------------------
            # Explicitly release current page image
            # -----------------------------------------------

            del image
            del page

            print(
                f"Page {page_number}: RAM image released."
            )

    finally:
        document.close()

    print()
    print("=" * 90)
    print("ALL PAGES PROCESSED")
    print("=" * 90)
    print("PDF document closed.")


if __name__ == "__main__":
    test_student_23_in_memory_extraction()