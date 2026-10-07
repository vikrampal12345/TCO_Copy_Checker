import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from app.core.model_provider import get_model_provider


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "pages"
    / "Student_2"
    / "vision_tiles"
    / "bottom_left.jpg"
)


# ============================================================
# IMAGE
# ============================================================

def image_to_data_url(image_path: Path) -> str:

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(image_path)
    image.load()
    image = image.convert("RGB")

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# TEST
# ============================================================

def test_structured_vision_v2():

    print("\n")
    print("=" * 70)
    print("TCO COPY CHECKER - STRUCTURED VISION V2")
    print("=" * 70)

    provider = get_model_provider()

    client = provider.get_client()
    model = provider.get_vision_model()

    print(f"\nProvider: {provider.get_provider_name()}")
    print(f"Vision model: {model}")
    print(f"Image: {IMAGE_PATH}")

    image_url = image_to_data_url(
        IMAGE_PATH
    )

    prompt = """
You are reading a student's handwritten answer sheet.

This image is ONE region of the page.

Read the questions from TOP TO BOTTOM.

IMPORTANT:
The previous visual inspection suggests that this region
contains questions 7 through 20.

Your job is to map EACH question number to the student's
visible answer.

Do NOT combine answers.

For example, if the page visually shows:

7 A
8 C
9 C

you MUST return:

7 -> A
8 -> C
9 -> C

Return ONLY valid JSON.

Use exactly this format:

{
  "questions": [
    {
      "question_no": 7,
      "answer": "A",
      "confidence": 0.95
    },
    {
      "question_no": 8,
      "answer": "C",
      "confidence": 0.95
    }
  ]
}

Rules:

1. One JSON object per question.
2. question_no must be the actual visible question number.
3. answer must contain the student's visible answer.
4. For MCQ answers, use A, B, C, D when clearly visible.
5. Never combine multiple answers into one object.
6. Never use question_no=null when a visible question number exists.
7. Preserve top-to-bottom order.
8. If the question number is genuinely unreadable, use null.
9. If the answer is genuinely unreadable, use "[UNCLEAR]".
10. Do NOT assign marks.
11. Do NOT judge correctness.
12. Do NOT invent answers.
13. Do NOT include explanations outside JSON.
14. confidence must be between 0 and 1.

Before generating the final JSON, mentally inspect the image
from top to bottom and establish the question number -> answer
mapping based on their spatial positions.
"""

    # ========================================================
    # MODEL REQUEST
    # ========================================================

    try:

        response = (
            client
            .chat
            .completions
            .create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise handwritten "
                            "answer-sheet extraction system. "
                            "Return JSON only."
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
                                    "url": image_url
                                },
                            },
                        ],
                    },
                ],
            )
        )

    except Exception as exc:

        pytest.fail(
            f"Vision request failed:\n{exc}"
        )

    raw = (
        response
        .choices[0]
        .message
        .content
    )

    print("\n")
    print("=" * 70)
    print("RAW MODEL OUTPUT")
    print("=" * 70)

    print(raw)

    assert raw is not None
    assert raw.strip()

    # ========================================================
    # PARSE JSON
    # ========================================================

    cleaned = raw.strip()

    if cleaned.startswith("```"):

        cleaned = (
            cleaned
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    try:

        data = json.loads(cleaned)

    except json.JSONDecodeError as exc:

        pytest.fail(
            "Model did not return valid JSON.\n\n"
            f"RAW OUTPUT:\n{raw}\n\n"
            f"JSON ERROR:\n{exc}"
        )

    # ========================================================
    # VALIDATE
    # ========================================================

    assert isinstance(data, dict)

    assert "questions" in data

    assert isinstance(
        data["questions"],
        list,
    )

    seen_numbers = set()

    for item in data["questions"]:

        assert isinstance(
            item,
            dict,
        )

        assert "question_no" in item
        assert "answer" in item
        assert "confidence" in item

        question_no = item["question_no"]

        answer = item["answer"]

        confidence = item["confidence"]

        if question_no is not None:

            assert isinstance(
                question_no,
                int,
            )

            assert question_no not in seen_numbers

            seen_numbers.add(question_no)

        assert isinstance(
            answer,
            str,
        )

        assert isinstance(
            confidence,
            (int, float),
        )

        assert 0 <= confidence <= 1

    # ========================================================
    # PRINT STRUCTURED RESULT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STRUCTURED RESULT")
    print("=" * 70)

    print(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n")
    print("=" * 70)
    print("STRUCTURED VISION V2 PASSED")
    print("=" * 70)