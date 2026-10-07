import base64
import io
import json
from pathlib import Path

from PIL import Image
import pytest

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


def test_structured_vision():

    print("\n")
    print("=" * 70)
    print("TCO COPY CHECKER - STRUCTURED VISION TEST")
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
Analyze this cropped region of a student's answer sheet.

Extract ONLY questions and visible student answers.

Return ONLY valid JSON.

Required structure:

{
  "questions": [
    {
      "question_no": 1,
      "answer": "A",
      "complete": true,
      "formula": null,
      "diagram": null,
      "confidence": 0.8
    }
  ]
}

Rules:

1. Create one object for EACH visible question.
2. Do not combine multiple questions into one object.
3. Do not invent question numbers.
4. Do not invent answers.
5. If the question number cannot be read, use null.
6. If the answer cannot be read, use "[UNCLEAR]".
7. Do not assign marks.
8. Do not judge correctness.
9. confidence must be between 0 and 1.
10. Do not add any explanation outside the JSON.
11. Ignore the words "Question" and "QCM" unless they are actually part
    of a question number.
"""

    try:

        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise document vision "
                        "extractor. Return valid JSON only."
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

    except Exception as exc:

        pytest.fail(
            f"Vision request failed:\n{exc}"
        )

    raw_result = (
        response
        .choices[0]
        .message
        .content
    )

    print("\n")
    print("=" * 70)
    print("RAW MODEL OUTPUT")
    print("=" * 70)

    print(raw_result)

    assert raw_result is not None
    assert raw_result.strip()

    # --------------------------------------------------------
    # Try JSON parsing
    # --------------------------------------------------------

    try:

        data = json.loads(
            raw_result
        )

    except json.JSONDecodeError:

        # Sometimes local models wrap JSON in ```json
        # Remove code fences and retry.

        cleaned = (
            raw_result
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        try:

            data = json.loads(
                cleaned
            )

        except json.JSONDecodeError as exc:

            pytest.fail(
                "Model did not return valid JSON.\n\n"
                f"Raw output:\n{raw_result}\n\n"
                f"JSON error:\n{exc}"
            )

    # --------------------------------------------------------
    # Validate structure
    # --------------------------------------------------------

    assert isinstance(data, dict)

    assert "questions" in data

    assert isinstance(
        data["questions"],
        list,
    )

    print("\n")
    print("=" * 70)
    print("PARSED JSON")
    print("=" * 70)

    print(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # Validate every question
    # --------------------------------------------------------

    for question in data["questions"]:

        assert isinstance(
            question,
            dict,
        )

        assert "question_no" in question
        assert "answer" in question
        assert "complete" in question
        assert "formula" in question
        assert "diagram" in question
        assert "confidence" in question

        confidence = question[
            "confidence"
        ]

        assert isinstance(
            confidence,
            (int, float),
        )

        assert 0 <= confidence <= 1

    print("\n")
    print("=" * 70)
    print("STRUCTURED VISION TEST PASSED")
    print("=" * 70)