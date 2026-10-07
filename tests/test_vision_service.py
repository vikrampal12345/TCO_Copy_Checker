import base64
from pathlib import Path

import pytest
from openai import OpenAI

from app.core.model_provider import get_model_provider


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# First try the manually resized vision image.
# This is the image we created earlier.
IMAGE_PATH = Path(
    "data/pages/Student_2/page_001_vision.jpg"
)

# Fallback locations.
# This makes the test more robust if the page was rendered
# into a job-specific folder.
FALLBACK_IMAGE_PATHS = [
    Path("data/pages/Student_2/page_001.png"),
    Path("data/outputs/pages/Student_2/page_001.png"),
]


# ---------------------------------------------------------
# FIND IMAGE
# ---------------------------------------------------------

def find_image() -> Path:
    """
    Find the first valid test image.
    """

    if IMAGE_PATH.exists():
        return IMAGE_PATH

    for path in FALLBACK_IMAGE_PATHS:
        if path.exists():
            return path

    # Search recursively as a final fallback.
    candidates = [
        Path("data/pages").rglob("page_001_vision.jpg"),
        Path("data/pages").rglob("page_001.png"),
        Path("data/outputs/pages").rglob("page_001.png"),
    ]

    for iterator in candidates:
        for candidate in iterator:
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "No suitable test image was found.\n"
        "Checked:\n"
        f"  {IMAGE_PATH}\n"
        + "\n".join(
            f"  {path}"
            for path in FALLBACK_IMAGE_PATHS
        )
    )


# ---------------------------------------------------------
# IMAGE -> DATA URL
# ---------------------------------------------------------

def image_to_data_url(
    image_path: Path,
) -> str:
    """
    Convert an image file into a base64 data URL.

    IMPORTANT:
    MIME type is detected from the actual file extension.
    """

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    suffix = image_path.suffix.lower()

    mime_type = mime_types.get(suffix)

    if mime_type is None:
        raise ValueError(
            f"Unsupported image format: {suffix}"
        )

    image_bytes = image_path.read_bytes()

    if not image_bytes:
        raise ValueError(
            f"Image file is empty: {image_path}"
        )

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


# ---------------------------------------------------------
# MAIN TEST
# ---------------------------------------------------------

def test_real_vision_model():

    # -----------------------------------------------------
    # Find image
    # -----------------------------------------------------

    image_path = find_image()

    print("\n========================================")
    print("REAL VISION MODEL TEST")
    print("========================================")

    print(f"Image: {image_path}")
    print(
        f"Size: {image_path.stat().st_size / 1024:.2f} KB"
    )
    print(f"Format: {image_path.suffix.lower()}")

    # -----------------------------------------------------
    # Load provider
    # -----------------------------------------------------

    provider = get_model_provider()

    client = provider.get_client()
    vision_model = provider.get_vision_model()

    print(f"Provider: {provider.get_provider_name()}")
    print(f"Vision model: {vision_model}")

    # -----------------------------------------------------
    # Convert image
    # -----------------------------------------------------

    data_url = image_to_data_url(
        image_path
    )

    # -----------------------------------------------------
    # Vision prompt
    # -----------------------------------------------------

    prompt = """
You are checking a student's handwritten answer sheet.

Analyze ONLY what is visibly present in the image.

Return the result in this structure:

PAGE:
<number>

QUESTIONS:
- Question: <question number>
  Answer: <exact visible handwritten answer>
  Complete: <yes/no/unclear>
  Formula: <formula if visible, otherwise None>
  Diagram: <description if visible, otherwise None>
  Confidence: <0.0 to 1.0>

IMPORTANT RULES:

1. Do not invent text that is not visible.
2. Read handwritten content carefully.
3. Identify every question visible on the page.
4. Preserve the actual question number.
5. If handwriting is unclear, explicitly say:
   "Unclear"
6. If there is no answer, say:
   "No answer visible"
7. Do not award marks.
8. Do not decide whether the answer is correct.
9. Do not guess missing words.
10. Mention formulas separately.
11. Mention diagrams separately.
12. Confidence must reflect how clearly the handwriting can be read.
13. Return plain text only.
"""

    # -----------------------------------------------------
    # API request
    # -----------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=vision_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful handwritten "
                        "answer-sheet vision analyzer."
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
                                "url": data_url,
                            },
                        },
                    ],
                },
            ],
        )

    except Exception as exc:

        print("\nVISION API ERROR")
        print("----------------------------------------")
        print(type(exc).__name__)
        print(str(exc))

        pytest.fail(
            f"Vision model request failed: {exc}"
        )

    # -----------------------------------------------------
    # Extract output
    # -----------------------------------------------------

    result = response.choices[0].message.content

    print("\n========================================")
    print("MODEL RESULT")
    print("========================================")
    print(result)

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------

    assert result is not None
    assert len(result.strip()) > 0

    print("\nVision test completed successfully.")