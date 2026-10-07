import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from app.core.model_provider import get_model_provider


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "pages"
    / "Student_2"
    / "page_001_vision.jpg"
)

TILE_DIR = (
    PROJECT_ROOT
    / "data"
    / "pages"
    / "Student_2"
    / "vision_tiles_8"
)

TILE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PROMPT = """
Analyze this small region of a student's handwritten
answer sheet.

Read the region from TOP TO BOTTOM.

Identify every visible question number and the student's
answer associated with that question.

Return ONLY valid JSON:

{
  "questions": [
    {
      "question_no": 7,
      "answer": "A",
      "confidence": 0.9
    }
  ]
}

Rules:

1. One object for each visible question.
2. Keep question number and answer together based on
   their physical position in the image.
3. For MCQs use A, B, C, or D when clearly visible.
4. Never combine multiple questions.
5. Never invent question numbers.
6. Never invent answers.
7. If question number cannot be read, use null.
8. If answer cannot be read, use "[UNCLEAR]".
9. Do not assign marks.
10. Do not judge correctness.
11. confidence must be between 0 and 1.
12. Ignore headings and instructions.
13. Return no text outside the JSON.
"""


def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{path}"
        )

    image = Image.open(path)
    image.load()

    return image.convert("RGB")


def image_to_data_url(path: Path) -> str:
    image = load_image(path)

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95,
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


def create_8_tiles(
    image: Image.Image,
) -> list[tuple[str, Path]]:

    width, height = image.size

    # Two columns and four rows
    rows = 4
    cols = 2

    tile_width = width // cols
    tile_height = height // rows

    overlap_x = int(tile_width * 0.10)
    overlap_y = int(tile_height * 0.10)

    tiles = []

    for row in range(rows):

        for col in range(cols):

            x1 = (
                col * tile_width
                - (overlap_x if col > 0 else 0)
            )

            y1 = (
                row * tile_height
                - (overlap_y if row > 0 else 0)
            )

            x2 = (
                (col + 1) * tile_width
                + (overlap_x if col < cols - 1 else 0)
            )

            y2 = (
                (row + 1) * tile_height
                + (overlap_y if row < rows - 1 else 0)
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            tile = image.crop(
                (x1, y1, x2, y2)
            )

            tile_number = (
                row * cols + col + 1
            )

            name = f"tile_{tile_number:02d}"

            path = (
                TILE_DIR
                / f"{name}.jpg"
            )

            tile.save(
                path,
                format="JPEG",
                quality=95,
                optimize=True,
            )

            tiles.append(
                (name, path)
            )

            print(
                f"Created {name}: "
                f"{tile.size[0]} x {tile.size[1]}"
            )

    return tiles


def parse_json(raw: str) -> dict:

    cleaned = raw.strip()

    if cleaned.startswith("```"):
        cleaned = (
            cleaned
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    data = json.loads(cleaned)

    if not isinstance(data, dict):
        raise ValueError(
            "Model JSON root must be an object."
        )

    if "questions" not in data:
        raise ValueError(
            "Missing 'questions'."
        )

    if not isinstance(
        data["questions"],
        list,
    ):
        raise ValueError(
            "'questions' must be a list."
        )

    return data


def analyze_tile(
    client,
    model: str,
    name: str,
    path: Path,
) -> dict:

    print("\n")
    print("=" * 70)
    print(f"PROCESSING {name}")
    print("=" * 70)

    image_url = image_to_data_url(
        path
    )

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
                        "answer extractor. Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT,
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

    raw = (
        response
        .choices[0]
        .message
        .content
    )

    print("\nRAW OUTPUT:")
    print(raw)

    try:
        data = parse_json(raw)

    except Exception as exc:
        raise ValueError(
            f"Invalid JSON from {name}:\n{exc}\n\n"
            f"Raw output:\n{raw}"
        )

    print("\nPARSED:")
    print(
        json.dumps(
            data,
            indent=2,
        )
    )

    return data


def merge_results(
    results: list[tuple[str, dict]],
) -> list[dict]:

    grouped = {}

    unresolved = []

    for tile_name, result in results:

        for item in result["questions"]:

            q_no = item.get(
                "question_no"
            )

            answer = item.get(
                "answer",
                "[UNCLEAR]",
            )

            confidence = float(
                item.get(
                    "confidence",
                    0.0,
                )
            )

            candidate = {
                "question_no": q_no,
                "answer": answer,
                "confidence": confidence,
                "source_tile": tile_name,
            }

            if q_no is None:

                unresolved.append(
                    candidate
                )

            else:

                grouped.setdefault(
                    q_no,
                    [],
                ).append(
                    candidate
                )

    merged = []

    for q_no, candidates in grouped.items():

        candidates.sort(
            key=lambda x: x["confidence"],
            reverse=True,
        )

        best = candidates[0]

        merged.append(
            best
        )

    merged.sort(
        key=lambda x: x["question_no"]
    )

    # Do not automatically convert unresolved items into
    # question numbers. Keep them separate for human review.
    return merged, unresolved


def test_8_tile_extraction():

    print("\n")
    print("=" * 70)
    print("TCO COPY CHECKER - 8 TILE EXTRACTION")
    print("=" * 70)

    image = load_image(
        SOURCE_IMAGE
    )

    print(
        f"\nSource:"
        f" {image.size[0]} x {image.size[1]}"
    )

    tiles = create_8_tiles(
        image
    )

    provider = get_model_provider()

    client = provider.get_client()

    model = provider.get_vision_model()

    print(
        f"\nProvider:"
        f" {provider.get_provider_name()}"
    )

    print(
        f"Vision model:"
        f" {model}"
    )

    results = []

    for name, path in tiles:

        try:

            data = analyze_tile(
                client,
                model,
                name,
                path,
            )

        except Exception as exc:

            pytest.fail(
                f"{name} failed:\n{exc}"
            )

        results.append(
            (name, data)
        )

    merged, unresolved = merge_results(
        results
    )

    print("\n")
    print("=" * 70)
    print("MERGED PAGE RESULT")
    print("=" * 70)

    print(
        json.dumps(
            {
                "page_no": 1,
                "questions": merged,
                "unresolved_regions": unresolved,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n")
    print("=" * 70)
    print("QUESTION SUMMARY")
    print("=" * 70)

    for item in merged:

        print(
            f"Q{item['question_no']}: "
            f"{item['answer']} "
            f"(confidence="
            f"{item['confidence']})"
        )

    print("\nDetected questions:", len(merged))
    print("Unresolved regions:", len(unresolved))

    assert isinstance(
        merged,
        list,
    )

    print("\n")
    print("=" * 70)
    print("8 TILE EXTRACTION TEST PASSED")
    print("=" * 70)