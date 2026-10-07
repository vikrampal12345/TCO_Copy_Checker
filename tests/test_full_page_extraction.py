import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from app.core.model_provider import get_model_provider


# ============================================================
# CONFIG
# ============================================================

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
    / "vision_tiles"
)

TILE_NAMES = [
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
]


# ============================================================
# IMAGE HELPERS
# ============================================================

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


# ============================================================
# TILE CREATION
# ============================================================

def create_tiles(
    image: Image.Image,
) -> list[tuple[str, Path]]:
    """
    Create four overlapping tiles.

        +-------------------+-------------------+
        |                   |                   |
        |     top_left      |     top_right     |
        |                   |                   |
        +-------------------+-------------------+
        |                   |                   |
        |    bottom_left    |    bottom_right   |
        |                   |                   |
        +-------------------+-------------------+
    """

    TILE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    width, height = image.size

    overlap_x = int(width * 0.08)
    overlap_y = int(height * 0.08)

    mid_x = width // 2
    mid_y = height // 2

    regions = {
        "top_left": (
            0,
            0,
            min(width, mid_x + overlap_x),
            min(height, mid_y + overlap_y),
        ),
        "top_right": (
            max(0, mid_x - overlap_x),
            0,
            width,
            min(height, mid_y + overlap_y),
        ),
        "bottom_left": (
            0,
            max(0, mid_y - overlap_y),
            min(width, mid_x + overlap_x),
            height,
        ),
        "bottom_right": (
            max(0, mid_x - overlap_x),
            max(0, mid_y - overlap_y),
            width,
            height,
        ),
    }

    results = []

    for name, box in regions.items():

        tile_path = (
            TILE_DIR
            / f"{name}.jpg"
        )

        # Recreate from source to ensure consistency.
        tile = image.crop(box)

        tile.save(
            tile_path,
            format="JPEG",
            quality=95,
            optimize=True,
        )

        results.append(
            (
                name,
                tile_path,
            )
        )

    return results


# ============================================================
# PROMPT
# ============================================================

VISION_PROMPT = """
You are extracting answers from one region of a student's
handwritten answer sheet.

Read this image carefully from TOP TO BOTTOM.

Return ONLY valid JSON.

Required format:

{
  "questions": [
    {
      "question_no": 7,
      "answer": "A",
      "confidence": 0.95
    }
  ]
}

RULES:

1. One object per visible question.
2. question_no must be the actual visible question number.
3. answer must be the student's visible answer.
4. For MCQs, use A, B, C, or D when clearly visible.
5. Never combine multiple questions into one object.
6. Preserve top-to-bottom order.
7. Never invent question numbers.
8. Never invent answers.
9. If a question number is genuinely unreadable, use null.
10. If the answer is genuinely unreadable, use "[UNCLEAR]".
11. Do not assign marks.
12. Do not judge correctness.
13. confidence must be between 0 and 1.
14. Return no explanation outside JSON.
15. Ignore page borders, printed instructions, and headings.
16. Only report actual question-answer pairs.
"""


# ============================================================
# JSON PARSING
# ============================================================

def parse_model_json(raw: str) -> dict:
    """
    Parse JSON returned by a local vision model.

    Handles occasional markdown code fences.
    """

    if raw is None:
        raise ValueError(
            "Model returned None."
        )

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
        raise ValueError(
            "Model did not return valid JSON.\n\n"
            f"RAW OUTPUT:\n{raw}\n\n"
            f"JSON ERROR:\n{exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "JSON root must be an object."
        )

    if "questions" not in data:
        raise ValueError(
            "JSON does not contain 'questions'."
        )

    if not isinstance(
        data["questions"],
        list,
    ):
        raise ValueError(
            "'questions' must be a list."
        )

    return data


# ============================================================
# TILE EXTRACTION
# ============================================================

def analyze_tile(
    client,
    model: str,
    tile_name: str,
    tile_path: Path,
) -> dict:

    print("\n")
    print("=" * 70)
    print(f"PROCESSING TILE: {tile_name}")
    print("=" * 70)

    print(
        f"Image: {tile_path}"
    )

    image_url = image_to_data_url(
        tile_path
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
                        "answer extraction system. "
                        "Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": VISION_PROMPT,
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
    print("-" * 70)
    print(raw)

    data = parse_model_json(
        raw
    )

    print("\nPARSED TILE DATA:")
    print(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
    )

    return data


# ============================================================
# MERGE RESULTS
# ============================================================

def merge_tile_results(
    tile_results: list[tuple[str, dict]],
) -> list[dict]:
    """
    Merge questions from all tiles.

    Since tiles overlap, the same question may appear more
    than once. We select the candidate with the strongest
    confidence.

    This is deterministic logic, not LLM reasoning.
    """

    candidates: dict[int, list[dict]] = {}

    unknown_questions = []

    for tile_name, result in tile_results:

        for item in result.get(
            "questions",
            [],
        ):

            question_no = item.get(
                "question_no"
            )

            answer = item.get(
                "answer",
                "[UNCLEAR]",
            )

            confidence = item.get(
                "confidence",
                0.0,
            )

            candidate = {
                "question_no": question_no,
                "answer": answer,
                "confidence": confidence,
                "source_tile": tile_name,
            }

            if question_no is None:
                unknown_questions.append(
                    candidate
                )
                continue

            candidates.setdefault(
                question_no,
                [],
            ).append(
                candidate
            )

    # --------------------------------------------------------
    # Choose best candidate per question
    # --------------------------------------------------------

    merged = []

    for question_no, items in candidates.items():

        # Highest confidence first.
        items_sorted = sorted(
            items,
            key=lambda x: x["confidence"],
            reverse=True,
        )

        best = items_sorted[0]

        # Preserve useful disagreement information.
        alternatives = [
            item
            for item in items_sorted[1:]
            if (
                item["answer"]
                != best["answer"]
            )
        ]

        merged_item = {
            "question_no": question_no,
            "answer": best["answer"],
            "confidence": best["confidence"],
            "source_tile": best["source_tile"],
        }

        if alternatives:
            merged_item[
                "conflicting_candidates"
            ] = alternatives

        merged.append(
            merged_item
        )

    # --------------------------------------------------------
    # Sort question numbers
    # --------------------------------------------------------

    merged.sort(
        key=lambda x: x["question_no"]
    )

    # --------------------------------------------------------
    # Append unresolved items separately
    # --------------------------------------------------------

    for item in unknown_questions:

        merged.append(
            {
                "question_no": None,
                "answer": item["answer"],
                "confidence": item["confidence"],
                "source_tile": item["source_tile"],
                "review_required": True,
            }
        )

    return merged


# ============================================================
# MAIN TEST
# ============================================================

def test_full_page_extraction():

    print("\n")
    print("=" * 70)
    print("TCO COPY CHECKER")
    print("FULL PAGE → 4 TILES → JSON → MERGE")
    print("=" * 70)

    # --------------------------------------------------------
    # Source image
    # --------------------------------------------------------

    image = load_image(
        SOURCE_IMAGE
    )

    print(
        f"\nSource image:"
        f" {image.size[0]} x {image.size[1]}"
    )

    # --------------------------------------------------------
    # Create tiles
    # --------------------------------------------------------

    tiles = create_tiles(
        image
    )

    print(
        f"\nCreated {len(tiles)} tiles."
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    provider = get_model_provider()

    client = provider.get_client()

    model = provider.get_vision_model()

    print(
        f"\nProvider: "
        f"{provider.get_provider_name()}"
    )

    print(
        f"Vision model: "
        f"{model}"
    )

    # --------------------------------------------------------
    # Process tiles
    # --------------------------------------------------------

    tile_results = []

    for tile_name, tile_path in tiles:

        try:

            result = analyze_tile(
                client=client,
                model=model,
                tile_name=tile_name,
                tile_path=tile_path,
            )

        except Exception as exc:

            pytest.fail(
                f"\nTile '{tile_name}' failed:\n"
                f"{type(exc).__name__}: {exc}"
            )

        tile_results.append(
            (
                tile_name,
                result,
            )
        )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = merge_tile_results(
        tile_results
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MERGED PAGE RESULT")
    print("=" * 70)

    print(
        json.dumps(
            {
                "page_no": 1,
                "questions": merged,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    assert isinstance(
        merged,
        list,
    )

    for item in merged:

        assert "answer" in item
        assert "confidence" in item

        confidence = item[
            "confidence"
        ]

        assert 0 <= confidence <= 1

    # --------------------------------------------------------
    # Print question summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("QUESTION SUMMARY")
    print("=" * 70)

    numbered = [
        item
        for item in merged
        if item["question_no"] is not None
    ]

    for item in numbered:

        print(
            f"Q{item['question_no']}: "
            f"{item['answer']} "
            f"(confidence="
            f"{item['confidence']})"
        )

    print("\n")
    print(
        f"Detected numbered questions: "
        f"{len(numbered)}"
    )

    print("\n")
    print("=" * 70)
    print("FULL PAGE EXTRACTION TEST PASSED")
    print("=" * 70)