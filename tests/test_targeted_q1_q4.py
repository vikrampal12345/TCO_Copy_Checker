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

SOURCE_TILE = (
    PROJECT_ROOT
    / "data"
    / "pages"
    / "Student_2"
    / "vision_tiles_8"
    / "tile_03.jpg"
)

TARGET_TILE_DIR = (
    PROJECT_ROOT
    / "data"
    / "pages"
    / "Student_2"
    / "vision_tiles_q1_q4"
)

TARGET_TILE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(
    path: Path,
) -> Image.Image:
    """
    Load an image and convert it to RGB.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file:\n{path}"
        )

    try:
        image = Image.open(path)

        # Force complete image decoding.
        image.load()

        return image.convert("RGB")

    except Exception as exc:
        raise ValueError(
            f"Could not open image:\n"
            f"{path}\n\n"
            f"Reason: {exc}"
        ) from exc


# ============================================================
# CREATE HORIZONTAL STRIPS
# ============================================================

def create_horizontal_strips(
    image: Image.Image,
) -> list[tuple[str, Path]]:
    """
    Split tile_03 into four overlapping horizontal strips.

    Example:

        strip_01
        strip_02
        strip_03
        strip_04

    Overlap helps avoid losing a question at a boundary.
    """

    width, height = image.size

    strip_count = 4

    overlap = int(
        height * 0.15
    )

    base_height = height // strip_count

    results = []

    for index in range(strip_count):

        # ----------------------------------------------------
        # Calculate vertical boundaries
        # ----------------------------------------------------

        y1 = (
            index * base_height
            - (
                overlap
                if index > 0
                else 0
            )
        )

        y2 = (
            (index + 1) * base_height
            + (
                overlap
                if index < strip_count - 1
                else 0
            )
        )

        # Keep coordinates inside image.
        y1 = max(
            0,
            y1,
        )

        y2 = min(
            height,
            y2,
        )

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        strip = image.crop(
            (
                0,
                y1,
                width,
                y2,
            )
        )

        name = (
            f"strip_{index + 1:02d}"
        )

        path = (
            TARGET_TILE_DIR
            / f"{name}.jpg"
        )

        # ----------------------------------------------------
        # Save normalized JPEG
        # ----------------------------------------------------

        strip.save(
            path,
            format="JPEG",
            quality=95,
            optimize=True,
        )

        results.append(
            (
                name,
                path,
            )
        )

        print(
            f"\nCreated {name}: "
            f"{strip.size[0]} x "
            f"{strip.size[1]}"
        )

        print(
            f"Path: {path}"
        )

    return results


# ============================================================
# IMAGE -> DATA URL
# ============================================================

def image_to_data_url(
    path: Path,
) -> str:
    """
    Normalize image to JPEG and convert it to a base64
    data URL for the LM Studio OpenAI-compatible API.
    """

    image = load_image(
        path
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95,
        optimize=True,
    )

    image_bytes = (
        buffer.getvalue()
    )

    if not image_bytes:
        raise ValueError(
            "Image encoding produced empty data."
        )

    encoded = (
        base64
        .b64encode(image_bytes)
        .decode("utf-8")
    )

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# TARGETED PROMPT
# ============================================================

TARGETED_PROMPT = """
You are carefully inspecting a small cropped region from
a student's handwritten answer sheet.

TARGET QUESTIONS:
- Question 1
- Question 4

Your primary goal is to find Question 1 and/or Question 4
in this image.

Look carefully for:

- printed question numbers
- handwritten answers
- selected MCQ options
- the spatial relationship between the question number
  and its answer

Return ONLY valid JSON.

Preferred format:

{
  "questions": [
    {
      "question_no": 1,
      "answer": "A",
      "confidence": 0.9
    },
    {
      "question_no": 4,
      "answer": "B",
      "confidence": 0.9
    }
  ]
}

Rules:

1. Return Question 1 only if it is actually visible.
2. Return Question 4 only if it is actually visible.
3. Do not invent either question.
4. The answer must belong to that exact question based
   on its physical position.
5. For MCQ answers, use A, B, C, or D when clearly visible.
6. If the question number is visible but the answer is
   unreadable, use:
   "[UNCLEAR]"
7. If the answer is not visible, use:
   "[UNCLEAR]"
8. If neither target question is visible in this crop,
   return:
   {
     "questions": []
   }
9. Do not infer an answer from nearby question numbers.
10. Do not use previous model outputs.
11. Do not assign marks.
12. Do not judge correctness.
13. Do not invent missing text.
14. confidence must be between 0 and 1.
15. Do not include explanations outside JSON.

IMPORTANT:

An empty result is better than a guessed result.
"""


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(
    raw: str,
) -> dict:
    """
    Parse and normalize model JSON.

    Supported model outputs:

    FORMAT 1:
    {
        "questions": [...]
    }

    FORMAT 2:
    [
        {
            "question_no": 4,
            "answer": "[UNCLEAR]",
            "confidence": 0.9
        }
    ]

    Both are normalized to:

    {
        "questions": [...]
    }
    """

    # --------------------------------------------------------
    # Empty output
    # --------------------------------------------------------

    if raw is None:
        raise ValueError(
            "Model returned no content."
        )

    cleaned = raw.strip()

    if not cleaned:
        raise ValueError(
            "Model returned an empty response."
        )

    # --------------------------------------------------------
    # Remove markdown code fences
    # --------------------------------------------------------

    if cleaned.startswith("```"):

        lines = cleaned.splitlines()

        # Remove opening fence
        if (
            lines
            and lines[0].strip().startswith("```")
        ):
            lines = lines[1:]

        # Remove closing fence
        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        cleaned = "\n".join(
            lines
        ).strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        data = json.loads(
            cleaned
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Model did not return valid JSON.\n\n"
            "RAW OUTPUT:\n"
            f"{raw}\n\n"
            "JSON ERROR:\n"
            f"{exc}"
        ) from exc

    # --------------------------------------------------------
    # Model returned a LIST
    # --------------------------------------------------------

    if isinstance(data, list):

        data = {
            "questions": data
        }

    # --------------------------------------------------------
    # Root must now be a dictionary
    # --------------------------------------------------------

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "JSON root must be either "
            "an object or an array."
        )

    # --------------------------------------------------------
    # Questions field
    # --------------------------------------------------------

    if "questions" not in data:
        raise ValueError(
            "JSON object is missing "
            "'questions'."
        )

    if not isinstance(
        data["questions"],
        list,
    ):
        raise ValueError(
            "'questions' must be a list."
        )

    # --------------------------------------------------------
    # Validate question entries
    # --------------------------------------------------------

    normalized_questions = []

    for index, item in enumerate(
        data["questions"],
        start=1,
    ):

        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                f"Question item #{index} "
                "must be an object."
            )

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

        # ----------------------------------------------------
        # Normalize confidence
        # ----------------------------------------------------

        try:
            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                f"Invalid confidence for "
                f"question item #{index}: "
                f"{confidence}"
            ) from exc

        if not 0 <= confidence <= 1:
            raise ValueError(
                f"Confidence must be "
                f"between 0 and 1, got "
                f"{confidence}."
            )

        # ----------------------------------------------------
        # Validate question number
        # ----------------------------------------------------

        if question_no is not None:

            if isinstance(
                question_no,
                bool,
            ):
                raise ValueError(
                    "question_no cannot be bool."
                )

            try:

                question_no = int(
                    question_no
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    f"Invalid question_no: "
                    f"{question_no}"
                ) from exc

        # ----------------------------------------------------
        # Normalize answer
        # ----------------------------------------------------

        answer = str(
            answer
        ).strip()

        if not answer:
            answer = "[UNCLEAR]"

        normalized_questions.append(
            {
                "question_no": question_no,
                "answer": answer,
                "confidence": confidence,
            }
        )

    # --------------------------------------------------------
    # Final normalized object
    # --------------------------------------------------------

    return {
        "questions": normalized_questions
    }


# ============================================================
# ANALYZE ONE STRIP
# ============================================================

def analyze_strip(
    client,
    model: str,
    name: str,
    path: Path,
) -> dict:
    """
    Send one targeted image strip to the vision model.
    """

    print("\n")
    print("=" * 70)
    print(
        f"PROCESSING {name}"
    )
    print("=" * 70)

    print(
        f"Image: {path}"
    )

    image_url = image_to_data_url(
        path
    )

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
                                "text": TARGETED_PROMPT,
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

        raise RuntimeError(
            f"Vision request failed for {name}:\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Extract model response
    # --------------------------------------------------------

    if (
        not response.choices
    ):
        raise RuntimeError(
            f"Model returned no choices for {name}."
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

    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    data = parse_json(
        raw
    )

    print("\nPARSED OUTPUT:")
    print(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
    )

    return data


# ============================================================
# MERGE TARGETED RESULTS
# ============================================================

def merge_targeted_results(
    results: list[tuple[str, dict]],
) -> dict[int, list[dict]]:
    """
    Merge candidate results for Q1 and Q4.

    We keep ALL candidates.

    Later the strongest candidate can be selected
    deterministically.
    """

    grouped: dict[int, list[dict]] = {
        1: [],
        4: [],
    }

    for strip_name, result in results:

        for item in result.get(
            "questions",
            [],
        ):

            question_no = item.get(
                "question_no"
            )

            if question_no not in (
                1,
                4,
            ):
                continue

            candidate = {
                "question_no": question_no,
                "answer": item.get(
                    "answer",
                    "[UNCLEAR]",
                ),
                "confidence": float(
                    item.get(
                        "confidence",
                        0.0,
                    )
                ),
                "source_strip": strip_name,
            }

            grouped[
                question_no
            ].append(
                candidate
            )

    return grouped


# ============================================================
# SELECT BEST CANDIDATE
# ============================================================

def select_best_candidate(
    candidates: list[dict],
) -> dict | None:
    """
    Select the candidate with highest confidence.

    If multiple candidates have the same confidence,
    prefer a readable answer over [UNCLEAR].
    """

    if not candidates:
        return None

    def sort_key(item: dict):

        answer = item.get(
            "answer",
            "[UNCLEAR]",
        )

        readable = (
            answer != "[UNCLEAR]"
        )

        return (
            float(
                item.get(
                    "confidence",
                    0.0,
                )
            ),
            readable,
        )

    return max(
        candidates,
        key=sort_key,
    )


# ============================================================
# MAIN TEST
# ============================================================

def test_targeted_q1_q4():

    print("\n")
    print("=" * 70)
    print(
        "TCO COPY CHECKER"
    )
    print(
        "TARGETED QUESTION 1 / QUESTION 4 TEST"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Check source tile
    # --------------------------------------------------------

    if not SOURCE_TILE.exists():
        pytest.fail(
            "Required source tile does not exist:\n"
            f"{SOURCE_TILE}"
        )

    # --------------------------------------------------------
    # Load tile_03
    # --------------------------------------------------------

    image = load_image(
        SOURCE_TILE
    )

    print(
        f"\nSource tile:"
        f" {image.size[0]} x "
        f"{image.size[1]}"
    )

    # --------------------------------------------------------
    # Create strips
    # --------------------------------------------------------

    strips = create_horizontal_strips(
        image
    )

    print(
        f"\nCreated "
        f"{len(strips)} strips."
    )

    # --------------------------------------------------------
    # Get model provider
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Process all strips
    # --------------------------------------------------------

    results = []

    for name, path in strips:

        try:

            result = analyze_strip(
                client=client,
                model=model,
                name=name,
                path=path,
            )

        except Exception as exc:

            pytest.fail(
                f"\nFailed on {name}:\n"
                f"{type(exc).__name__}: {exc}"
            )

        results.append(
            (
                name,
                result,
            )
        )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = merge_targeted_results(
        results
    )

    # --------------------------------------------------------
    # Select best Q1
    # --------------------------------------------------------

    best_q1 = select_best_candidate(
        merged[1]
    )

    # --------------------------------------------------------
    # Select best Q4
    # --------------------------------------------------------

    best_q4 = select_best_candidate(
        merged[4]
    )

    # --------------------------------------------------------
    # Final structure
    # --------------------------------------------------------

    final_result = {
        "Q1": merged[1],
        "Q4": merged[4],
        "best_Q1": best_q1,
        "best_Q4": best_q4,
    }

    # --------------------------------------------------------
    # Print candidates
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "ALL TARGETED CANDIDATES"
    )
    print("=" * 70)

    print(
        json.dumps(
            final_result,
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # Best candidates
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "BEST CANDIDATES"
    )
    print("=" * 70)

    if best_q1 is None:

        print(
            "Q1: NOT DETECTED"
        )

    else:

        print(
            f"Q1: "
            f"{best_q1['answer']} "
            f"(confidence="
            f"{best_q1['confidence']}, "
            f"source="
            f"{best_q1['source_strip']})"
        )

    if best_q4 is None:

        print(
            "Q4: NOT DETECTED"
        )

    else:

        print(
            f"Q4: "
            f"{best_q4['answer']} "
            f"(confidence="
            f"{best_q4['confidence']}, "
            f"source="
            f"{best_q4['source_strip']})"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert isinstance(
        merged,
        dict,
    )

    assert 1 in merged
    assert 4 in merged

    for question_no in (
        1,
        4,
    ):

        for candidate in merged[
            question_no
        ]:

            assert (
                candidate["question_no"]
                == question_no
            )

            assert isinstance(
                candidate["answer"],
                str,
            )

            assert 0 <= (
                candidate["confidence"]
            ) <= 1

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "TARGETED Q1/Q4 TEST PASSED"
    )
    print("=" * 70)