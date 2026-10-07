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

TILE_DIR = (
    PROJECT_ROOT
    / "data"
    / "pages"
    / "Student_2"
    / "vision_tiles_8"
)

SEARCH_DIR = (
    PROJECT_ROOT
    / "data"
    / "pages"
    / "Student_2"
    / "vision_tiles_q1_search"
)

SEARCH_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Q1 could lie near the boundary between these regions.
SOURCE_TILES = [
    "tile_01.jpg",
    "tile_02.jpg",
    "tile_03.jpg",
]


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(
    path: Path,
) -> Image.Image:

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{path}"
        )

    image = Image.open(path)
    image.load()

    return image.convert("RGB")


# ============================================================
# CREATE TWO OVERLAPPING STRIPS
# ============================================================

def create_search_strips(
    image: Image.Image,
    tile_name: str,
) -> list[tuple[str, Path]]:

    width, height = image.size

    # Split into two horizontal regions.
    midpoint = height // 2

    overlap = int(
        height * 0.20
    )

    regions = [
        (
            "top",
            0,
            min(
                height,
                midpoint + overlap,
            ),
        ),
        (
            "bottom",
            max(
                0,
                midpoint - overlap,
            ),
            height,
        ),
    ]

    results = []

    for region_name, y1, y2 in regions:

        crop = image.crop(
            (
                0,
                y1,
                width,
                y2,
            )
        )

        output_name = (
            tile_name.replace(
                ".jpg",
                "",
            )
            + "_"
            + region_name
            + ".jpg"
        )

        output_path = (
            SEARCH_DIR
            / output_name
        )

        crop.save(
            output_path,
            format="JPEG",
            quality=97,
            optimize=True,
        )

        name = output_name.replace(
            ".jpg",
            "",
        )

        results.append(
            (
                name,
                output_path,
            )
        )

        print(
            f"\nCreated {name}: "
            f"{crop.size[0]} x "
            f"{crop.size[1]}"
        )

    return results


# ============================================================
# IMAGE -> DATA URL
# ============================================================

def image_to_data_url(
    path: Path,
) -> str:

    image = load_image(
        path
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=97,
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
# PROMPT
# ============================================================

Q1_PROMPT = """
You are inspecting a cropped region of a student's
handwritten answer sheet.

Your ONLY target is QUESTION 1.

Look extremely carefully for the printed question number
"1" and the student's answer associated with it.

The answer may be:
A, B, C, D, or another handwritten response.

Return ONLY valid JSON:

{
  "questions": [
    {
      "question_no": 1,
      "answer": "A",
      "confidence": 0.9
    }
  ]
}

Rules:

1. Return Question 1 ONLY if it is actually visible.
2. Do not guess.
3. Do not infer the answer from Q2, Q3, Q4, etc.
4. Use the physical position of the answer relative to
   Question 1.
5. If Question 1 is visible but its answer cannot be read,
   return:
   "answer": "[UNCLEAR]"
6. If Question 1 is not visible in this crop, return:
   {
     "questions": []
   }
7. Do not return Question 2, 3, 4, 5, etc.
8. Do not assign marks.
9. Do not judge correctness.
10. Do not invent missing text.
11. confidence must be between 0 and 1.
12. Return no explanation outside JSON.

IMPORTANT:
An empty result is better than a guessed result.
"""


# ============================================================
# PARSE JSON
# ============================================================

def parse_json(
    raw: str,
) -> dict:

    if raw is None:
        raise ValueError(
            "Model returned no content."
        )

    cleaned = raw.strip()

    # Remove markdown code fences.
    if cleaned.startswith("```"):

        lines = cleaned.splitlines()

        if (
            lines
            and lines[0]
            .strip()
            .startswith("```")
        ):
            lines = lines[1:]

        if (
            lines
            and lines[-1]
            .strip()
            == "```"
        ):
            lines = lines[:-1]

        cleaned = "\n".join(
            lines
        ).strip()

    try:

        data = json.loads(
            cleaned
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Invalid JSON.\n\n"
            f"RAW OUTPUT:\n{raw}\n\n"
            f"ERROR:\n{exc}"
        ) from exc

    # Model may return:
    #
    # [...]
    #
    # instead of:
    #
    # {"questions": [...]}

    if isinstance(
        data,
        list,
    ):

        data = {
            "questions": data
        }

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "JSON root must be object or list."
        )

    if "questions" not in data:
        raise ValueError(
            "Missing questions field."
        )

    if not isinstance(
        data["questions"],
        list,
    ):
        raise ValueError(
            "questions must be a list."
        )

    normalized = []

    for item in data["questions"]:

        if not isinstance(
            item,
            dict,
        ):
            continue

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

        try:
            confidence = float(
                confidence
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        if not 0 <= confidence <= 1:
            confidence = 0.0

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

        normalized.append(
            {
                "question_no": question_no,
                "answer": str(answer).strip(),
                "confidence": confidence,
            }
        )

    return {
        "questions": normalized
    }


# ============================================================
# ANALYZE ONE REGION
# ============================================================

def analyze_region(
    client,
    model: str,
    region_name: str,
    image_path: Path,
) -> dict:

    print("\n")
    print("=" * 70)
    print(
        f"PROCESSING {region_name}"
    )
    print("=" * 70)

    print(
        f"Image: {image_path}"
    )

    image_url = image_to_data_url(
        image_path
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
                            "You are a precise "
                            "handwritten answer "
                            "extraction system. "
                            "Return JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": Q1_PROMPT,
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
            f"Vision request failed for "
            f"{region_name}:\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc

    raw = (
        response
        .choices[0]
        .message
        .content
    )

    print("\nRAW OUTPUT:")
    print("-" * 70)
    print(raw)

    data = parse_json(
        raw
    )

    print("\nPARSED:")
    print(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
    )

    return data


# ============================================================
# COLLECT Q1 CANDIDATES
# ============================================================

def collect_q1_candidates(
    results: list[
        tuple[str, dict]
    ],
) -> list[dict]:

    candidates = []

    for region_name, result in results:

        for item in result.get(
            "questions",
            [],
        ):

            if item.get(
                "question_no"
            ) != 1:
                continue

            candidates.append(
                {
                    "question_no": 1,
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
                    "source_region": region_name,
                }
            )

    return candidates


# ============================================================
# SELECT BEST CANDIDATE
# ============================================================

def select_best_q1(
    candidates: list[dict],
) -> dict | None:

    if not candidates:
        return None

    # Prefer:
    # 1. readable answer
    # 2. higher confidence

    def ranking(item):

        answer = item[
            "answer"
        ]

        readable = (
            answer != "[UNCLEAR]"
        )

        return (
            readable,
            item["confidence"],
        )

    return max(
        candidates,
        key=ranking,
    )


# ============================================================
# MAIN TEST
# ============================================================

def test_targeted_q1_search():

    print("\n")
    print("=" * 70)
    print(
        "TCO COPY CHECKER"
    )
    print(
        "TARGETED QUESTION 1 SEARCH"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load target tiles
    # --------------------------------------------------------

    all_regions = []

    for tile_filename in SOURCE_TILES:

        tile_path = (
            TILE_DIR
            / tile_filename
        )

        if not tile_path.exists():

            pytest.fail(
                f"Missing tile:\n"
                f"{tile_path}"
            )

        image = load_image(
            tile_path
        )

        print(
            f"\nSource tile "
            f"{tile_filename}: "
            f"{image.size[0]} x "
            f"{image.size[1]}"
        )

        regions = create_search_strips(
            image,
            tile_filename,
        )

        all_regions.extend(
            regions
        )

    print(
        f"\nTotal regions to inspect: "
        f"{len(all_regions)}"
    )

    # --------------------------------------------------------
    # Provider
    # --------------------------------------------------------

    provider = (
        get_model_provider()
    )

    client = provider.get_client()

    model = (
        provider.get_vision_model()
    )

    print(
        f"\nProvider: "
        f"{provider.get_provider_name()}"
    )

    print(
        f"Vision model: "
        f"{model}"
    )

    # --------------------------------------------------------
    # Process every region
    # --------------------------------------------------------

    results = []

    for region_name, image_path in all_regions:

        try:

            data = analyze_region(
                client=client,
                model=model,
                region_name=region_name,
                image_path=image_path,
            )

        except Exception as exc:

            pytest.fail(
                f"\nFailed on "
                f"{region_name}:\n"
                f"{exc}"
            )

        results.append(
            (
                region_name,
                data,
            )
        )

    # --------------------------------------------------------
    # Collect candidates
    # --------------------------------------------------------

    candidates = collect_q1_candidates(
        results
    )

    # --------------------------------------------------------
    # Best Q1
    # --------------------------------------------------------

    best = select_best_q1(
        candidates
    )

    # --------------------------------------------------------
    # Print all candidates
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "Q1 CANDIDATES"
    )
    print("=" * 70)

    print(
        json.dumps(
            candidates,
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # Print best candidate
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "BEST Q1 RESULT"
    )
    print("=" * 70)

    if best is None:

        print(
            "Q1: NOT DETECTED"
        )

    else:

        print(
            f"Q1: "
            f"{best['answer']} "
            f"(confidence="
            f"{best['confidence']}, "
            f"source="
            f"{best['source_region']})"
        )

    # --------------------------------------------------------
    # Final JSON
    # --------------------------------------------------------

    final_result = {
        "question_no": 1,
        "best_candidate": best,
        "all_candidates": candidates,
    }

    print("\n")
    print("=" * 70)
    print(
        "FINAL Q1 RESULT"
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
    # Validation
    # --------------------------------------------------------

    assert isinstance(
        candidates,
        list,
    )

    for candidate in candidates:

        assert (
            candidate["question_no"]
            == 1
        )

        assert isinstance(
            candidate["answer"],
            str,
        )

        assert 0 <= (
            candidate["confidence"]
        ) <= 1

    print("\n")
    print("=" * 70)
    print(
        "TARGETED Q1 SEARCH PASSED"
    )
    print("=" * 70)