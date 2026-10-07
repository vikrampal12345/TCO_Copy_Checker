import base64
import io
from pathlib import Path

import pytest
from PIL import Image
from openai import BadRequestError

from app.core.model_provider import get_model_provider


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
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

TILE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    try:
        image = Image.open(image_path)
        image.load()
        return image.convert("RGB")

    except Exception as exc:
        raise ValueError(
            f"Could not open image: {image_path}\n"
            f"Reason: {exc}"
        ) from exc


# ============================================================
# CREATE OVERLAPPING TILES
# ============================================================

def create_tiles(
    image: Image.Image,
) -> list[tuple[str, Path]]:
    """
    Split the page into 4 overlapping regions.

    Layout:

        +-------------+-------------+
        |             |             |
        |   top-left  | top-right   |
        |             |             |
        +-------------+-------------+
        |             |             |
        | bottom-left | bottom-right|
        |             |             |
        +-------------+-------------+

    Overlap helps prevent content near tile boundaries
    from being cut off.
    """

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

        tile = image.crop(box)

        tile_path = (
            TILE_DIR
            / f"{name}.jpg"
        )

        tile.save(
            tile_path,
            format="JPEG",
            quality=95,
            optimize=True,
        )

        results.append(
            (name, tile_path)
        )

        print(
            f"\nCreated {name}: "
            f"{tile.size[0]} x {tile.size[1]}"
        )

        print(
            f"Path: {tile_path}"
        )

    return results


# ============================================================
# IMAGE -> DATA URL
# ============================================================

def image_to_data_url(
    image_path: Path,
) -> str:

    image = load_image(image_path)

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95,
        optimize=True,
    )

    image_bytes = buffer.getvalue()

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# VISION PROMPT
# ============================================================

PROMPT = """
You are analyzing a cropped region of a student's
handwritten answer sheet.

Carefully read ONLY the visible content in this image.

Identify:

1. Every visible question number.
2. The student's handwritten answer.
3. Whether the answer is complete, incomplete,
   or unclear.
4. Any visible mathematical formula or equation.
5. Any visible diagram or figure.
6. Any text that cannot be read reliably.

Do NOT:
- assign marks
- judge correctness
- invent text
- assume missing content
- reconstruct words that are not visible

For handwriting that cannot be read reliably,
write [UNCLEAR].

Return exactly:

QUESTIONS:

- Question: <question number>
  Answer: <visible handwritten answer>
  Complete: yes/no/unclear
  Formula: <formula or None>
  Diagram: <description or None>
  Confidence: <0.0 to 1.0>

If no question number is visible:

Question: [UNCLEAR]

If no answer is visible:

Answer: No answer visible
"""


# ============================================================
# TEST REAL VISION
# ============================================================

def test_real_vision_tiles():

    print("\n")
    print("=" * 70)
    print("TCO COPY CHECKER - TILED VISION TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = load_image(
        IMAGE_PATH
    )

    print(
        f"\nOriginal page:"
        f" {image.size[0]} x {image.size[1]}"
    )

    # --------------------------------------------------------
    # Create tiles
    # --------------------------------------------------------

    tiles = create_tiles(image)

    # --------------------------------------------------------
    # Model provider
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
    # Process each tile
    # --------------------------------------------------------

    all_results = []

    for index, (tile_name, tile_path) in enumerate(
        tiles,
        start=1,
    ):

        print("\n")
        print("=" * 70)
        print(
            f"TILE {index}/4: {tile_name}"
        )
        print("=" * 70)

        print(
            f"Tile path: {tile_path}"
        )

        print(
            f"Tile size: "
            f"{Image.open(tile_path).size}"
        )

        image_url = image_to_data_url(
            tile_path
        )

        try:

            response = (
                client
                .chat.completions
                .create(
                    model=model,
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a careful "
                                "handwritten answer-sheet "
                                "vision analyzer. "
                                "Never invent text."
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
                                        "url": image_url,
                                    },
                                },
                            ],
                        },
                    ],
                )
            )

        except BadRequestError as exc:

            print("\nLM STUDIO ERROR:")
            print(exc)

            pytest.fail(
                f"LM Studio rejected tile "
                f"{tile_name}: {exc}"
            )

        except Exception as exc:

            print(
                f"\nTile failed: "
                f"{type(exc).__name__}"
            )

            print(exc)

            pytest.fail(
                f"Vision request failed "
                f"for {tile_name}: {exc}"
            )

        result = (
            response
            .choices[0]
            .message
            .content
        )

        all_results.append(
            (
                tile_name,
                result,
            )
        )

        print("\nMODEL RESULT:")
        print("-" * 70)
        print(result)

        assert result is not None
        assert len(result.strip()) > 0

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL TILE SUMMARY")
    print("=" * 70)

    for tile_name, result in all_results:

        print("\n")
        print(f"### {tile_name}")
        print(result)

    print("\n")
    print("=" * 70)
    print("TILED VISION TEST PASSED")
    print("=" * 70)