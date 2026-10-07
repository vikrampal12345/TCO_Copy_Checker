from __future__ import annotations

import base64
import io
import json
from collections.abc import Sequence
from typing import Any

from PIL import Image

from app.core.model_provider import get_model_provider
from app.schemas.submission import AggregatedSubmission
from app.schemas.submission_verification import (
    SubmissionVerificationResult,
)


class SubmissionVerifierModel:
    """
    Model adapter for combined submission verification.

    The verifier receives:
    - original page images
    - aggregated student-answer extraction

    Its job is to verify extraction correctness and completeness.

    It does NOT:
    - talk to the teacher
    - grade answers
    - write to the database
    - make teacher-facing decisions
    """

    def __init__(self, provider=None) -> None:
        self.provider = provider or get_model_provider()
        self.client = self.provider.get_client()
        self.model = self.provider.get_extraction_verifier_model()

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        text = raw.strip()

        if text.startswith("```"):
            lines = text.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Verifier model returned invalid JSON: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Verifier model response must be a JSON object."
            )

        return data

    @staticmethod
    def _image_to_data_url(
        image: Image.Image,
        *,
        max_dimension: int = 1800,
        jpeg_quality: int = 90,
    ) -> str:
        if not isinstance(image, Image.Image):
            raise TypeError(
                "Each page image must be a PIL.Image.Image."
            )

        image = image.convert("RGB")

        width, height = image.size
        largest_dimension = max(width, height)

        if largest_dimension > max_dimension:
            scale = max_dimension / largest_dimension

            new_size = (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            )

            image = image.resize(
                new_size,
                Image.Resampling.LANCZOS,
            )

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _build_prompt(
        submission: AggregatedSubmission,
        attempt: int,
        page_count: int,
    ) -> str:
        payload = submission.model_dump()

        return f"""
You are an independent verification model for a handwritten student
answer-sheet extraction pipeline.

You are given:
1. The original handwritten answer-sheet page images.
2. The aggregated extraction produced by another model.

Your job is ONLY to verify whether the extracted submission matches
the original handwritten pages well enough for downstream grading.

Do NOT grade the student's answers.
Do NOT assign marks.
Do NOT communicate with the teacher.

Check:
1. Question numbers against the original handwriting.
2. Whether important questions are missing.
3. Whether answers belong to the correct question numbers.
4. Whether extracted text is faithful to the handwriting.
5. Whether repeated observations are actually duplicates or conflicts.
6. Whether [UNCLEAR], unreadable, or hallucinated content remains.
7. Whether the complete submission is reliable enough to pass to grading.

This is verification attempt {attempt}.
Number of original pages supplied: {page_count}.

Return ONLY valid JSON:

{{
  "passed": true,
  "confidence": 0.95,
  "issues": [
    {{
      "question_no": 3,
      "issue_type": "extraction_error",
      "severity": "high",
      "message": "The extracted answer does not match the handwritten answer."
    }}
  ],
  "feedback": "Re-extract Q3 from the original page.",
  "warnings": []
}}

Rules:
- passed=true only when the combined extraction is sufficiently reliable.
- passed=false when an important extraction problem remains.
- confidence must be between 0 and 1.
- question_no may be null for submission-level problems.
- feedback must contain actionable retry instructions when passed=false.

Aggregated submission:

{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

    def verify(
        self,
        submission: AggregatedSubmission,
        *,
        attempt: int,
        page_images: Sequence[tuple[int, Image.Image]] | None = None,
        temperature: float = 0,
    ) -> SubmissionVerificationResult:
        if not isinstance(submission, AggregatedSubmission):
            raise TypeError(
                "submission must be an AggregatedSubmission"
            )

        if attempt < 1:
            raise ValueError("attempt must be >= 1")

        page_images = list(page_images or [])

        prompt = self._build_prompt(
            submission,
            attempt,
            len(page_images),
        )

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        for page_no, image in page_images:
            image_url = self._image_to_data_url(image)

            user_content.append(
                {
                    "type": "text",
                    "text": f"Original answer-sheet page {page_no}:",
                }
            )

            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                }
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict independent handwritten "
                        "answer-sheet extraction verification model. "
                        "Compare the provided original pages with the "
                        "aggregated extraction. Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=temperature,
        )

        if not response.choices:
            raise RuntimeError(
                "Verifier model returned no choices."
            )

        raw = response.choices[0].message.content

        if not raw:
            raise ValueError(
                "Verifier model returned an empty response."
            )

        data = self._parse_json_response(raw)

        data["attempt"] = attempt

        return SubmissionVerificationResult.model_validate(data)
