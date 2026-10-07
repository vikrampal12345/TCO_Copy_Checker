from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image

from app.schemas.extraction import PageExtractionResult
from app.schemas.extraction_verification import (
    ExtractionVerificationIssue,
    ExtractionVerificationResult,
)


class ExtractionVerifierModel:
    """
    Independent AI model used only for verifying extraction.

    Important:
    - It does NOT perform grading.
    - It does NOT modify extraction directly.
    - It only compares the original page with the extracted result.
    - It returns PASS/FAIL and feedback.

    Model/provider are intentionally configurable so the
    verifier can later be replaced independently from the
    extraction model.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv(
                "MODEL_BASE_URL",
                "http://127.0.0.1:1234/v1",
            )
        )

        self.api_key = (
            api_key
            or os.getenv(
                "MODEL_API_KEY",
                "lm-studio",
            )
        )

        self.model_name = (
            model_name
            or os.getenv(
                "EXTRACTION_VERIFIER_MODEL_NAME",
                "qwen2.5-vl-3b-instruct",
            )
        )

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    # =========================================================
    # IMAGE ENCODING
    # =========================================================

    @staticmethod
    def _image_to_data_uri(
        image_path: str | Path,
    ) -> str:
        """
        Load the original page image and convert it into an
        in-memory JPEG data URI.

        The JPEG is NOT written to disk.
        """

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Page image not found: {path}"
            )

        image = Image.open(path).convert("RGB")

        image.thumbnail(
            (2000, 2000),
            Image.Resampling.LANCZOS,
        )

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=90,
            optimize=True,
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return (
            "data:image/jpeg;base64,"
            + encoded
        )

    # =========================================================
    # VERIFY
    # =========================================================

    def verify(
        self,
        image_path: str | Path,
        extraction: PageExtractionResult,
        attempt: int = 1,
    ) -> ExtractionVerificationResult:
        """
        Verify one extraction result against the original page.
        """

        if attempt < 1:
            raise ValueError(
                "attempt must be >= 1"
            )

        image_data_uri = (
            self._image_to_data_uri(
                image_path
            )
        )

        extracted_payload = (
            extraction.model_dump()
        )

        prompt = self._build_prompt(
            page_no=extraction.page_no,
            extraction=extracted_payload,
            attempt=attempt,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an independent verification "
                        "model for handwritten answer extraction. "
                        "Never invent content. Never grade the "
                        "student answer."
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

        raw_text = (
            response.choices[0]
            .message
            .content
            or ""
        ).strip()

        parsed = self._parse_json(
            raw_text
        )

        return self._normalize_result(
            parsed=parsed,
            page_no=extraction.page_no,
            attempt=attempt,
        )

    # =========================================================
    # PROMPT
    # =========================================================

    @staticmethod
    def _build_prompt(
        page_no: int,
        extraction: dict[str, Any],
        attempt: int,
    ) -> str:
        extraction_json = json.dumps(
            extraction,
            indent=2,
            ensure_ascii=False,
        )

        return f"""
You are verifying the output of another AI model.

This is page {page_no} of a handwritten student answer sheet.

The original page image is attached.

Below is the extraction produced by the FIRST model:

{extraction_json}

Your job is ONLY to verify whether the extraction faithfully
represents what is visibly written on the attached page.

Do NOT grade the student.
Do NOT determine whether the student's answer is correct.
Do NOT rewrite the extraction as a corrected final answer.

CHECK:

1. Question number:
   - Is the question number actually visible?
   - Was it associated with the correct answer?
   - Is the number likely misread?

2. Answer:
   - Is the extracted answer actually visible?
   - Is text missing?
   - Is text hallucinated?
   - Is the answer incorrectly transcribed?
   - Is an answer attached to the wrong question?

3. Structure:
   - Are multiple questions merged?
   - Is one question duplicated?
   - Is an answer assigned to the wrong question?
   - Is important handwritten content missing?

4. Unclear handwriting:
   - If the page is genuinely unreadable, mark it accordingly.
   - Do not invent a correction.

QUESTION NUMBER LOCATION IS NOT FIXED.
A student may write the question number:
- on the left
- above the answer
- on the same line
- inside a circle
- underlined
- near the answer
- in another natural location

Return ONLY valid JSON.

Use exactly this structure:

{{
  "page_no": {page_no},
  "passed": true,
  "confidence": 0.95,
  "issues": [],
  "feedback": null,
  "warnings": []
}}

For an incorrect extraction:

{{
  "page_no": {page_no},
  "passed": false,
  "confidence": 0.82,
  "issues": [
    {{
      "question_no": 4,
      "issue_type": "wrong_answer_transcription",
      "severity": "high",
      "message": "The extracted answer does not match the handwriting."
    }}
  ],
  "feedback": "Re-extract Question 4 carefully from the original page.",
  "warnings": []
}}

Attempt number: {attempt}
"""

    # =========================================================
    # JSON PARSING
    # =========================================================

    @staticmethod
    def _parse_json(
        raw_text: str,
    ) -> dict[str, Any]:
        text = raw_text.strip()

        if text.startswith("```"):
            lines = text.splitlines()

            if (
                lines
                and lines[0].strip().startswith("```")
            ):
                lines = lines[1:]

            if (
                lines
                and lines[-1].strip() == "```"
            ):
                lines = lines[:-1]

            text = "\n".join(
                lines
            ).strip()

        try:
            parsed = json.loads(text)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Extraction verifier returned invalid JSON."
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "Extraction verifier must return a JSON object."
            )

        return parsed

    # =========================================================
    # NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_result(
        parsed: dict[str, Any],
        page_no: int,
        attempt: int,
    ) -> ExtractionVerificationResult:
        issues: list[
            ExtractionVerificationIssue
        ] = []

        raw_issues = parsed.get(
            "issues",
            [],
        )

        if isinstance(
            raw_issues,
            list,
        ):
            for item in raw_issues:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                question_no = item.get(
                    "question_no"
                )

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

                issue_type = str(
                    item.get(
                        "issue_type",
                        "verification_issue",
                    )
                ).strip()

                severity = str(
                    item.get(
                        "severity",
                        "medium",
                    )
                ).strip().lower()

                if severity not in {
                    "low",
                    "medium",
                    "high",
                }:
                    severity = "medium"

                message = str(
                    item.get(
                        "message",
                        "Extraction issue detected.",
                    )
                ).strip()

                issues.append(
                    ExtractionVerificationIssue(
                        question_no=question_no,
                        issue_type=issue_type
                        or "verification_issue",
                        severity=severity,
                        message=message
                        or "Extraction issue detected.",
                    )
                )

        confidence_raw = parsed.get(
            "confidence",
            0.0,
        )

        try:
            confidence = float(
                confidence_raw
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        feedback_raw = parsed.get(
            "feedback"
        )

        feedback = (
            str(feedback_raw).strip()
            if feedback_raw
            else None
        )

        warnings_raw = parsed.get(
            "warnings",
            [],
        )

        warnings: list[str] = []

        if isinstance(
            warnings_raw,
            list,
        ):
            warnings = [
                str(item).strip()
                for item in warnings_raw
                if str(item).strip()
            ]

        passed = bool(
            parsed.get(
                "passed",
                False,
            )
        )

        return ExtractionVerificationResult(
            page_no=page_no,
            passed=passed,
            confidence=confidence,
            issues=issues,
            feedback=feedback,
            warnings=warnings,
            attempt=attempt,
        )