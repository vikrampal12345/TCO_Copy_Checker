from PIL import Image

from app.schemas.submission import (
    AggregatedAnswer,
    AggregatedSubmission,
)
from app.services.submission_verifier_model import (
    SubmissionVerifierModel,
)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self):
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs

        return FakeResponse(
            """
            {
              "passed": true,
              "confidence": 0.94,
              "issues": [],
              "feedback": null,
              "warnings": []
            }
            """
        )


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


class FakeProvider:
    def __init__(self):
        self.client = FakeClient()

    def get_client(self):
        return self.client

    def get_vision_model(self):
        return "fake-extraction-model"

    def get_extraction_verifier_model(self):
        return "fake-verifier-model"


def make_submission():
    return AggregatedSubmission(
        job_id="JOB001",
        student_number="STU047",
        total_pages=2,
        processed_pages=2,
        answers=[
            AggregatedAnswer(
                question_no=1,
                answer="The answer to question one.",
                confidence=0.92,
                status="EXTRACTED",
            ),
            AggregatedAnswer(
                question_no=2,
                answer="The answer to question two.",
                confidence=0.88,
                status="EXTRACTED",
            ),
        ],
        expected_questions=[1, 2],
        missing_questions=[],
        repeated_questions=[],
        conflicting_questions=[],
        unresolved_items=0,
        review_required=False,
    )


def make_page_image():
    return Image.new(
        "RGB",
        (100, 100),
        "white",
    )


def test_verifier_model_returns_structured_result():
    provider = FakeProvider()

    model = SubmissionVerifierModel(provider=provider)

    result = model.verify(
        make_submission(),
        attempt=1,
        page_images=[
            (1, make_page_image()),
        ],
    )

    assert result.passed is True
    assert result.confidence == 0.94
    assert result.attempt == 1
    assert result.issues == []


def test_verifier_model_sends_correct_model_and_temperature():
    provider = FakeProvider()

    model = SubmissionVerifierModel(provider=provider)

    model.verify(
        make_submission(),
        attempt=2,
        page_images=[
            (1, make_page_image()),
        ],
        temperature=0,
    )

    request = provider.client.chat.completions.last_request

    assert request["model"] == "fake-verifier-model"
    assert request["temperature"] == 0
    assert len(request["messages"]) == 2


def test_verifier_model_request_contains_original_page_image():
    provider = FakeProvider()

    model = SubmissionVerifierModel(provider=provider)

    model.verify(
        make_submission(),
        attempt=1,
        page_images=[
            (1, make_page_image()),
            (2, make_page_image()),
        ],
    )

    request = provider.client.chat.completions.last_request
    user_content = request["messages"][1]["content"]

    image_items = [
        item
        for item in user_content
        if item["type"] == "image_url"
    ]

    assert len(image_items) == 2

    for item in image_items:
        assert "image_url" in item
        assert item["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )


def test_verifier_model_prompt_contains_submission_details():
    submission = make_submission()
    provider = FakeProvider()

    model = SubmissionVerifierModel(provider=provider)

    model.verify(
        submission,
        attempt=1,
        page_images=[
            (1, make_page_image()),
        ],
    )

    request = provider.client.chat.completions.last_request
    user_content = request["messages"][1]["content"]

    text_items = [
        item["text"]
        for item in user_content
        if item["type"] == "text"
    ]

    combined_text = "\n".join(text_items)

    assert "STU047" in combined_text
    assert "JOB001" in combined_text
    assert "question_no" in combined_text
    assert "The answer to question one." in combined_text
    assert "Original answer-sheet page 1" in combined_text


def test_parse_json_response_handles_markdown_fence():
    raw = """
    ```json
    {
      "passed": false,
      "confidence": 0.55,
      "issues": [
        {
          "question_no": 3,
          "issue_type": "extraction_error",
          "severity": "high",
          "message": "Answer is incomplete."
        }
      ],
      "feedback": "Re-extract Q3.",
      "warnings": []
    }
    ```
    """

    parsed = SubmissionVerifierModel._parse_json_response(raw)

    assert parsed["passed"] is False
    assert parsed["confidence"] == 0.55
    assert parsed["issues"][0]["question_no"] == 3
    assert parsed["feedback"] == "Re-extract Q3."


def test_invalid_json_raises_value_error():
    try:
        SubmissionVerifierModel._parse_json_response(
            "this is not json"
        )
    except ValueError as exc:
        assert "invalid JSON" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for invalid JSON"
        )
