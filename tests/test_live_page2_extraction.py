from app.services.in_memory_answer_extraction_service import (
    InMemoryAnswerExtractionService,
)
from app.services.in_memory_page_processor import (
    InMemoryPageProcessor,
)


PDF_PATH = r"data\test_copies\Student_23.pdf"
PAGE_NO = 2


def test_live_page2_extraction_only():

    print("\n" + "=" * 100)
    print("TCO COPY CHECKER")
    print("PAGE 2 WHOLE-PAGE EXTRACTION ONLY")
    print("=" * 100)
    print(f"PDF: {PDF_PATH}")
    print(f"Page under test: {PAGE_NO}")

    processor = InMemoryPageProcessor()
    extractor = InMemoryAnswerExtractionService()

    print(
        f"Extraction model: "
        f"{extractor.vision_service.get_model_name()}"
    )

    image = processor.render_page(
        PDF_PATH,
        PAGE_NO,
    )

    try:
        result = extractor.extract_page_in_memory(
            image=image,
            page_no=PAGE_NO,
            expected_question_numbers=[],
            enable_recovery=False,
            feedback=None,
        )

        print("\n" + "=" * 100)
        print("PAGE 2 EXTRACTED CONTENT")
        print("=" * 100)

        for item in result.questions:
            print(
                f"\nQ{item.question_no}"
                f" | confidence={item.confidence:.2f}"
                f" | status={item.status}"
            )
            print(item.answer)

        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print(
            f"Questions extracted: "
            f"{len(result.questions)}"
        )
        print(
            f"Unresolved items: "
            f"{len(result.unresolved_items)}"
        )
        print(
            f"Review required: "
            f"{result.review_required}"
        )

        assert len(result.questions) > 0

    finally:
        image.close()
