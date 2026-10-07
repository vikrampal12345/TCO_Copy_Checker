from pathlib import Path

from app.services.page_service import PageService


def test_render_student_2_pages() -> None:
    pdf_path = Path("data/test_copies/Student_2.pdf")
    output_dir = Path("data/pages/Student_2")

    service = PageService()

    pages = service.render_pages(
        pdf_path=pdf_path,
        output_dir=output_dir,
    )

    print(f"Rendered pages: {len(pages)}")

    for page in pages:
        print(page)


if __name__ == "__main__":
    test_render_student_2_pages()