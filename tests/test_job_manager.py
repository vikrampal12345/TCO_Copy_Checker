from pathlib import Path

from app.services.job_manager import JobManager


def test_create_student_2_job() -> None:
    pdf_path = Path("data/test_copies/Student_2.pdf")

    manager = JobManager()

    job = manager.create_job(
        pdf_path=pdf_path,
        student_id="TEST_STUDENT_001",
        assessment_id="TEST_ASSESSMENT_001",
        subject="Physics",
        class_level="12",
    )

    assert job.total_pages == 4
    assert len(job.pages) == 4

    for index, page in enumerate(job.pages, start=1):
        assert page.page_number == index
        assert page.image_path is not None
        assert Path(page.image_path).exists()

    print("\nJob ID:", job.job_id)
    print("Total pages:", job.total_pages)

    for page in job.pages:
        print(
            f"Page {page.page_number}: "
            f"{page.status} -> {page.image_path}"
        )