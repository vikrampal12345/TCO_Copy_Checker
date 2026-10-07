from __future__ import annotations

from collections import Counter

from app.schemas.marking_scheme import MarkingScheme


class MarkingSchemeValidationError(ValueError):
    """Raised when a marking scheme is structurally invalid."""


class MarkingSchemeService:
    """
    Deterministic validation for marking schemes.

    This service does NOT generate the rubric.
    Generation will be handled by the Answer Sheet
    Evaluation workflow / model layer.

    This service only validates the result.
    """

    @staticmethod
    def validate(scheme: MarkingScheme) -> MarkingScheme:
        """
        Validate and return the same scheme if valid.
        """

        if not scheme.questions:
            raise MarkingSchemeValidationError(
                "Marking scheme must contain at least one question."
            )

        # ----------------------------------------------------
        # Duplicate question numbers
        # ----------------------------------------------------

        question_numbers = [
            question.question_no
            for question in scheme.questions
        ]

        duplicates = [
            question_no
            for question_no, count in Counter(
                question_numbers
            ).items()
            if count > 1
        ]

        if duplicates:
            raise MarkingSchemeValidationError(
                "Duplicate question number(s): "
                + ", ".join(map(str, sorted(duplicates)))
            )

        # ----------------------------------------------------
        # Validate each question
        # ----------------------------------------------------

        calculated_total = 0.0

        for question in scheme.questions:

            criteria_total = sum(
                criterion.marks
                for criterion in question.criteria
            )

            # Criteria cannot exceed question marks.
            if criteria_total > question.max_marks:
                raise MarkingSchemeValidationError(
                    f"Q{question.question_no}: criterion marks "
                    f"({criteria_total}) exceed question max "
                    f"marks ({question.max_marks})."
                )

            calculated_total += question.max_marks

        # ----------------------------------------------------
        # Total marks must match question totals
        # ----------------------------------------------------

        if abs(
            calculated_total - scheme.total_marks
        ) > 1e-6:
            raise MarkingSchemeValidationError(
                "Assessment total marks do not match the "
                "sum of question maximum marks. "
                f"Expected {calculated_total}, "
                f"got {scheme.total_marks}."
            )

        return scheme

    # ========================================================
    # STATE HELPERS
    # ========================================================

    @staticmethod
    def approve(
        scheme: MarkingScheme,
    ) -> MarkingScheme:
        """
        Teacher-approved scheme.
        """

        scheme.status = "approved"
        return scheme

    @staticmethod
    def lock(
        scheme: MarkingScheme,
    ) -> MarkingScheme:
        """
        Lock an approved marking scheme.

        Only an approved scheme can be locked.
        """

        if scheme.status != "approved":
            raise MarkingSchemeValidationError(
                "Only an approved marking scheme can be locked."
            )

        scheme.status = "locked"
        return scheme

    @staticmethod
    def reject(
        scheme: MarkingScheme,
        teacher_feedback: str,
    ) -> MarkingScheme:
        """
        Mark a proposed scheme as rejected and preserve
        teacher feedback for regeneration.
        """

        feedback = teacher_feedback.strip()

        if not feedback:
            raise MarkingSchemeValidationError(
                "Teacher feedback is required when rejecting "
                "a marking scheme."
            )

        scheme.status = "rejected"
        scheme.teacher_feedback = feedback
        scheme.version += 1

        return scheme