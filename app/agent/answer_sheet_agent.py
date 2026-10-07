from app.agent.answer_sheet_evaluation_agent import (
    AnswerSheetEvaluationAgent,
)


# ------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------
#
# Existing code currently imports:
#
#     from app.agent.answer_sheet_agent import AnswerSheetAgent
#
# Keep that import working while the official agent name
# becomes:
#
#     AnswerSheetEvaluationAgent
#

AnswerSheetAgent = AnswerSheetEvaluationAgent