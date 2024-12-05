from pydantic import BaseModel, Field
from typing import Optional


class Action(BaseModel):
    """
    Base class for all actions that can be taken by the agent.
    """

    tool_call_metadata: Optional[dict] = None

    def get_tool_message(self):
        from aide.backend import provider_to_tool_message_func

        """converts an action into  a tool message format that can be sent to an LLM"""
        obj = self.__dict__.copy()
        obj.pop("tool_call_metadata")

        tool_message_func = provider_to_tool_message_func[
            self.tool_call_metadata["provider"]
        ]
        return tool_message_func(
            self.tool_call_metadata["tool_call_id"], self.__class__.__name__, obj
        )


class Draft(Action):
    """
    Action to draft new solutions to the problem. Use this action when user intends to draft a new solution
    or modify the existing approach to completely new approach.
    """

    user_feedback: str | None = Field(
        description="""
            feedback/message from the user that lead to this action. It should be reforumlated 
            such that it can be posed as a independent request without chat history to other members of the team.
            It can also be None, if there is no useful feedback from the user.
        """
    )


class Improve(Action):
    """
    Action to improve the current solution. Use this action when user intends to improve the current solution
    by modifying the existing approach by adding or removing components to the current solution or by changing
    the hyperparameters of the current solution. Do not use this action when the current solution is in
    buggy state. Meaning the current solution wasn't able to produce metric value.
    """

    user_feedback: str | None = Field(
        description="""
            feedback/message from the user that lead to this action. It should be reforumlated 
            such that it can be posed as a independent request without chat history to other members of the team.
            It can also be None, if there is no useful feedback from the user.
        """
    )


class Debug(Action):
    """
    Action to debug the current solution. Use this action when user intends to debug the current solution
    by identifying and fixing the bugs in the current solution. Use this action only when the current solution
    is in buggy state. Meaning the current solution wasn't able to produce metric value.
    """

    user_feedback: str | None = Field(
        description="""
            feedback/message from the user that lead to this action. It should be reforumlated 
            such that it can be posed as a independent request without chat history to other members of the team.
            It can also be None, if there is no useful feedback from the user.
        """
    )


class Finish(Action):
    """
    Action to stop the agent from further actions. Use this action when user intends to stop the
    agent or is satisfied with the current solution.
    """


class SubmitReview(Action):
    """Submit a review evaluating the output of the training script."""

    is_bug: bool = Field(
        description="true if the output log shows that the execution failed or has some bug, otherwise false."
    )
    has_csv_submission: bool = Field(
        description="true if the code saves the predictions on the test data"
    )
    summary: str = Field(
        description="write a short summary (2-3 sentences) describing the empirical findings. Alternatively mention if there is a bug or the submission.csv was not properly produced. DO NOT suggest fixes or improvements."
    )
    metric: float | None = Field(
        description="If the code ran successfully, report the value of the validation metric. Otherwise, leave it null."
    )
    lower_is_better: bool = Field(
        description="true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy)."
    )
    missing_libraries: list[str] | None = Field(
        description="list of libraries that are missing in the code execution environment if the node is buggy due to missing libraries. Please only give out this list when the node is buggy due to missing libraries else None or empty list."
    )
