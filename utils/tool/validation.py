import logging

logger = logging.getLogger("extraction")

import operator
import uuid
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
)
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ValidationNode

from pydantic import BaseModel, Field, field_validator


# https://langchain-ai.github.io/langgraph/tutorials/extraction/retries/#define-the-validator-retry-graph

def _default_aggregator(messages: Sequence[AnyMessage]) -> AIMessage:
    for m in messages[::-1]:
        if m.type == "ai":
            return m
    raise ValueError("No AI message found in the sequence.")


class RetryStrategy(TypedDict, total=False):
    """The retry strategy for a tool call."""

    max_attempts: int
    """The maximum number of attempts to make."""
    fallback: Optional[
        Union[
            Runnable[Sequence[AnyMessage], AIMessage],
            Runnable[Sequence[AnyMessage], BaseMessage],
            Callable[[Sequence[AnyMessage]], AIMessage],
        ]
    ]
    """The function to use once validation fails."""
    aggregate_messages: Optional[Callable[[Sequence[AnyMessage]], AIMessage]]


def _bind_validator_with_retries(
    llm: Union[
        Runnable[Sequence[AnyMessage], AIMessage],
        Runnable[Sequence[BaseMessage], BaseMessage],
    ],
    *,
    validator: ValidationNode,
    retry_strategy: RetryStrategy,
    tool_choice: Optional[str] = None,
) -> Runnable[Union[List[AnyMessage], PromptValue], AIMessage]:
    """Binds a tool validators + retry logic to create a runnable validation graph.

    LLMs that support tool calling can generate structured JSON. However, they may not always
    perfectly follow your requested schema, especially if the schema is nested or has complex
    validation rules. This method allows you to bind a validation function to the LLM's output,
    so that any time the LLM generates a message, the validation function is run on it. If
    the validation fails, the method will retry the LLM with a fallback strategy, the simplest
    being just to add a message to the output with the validation errors and a request to fix them.

    The resulting runnable expects a list of messages as input and returns a single AI message.
    By default, the LLM can optionally NOT invoke tools, making this easier to incorporate into
    your existing chat bot. You can specify a tool_choice to force the validator to be run on
    the outputs.

    Args:
        llm (Runnable): The llm that will generate the initial messages (and optionally fallba)
        validator (ValidationNode): The validation logic.
        retry_strategy (RetryStrategy): The retry strategy to use.
            Possible keys:
            - max_attempts: The maximum number of attempts to make.
            - fallback: The LLM or function to use in case of validation failure.
            - aggregate_messages: A function to aggregate the messages over multiple turns.
                Defaults to fetching the last AI message.
        tool_choice: If provided, always run the validator on the tool output.

    Returns:
        Runnable: A runnable that can be invoked with a list of messages and returns a single AI message.
    """

    def add_or_overwrite_messages(left: list, right: Union[list, dict]) -> list:
        """Append messages. If the update is a 'finalized' output, replace the whole list."""
        if isinstance(right, dict) and "finalize" in right:
            finalized = right["finalize"]
            if not isinstance(finalized, list):
                finalized = [finalized]
            for m in finalized:
                if m.id is None:
                    m.id = str(uuid.uuid4())
            return finalized
        res = add_messages(left, right)
        if not isinstance(res, list):
            return [res]
        return res

    class State(TypedDict):
        messages: Annotated[list, add_or_overwrite_messages]
        attempt_number: Annotated[int, operator.add]
        initial_num_messages: int
        input_format: Literal["list", "dict"]

    builder = StateGraph(State)

    def dedict(x: State) -> list:
        """Get the messages from the state."""
        return x["messages"]

    model = dedict | llm | (lambda msg: {"messages": [msg], "attempt_number": 1})
    fbrunnable = retry_strategy.get("fallback")
    if fbrunnable is None:
        fb_runnable = llm
    elif isinstance(fbrunnable, Runnable):
        fb_runnable = fbrunnable  # type: ignore
    else:
        fb_runnable = RunnableLambda(fbrunnable)
    fallback = (
        dedict | fb_runnable | (lambda msg: {"messages": [msg], "attempt_number": 1})
    )

    def count_messages(state: State) -> dict:
        return {"initial_num_messages": len(state.get("messages", []))}

    builder.add_node("count_messages", count_messages)
    builder.add_node("llm", model)
    builder.add_node("fallback", fallback)

    # To support patch-based retries, we need to be able to
    # aggregate the messages over multiple turns.
    # The next sequence selects only the relevant messages
    # and then applies the validator
    select_messages = retry_strategy.get("aggregate_messages") or _default_aggregator

    def select_generated_messages(state: State) -> list:
        """Select only the messages generated within this loop."""
        selected = state["messages"][state["initial_num_messages"] :]
        return [select_messages(selected)]

    def endict_validator_output(x: Sequence[AnyMessage]) -> dict:
        if tool_choice and not x:
            return {
                "messages": [
                    HumanMessage(
                        content=f"ValidationError: please respond with a valid tool call [tool_choice={tool_choice}].",
                        additional_kwargs={"is_error": True},
                    )
                ]
            }
        return {"messages": x}

    validator_runnable = select_generated_messages | validator | endict_validator_output
    builder.add_node("validator", validator_runnable)

    class Finalizer:
        """Pick the final message to return from the retry loop."""

        def __init__(self, aggregator: Optional[Callable[[list], AIMessage]] = None):
            self._aggregator = aggregator or _default_aggregator

        def __call__(self, state: State) -> dict:
            """Return just the AI message."""
            initial_num_messages = state["initial_num_messages"]
            generated_messages = state["messages"][initial_num_messages:]
            return {
                "messages": {
                    "finalize": self._aggregator(generated_messages),
                }
            }

    # We only want to emit the final message
    builder.add_node("finalizer", Finalizer(retry_strategy.get("aggregate_messages")))

    # Define the connectivity
    builder.add_edge(START, "count_messages")
    builder.add_edge("count_messages", "llm")

    def route_validator(state: State):
        if state["messages"][-1].tool_calls or tool_choice is not None:
            return "validator"
        return END

    builder.add_conditional_edges("llm", route_validator, ["validator", END])
    builder.add_edge("fallback", "validator")
    max_attempts = retry_strategy.get("max_attempts", 3)

    def route_validation(state: State):
        if state["attempt_number"] > max_attempts:
            raise ValueError(
                f"Could not extract a valid value in {max_attempts} attempts."
            )
        for m in state["messages"][::-1]:
            if m.type == "ai":
                break
            if m.additional_kwargs.get("is_error"):
                return "fallback"
        return "finalizer"

    builder.add_conditional_edges(
        "validator", route_validation, ["finalizer", "fallback"]
    )

    builder.add_edge("finalizer", END)

    # These functions let the step be used in a MessageGraph
    # or a StateGraph with 'messages' as the key.
    def encode(x: Union[Sequence[AnyMessage], PromptValue]) -> dict:
        """Ensure the input is the correct format."""
        if isinstance(x, PromptValue):
            return {"messages": x.to_messages(), "input_format": "list"}
        if isinstance(x, list):
            return {"messages": x, "input_format": "list"}
        raise ValueError(f"Unexpected input type: {type(x)}")

    def decode(x: State) -> AIMessage:
        """Ensure the output is in the expected format."""
        return x["messages"][-1]

    return (
        encode | builder.compile().with_config(run_name="ValidationGraph") | decode
    ).with_config(run_name="ValidateWithRetries")


def bind_validator_with_retries(
    llm: BaseChatModel,
    *,
    tools: list,
    tool_choice: Optional[str] = None,
    max_attempts: int = 3,
) -> Runnable[Union[List[AnyMessage], PromptValue], AIMessage]:
    """Binds validators + retry logic ensure validity of generated tool calls.

    LLMs that support tool calling are good at generating structured JSON. However, they may
    not always perfectly follow your requested schema, especially if the schema is nested or
    has complex validation rules. This method allows you to bind a validation function to
    the LLM's output, so that any time the LLM generates a message, the validation function
    is run on it. If the validation fails, the method will retry the LLM with a fallback
    strategy, the simples being just to add a message to the output with the validation
    errors and a request to fix them.

    The resulting runnable expects a list of messages as input and returns a single AI message.
    By default, the LLM can optionally NOT invoke tools, making this easier to incorporate into
    your existing chat bot. You can specify a tool_choice to force the validator to be run on
    the outputs.

    Args:
        llm (Runnable): The llm that will generate the initial messages (and optionally fallba)
        validator (ValidationNode): The validation logic.
        retry_strategy (RetryStrategy): The retry strategy to use.
            Possible keys:
            - max_attempts: The maximum number of attempts to make.
            - fallback: The LLM or function to use in case of validation failure.
            - aggregate_messages: A function to aggregate the messages over multiple turns.
                Defaults to fetching the last AI message.
        tool_choice: If provided, always run the validator on the tool output.

    Returns:
        Runnable: A runnable that can be invoked with a list of messages and returns a single AI message.
    """
    bound_llm = llm.bind_tools(tools, tool_choice=tool_choice)
    retry_strategy = RetryStrategy(max_attempts=max_attempts)
    validator = ValidationNode(tools)
    return _bind_validator_with_retries(
        bound_llm,
        validator=validator,
        tool_choice=tool_choice,
        retry_strategy=retry_strategy,
    ).with_config(metadata={"retry_strategy": "default"})

def bind_validator_with_jsonpatch_retries(
    llm: BaseChatModel,
    *,
    tools: list,
    tool_choice: Optional[str] = None,
    max_attempts: int = 3,
) -> Runnable[Union[List[AnyMessage], PromptValue], AIMessage]:
    """Binds validators + retry logic ensure validity of generated tool calls.

    This method is similar to `bind_validator_with_retries`, but uses JSONPatch to correct
    validation errors caused by passing in incorrect or incomplete parameters in a previous
    tool call. This method requires the 'jsonpatch' library to be installed.

    Using patch-based function healing can be more efficient than repopulating the entire
    tool call from scratch, and it can be an easier task for the LLM to perform, since it typically
    only requires a few small changes to the existing tool call.

    Args:
        llm (Runnable): The llm that will generate the initial messages (and optionally fallba)
        tools (list): The tools to bind to the LLM.
        tool_choice (Optional[str]): The tool choice to use.
        max_attempts (int): The number of attempts to make.

    Returns:
        Runnable: A runnable that can be invoked with a list of messages and returns a single AI message.
    """

    try:
        import jsonpatch  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The 'jsonpatch' library is required for JSONPatch-based retries."
        )

    class JsonPatch(BaseModel):
        """A JSON Patch document represents an operation to be performed on a JSON document.

        Note that the op and path are ALWAYS required. Value is required for ALL operations except 'remove'.
        Examples:

        \`\`\`json
        {"op": "add", "path": "/a/b/c", "patch_value": 1}
        {"op": "replace", "path": "/a/b/c", "patch_value": 2}
        {"op": "remove", "path": "/a/b/c"}
        \`\`\`
        """

        op: Literal["add", "remove", "replace"] = Field(
            ...,
            description="The operation to be performed. Must be one of 'add', 'remove', 'replace'.",
        )
        path: str = Field(
            ...,
            description="A JSON Pointer path that references a location within the target document where the operation is performed.",
        )
        value: Any = Field(
            ...,
            description="The value to be used within the operation. REQUIRED for 'add', 'replace', and 'test' operations.",
        )

    class PatchFunctionParameters(BaseModel):
        """Respond with all JSONPatch operation to correct validation errors caused by passing in incorrect or incomplete parameters in a previous tool call."""

        tool_call_id: str = Field(
            ...,
            description="The ID of the original tool call that generated the error. Must NOT be an ID of a PatchFunctionParameters tool call.",
        )
        reasoning: str = Field(
            ...,
            description="Think step-by-step, listing each validation error and the"
            " JSONPatch operation needed to correct it. "
            "Cite the fields in the JSONSchema you referenced in developing this plan.",
        )
        patches: list[JsonPatch] = Field(
            ...,
            description="A list of JSONPatch operations to be applied to the previous tool call's response.",
        )

    bound_llm = llm.bind_tools(tools, tool_choice=tool_choice)
    fallback_llm = llm.bind_tools([PatchFunctionParameters])

    def aggregate_messages(messages: Sequence[AnyMessage]) -> AIMessage:
        # Get all the AI messages and apply json patches
        resolved_tool_calls: Dict[Union[str, None], ToolCall] = {}
        content: Union[str, List[Union[str, dict]]] = ""
        for m in messages:
            if m.type != "ai":
                continue
            if not content:
                content = m.content
            for tc in m.tool_calls:
                if tc["name"] == PatchFunctionParameters.__name__:
                    tcid = tc["args"]["tool_call_id"]
                    if tcid not in resolved_tool_calls:
                        logger.debug(
                            f"JsonPatch tool call ID {tc['args']['tool_call_id']} not found."
                            f"Valid tool call IDs: {list(resolved_tool_calls.keys())}"
                        )
                        tcid = next(iter(resolved_tool_calls.keys()), None)
                    orig_tool_call = resolved_tool_calls[tcid]
                    current_args = orig_tool_call["args"]
                    patches = tc["args"].get("patches") or []
                    orig_tool_call["args"] = jsonpatch.apply_patch(
                        current_args,
                        patches,
                    )
                    orig_tool_call["id"] = tc["id"]
                else:
                    resolved_tool_calls[tc["id"]] = tc.copy()
        return AIMessage(
            content=content,
            tool_calls=list(resolved_tool_calls.values()),
        )

    def format_exception(error: BaseException, call: ToolCall, schema: Type[BaseModel]):
        return (
            f"Error:\n\n\`\`\`\n{repr(error)}\n\`\`\`\n"
            "Expected Parameter Schema:\n\n" + f"\`\`\`json\n{schema.schema_json()}\n\`\`\`\n"
            f"Please respond with a JSONPatch to correct the error for tool_call_id=[{call['id']}]."
        )

    validator = ValidationNode(
        tools + [PatchFunctionParameters],
        format_error=format_exception,
    )
    retry_strategy = RetryStrategy(
        max_attempts=max_attempts,
        fallback=fallback_llm,
        aggregate_messages=aggregate_messages,
    )
    return _bind_validator_with_retries(
        bound_llm,
        validator=validator,
        retry_strategy=retry_strategy,
        tool_choice=tool_choice,
    ).with_config(metadata={"retry_strategy": "jsonpatch"})