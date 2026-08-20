"""T-3 plus schema derivation. No DB, no Redis, no LLM — pure and fast."""
import json

import pytest

from app.agent.tools import TOOL_REGISTRY, ToolContext, calculator, dispatch, tool


def test_registry_holds_the_five_tools():
    assert {
        "web_search",
        "calculator",
        "get_current_datetime",
        "summarise_text",
        "remember_fact",
    } <= set(TOOL_REGISTRY)


def test_schema_is_derived_from_signature_and_docstring():
    schema = TOOL_REGISTRY["summarise_text"].openai_schema()
    fn = schema["function"]

    assert fn["name"] == "summarise_text"
    assert "Summarise" in fn["description"]
    assert fn["parameters"]["properties"]["text"] == {"type": "string"}
    assert fn["parameters"]["properties"]["max_words"] == {"type": "integer", "default": 50}
    # `thought` first, then the parameters without a default, then `grounding`
    assert fn["parameters"]["required"] == ["thought", "text", "grounding"]


def test_injected_ctx_is_never_visible_to_the_llm():
    schema = TOOL_REGISTRY["remember_fact"].openai_schema()
    properties = schema["function"]["parameters"]["properties"]
    assert "fact" in properties
    assert "ctx" not in properties


def test_adding_a_tool_requires_only_the_function():
    @tool
    def get_weather(city: str) -> str:
        """Return the current weather for a city."""
        return f"{city}: 34C, clear skies."

    try:
        schema = TOOL_REGISTRY["get_weather"].openai_schema()
        # the reasoning parameters come for free — still nothing written by hand
        assert schema["function"]["parameters"]["required"] == ["thought", "city", "grounding"]
        assert schema["function"]["description"] == "Return the current weather for a city."
    finally:
        TOOL_REGISTRY.pop("get_weather")


def test_calculator_evaluates_arithmetic():
    # `ast.literal_eval` cannot do this at all — see the deviation note in NOTES.md.
    assert calculator("0.15 * 3787000") == "568050.0"
    assert calculator("2 + 2 * 3") == "8"


def test_calculator_rejects_non_arithmetic_nodes():
    with pytest.raises(ValueError):
        calculator("__import__('os').system('ls')")


def test_web_search_snippet_carries_a_number():
    # The reference scenario needs the model to extract a figure and feed it to
    # the calculator; a prose-only mock would fail the demo for unrelated reasons.
    results = json.loads(TOOL_REGISTRY["web_search"].fn("Dubai"))
    assert any("3,787,000" in r["snippet"] for r in results)


# --------------------------- T-3: dispatcher ------------------------------- #
async def test_unknown_tool_returns_error_string_not_a_raise():
    result = await dispatch("nonexistent_tool", "{}", None)
    assert result.startswith("Error: unknown tool 'nonexistent_tool'")
    assert "calculator" in result          # the model gets the valid list back


async def test_invalid_json_arguments_return_an_error_string():
    result = await dispatch("calculator", "{not json", None)
    assert result.startswith("Error: arguments were not valid JSON")


async def test_bad_arguments_return_an_error_string():
    result = await dispatch("calculator", '{"wrong_kwarg": 1}', None)
    assert result.startswith("Error: bad arguments for 'calculator'")


async def test_tool_exception_is_captured_as_an_error_string():
    result = await dispatch("calculator", '{"expression": "os.system(1)"}', None)
    assert result.startswith("Error executing 'calculator'")


async def test_async_tool_with_injected_context(memory):
    ctx = ToolContext(user_id="u1", run_id="r1", db=None, memory=memory)
    result = await dispatch("remember_fact", '{"fact": "Maya is my daughter"}', ctx)
    assert result == "Stored: Maya is my daughter"
    assert memory.facts == [("u1", "Maya is my daughter")]


# ------------------------ ReAct prompt composition -------------------------- #
def test_react_prompt_has_no_thought_action_scaffolding():
    """The 2022 text format would be parsed as prose, not as an action: the loop
    would see a message with no tool_calls and stop on iteration 1."""
    from app.agent.prompts import REACT_SYSTEM_PROMPT

    assert "Thought:" not in REACT_SYSTEM_PROMPT.replace('"Thought:"', "")
    assert "Observation:" not in REACT_SYSTEM_PROMPT
    for instruction in ["call a tool", "Error:", "plain text and no tool call"]:
        assert instruction in REACT_SYSTEM_PROMPT


def test_build_system_prompt_composes_persona_onto_the_instructions():
    from app.agent.prompts import REACT_SYSTEM_PROMPT, build_system_prompt

    assert build_system_prompt(None) == REACT_SYSTEM_PROMPT
    assert build_system_prompt("   ") == REACT_SYSTEM_PROMPT

    composed = build_system_prompt("You are terse.")
    assert composed.startswith(REACT_SYSTEM_PROMPT)
    assert composed.endswith("You are terse.")


# ------------------- reasoning parameters (thought / grounding) ------------- #
def test_thought_is_required_on_every_tool():
    for name, entry in TOOL_REGISTRY.items():
        params = entry.openai_schema()["function"]["parameters"]
        assert params["required"][0] == "thought", name
        assert params["properties"]["thought"]["type"] == "string", name


def test_grounding_is_required_only_where_there_are_values_to_ground():
    from app.agent.tools import GROUNDING_VALUES

    valued = TOOL_REGISTRY["calculator"].openai_schema()["function"]["parameters"]
    assert valued["properties"]["grounding"]["enum"] == GROUNDING_VALUES
    assert "grounding" in valued["required"]

    # get_current_datetime() takes nothing, so there is nothing to ground
    empty = TOOL_REGISTRY["get_current_datetime"].openai_schema()["function"]["parameters"]
    assert "grounding" not in empty["properties"]
    assert empty["required"] == ["thought"]


def test_parse_arguments_splits_reasoning_from_arguments():
    from app.agent.tools import parse_arguments

    parsed = parse_arguments('{"thought": "why", "grounding": "user", "expression": "2*21"}')
    assert parsed.thought == "why"
    assert parsed.grounding == "user"
    assert parsed.kwargs == {"expression": "2*21"}      # the tool never sees the rest
    assert parsed.error is None

    assert parse_arguments("{oops").error.startswith("Error: arguments were not valid JSON")
    assert parse_arguments("[1,2]").error.startswith("Error: arguments must be a JSON object")


async def test_invented_values_are_refused_before_the_tool_runs():
    """A model asked to 'add 3 numbers' with none given will compute 5 + 10 + 15
    unless something stops it. Self-reported invention is that something."""
    result = await dispatch(
        "calculator", '{"thought": "summing 5, 10 and 15", '
                      '"expression": "5 + 10 + 15", "grounding": "assumed"}'
    )
    assert result.startswith("Error: you marked these arguments as assumed")
    assert "invented values" in result


@pytest.mark.parametrize("grounding", ["user", "tool_result"])
async def test_grounded_calls_execute_normally(grounding):
    result = await dispatch(
        "calculator", '{"thought": "t", "expression": "12+30+7", "grounding": "%s"}' % grounding
    )
    assert result == "49"


async def test_a_call_without_grounding_is_not_blocked():
    """Only an explicit 'assumed' is refused: the schema asks the model for
    grounding, but a direct call (a test, another service) must still work."""
    assert await dispatch("calculator", '{"expression": "2*21"}') == "42"


# --------------------------- the ReAct prompt ------------------------------- #
def test_prompt_has_the_standard_react_structure():
    from app.agent.prompts import REACT_SYSTEM_PROMPT as prompt

    for section in ("## THE CYCLE", "## WHERE YOUR REASONING GOES", "## RULES",
                    "## WORKED EXAMPLE"):
        assert section in prompt, section
    for stage in ("REASON", "ACT", "OBSERVE"):
        assert stage in prompt, stage


def test_every_rule_is_present_and_numbered():
    """Pinned so a future rewrite cannot quietly drop a rule that was added to
    fix a specific observed failure."""
    from app.agent.prompts import REACT_SYSTEM_PROMPT as prompt

    expected = {
        "R1": "GROUND EVERY VALUE",
        "R2": "LOOK IT UP",
        "R3": "USE WHAT THE TOOL RETURNED",
        "R4": "CHAIN WHAT DEPENDS",
        "R5": "ERRORS ARE OBSERVATIONS",
        "R6": "DO NOT REPEAT YOURSELF",
        "R7": "FINISHING ENDS THE RUN",
        "R8": "YOU CANNOT ASK MID-RUN",
        "R9": "GROUND THE FINAL ANSWER",
    }
    for number, heading in expected.items():
        assert f"{number}. {heading}" in prompt, number


def test_prompt_explains_both_injected_parameters():
    from app.agent.prompts import REACT_SYSTEM_PROMPT as prompt

    assert "thought" in prompt and "grounding" in prompt
    assert "refused before the tool runs" in prompt


def test_worked_example_is_the_reference_scenario():
    from app.agent.prompts import REACT_SYSTEM_PROMPT as prompt

    assert "15% of the current population of Dubai" in prompt
    for tool_name in ("web_search", "calculator", "remember_fact"):
        assert tool_name in prompt, tool_name
    assert "0.15 * 3787000" in prompt
    # and the counter-example that motivated the grounding gate
    assert "5 + 10 + 15" in prompt


def test_prompt_carries_no_thought_action_scaffolding():
    """Requesting the 2022 text format from a tool-calling model makes it
    describe actions instead of taking them; the loop would stop on iteration 1."""
    from app.agent.prompts import REACT_SYSTEM_PROMPT as prompt

    assert "Thought:" not in prompt.replace('"Thought:"', "")
    assert "Observation:" not in prompt


def test_session_persona_is_appended_after_the_instructions():
    from app.agent.prompts import REACT_SYSTEM_PROMPT, build_system_prompt

    assert build_system_prompt(None) == REACT_SYSTEM_PROMPT
    assert build_system_prompt("  ") == REACT_SYSTEM_PROMPT

    composed = build_system_prompt("Answer in British English.")
    assert composed.startswith(REACT_SYSTEM_PROMPT)
    assert composed.endswith("Answer in British English.")
