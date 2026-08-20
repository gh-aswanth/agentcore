"""Coherence tests for the standalone examples.

These guard one specific bug that actually shipped: `ask_user` was registered as
a tool in `react_tool_calling.py` while its system prompt said nothing about
asking, so the model was free to assume a missing detail and carry on. A tool the
prompt never mentions is a tool the model will not reach for — and a prompt that
promises a tool that was withheld produces calls to something that is not there.
Both directions are asserted here.

Skipped when `examples/` is absent, which is the case inside the runtime image.
"""
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if not EXAMPLES.is_dir():                                     # pragma: no cover
    pytest.skip("examples/ not present", allow_module_level=True)

sys.path.insert(0, str(EXAMPLES))

import react_scratchpad as scratchpad        # noqa: E402
import react_tool_calling as tool_calling    # noqa: E402


# --------------------------- native tool calling ---------------------------- #
def test_tool_calling_prompt_tells_the_model_to_ask_when_it_can():
    prompt = tool_calling.build_system_prompt(interactive=True)
    assert "ask_user" in prompt
    # the rule is stated as a provenance principle, not a list of examples to
    # pattern-match against — that enumeration is what let "add 3 numbers" slip
    assert "the user's own words, or an earlier tool result" in prompt
    assert "you invented it" in prompt
    assert "never proceed on" in prompt


def test_tool_calling_prompt_never_promises_ask_user_when_withheld():
    prompt = tool_calling.build_system_prompt(interactive=False)
    assert "ask_user" not in prompt
    # still forbidden from inventing the missing piece — it must say what it needs
    assert "the user's own words, or an earlier tool result" in prompt
    assert "no one to ask in this session" in prompt


def test_every_tool_requires_a_thought():
    for name, entry in tool_calling.REGISTRY.items():
        schema = entry.schema()["function"]["parameters"]
        assert schema["required"][0] == "thought", name
        assert "thought" in schema["properties"], name


def test_thought_is_stripped_before_the_tool_runs():
    call = tool_calling.parse_call('{"thought": "why I am calling", "query": "Dubai"}')
    assert call.thought == "why I am calling"
    assert call.kwargs == {"query": "Dubai"}          # the tool never sees it


def test_strict_schemas_satisfy_structured_outputs_rules():
    for name, entry in tool_calling.REGISTRY.items():
        function = entry.schema(strict=True)["function"]
        params = function["parameters"]
        assert function["strict"] is True, name
        assert params["additionalProperties"] is False, name
        # strict mode requires every property to be listed as required...
        assert set(params["required"]) == set(params["properties"]), name
        # ...so an optional parameter has to be expressed as a nullable type
        for optional in entry.optional:
            assert "null" in params["properties"][optional]["type"], (name, optional)


def test_strict_nulls_do_not_reach_the_tool():
    """Under --strict the model sends `units: null` for an omitted optional; the
    function's own default must still apply."""
    call = tool_calling.parse_call('{"thought": "t", "city": "Dubai", "units": null}')
    assert call.kwargs == {"city": "Dubai"}
    assert "34C" in tool_calling.execute("get_weather", call.kwargs)


# ------------------------- the do-not-assume gate --------------------------- #
def test_value_carrying_tools_must_declare_where_the_values_came_from():
    for name, entry in tool_calling.REGISTRY.items():
        properties = entry.schema()["function"]["parameters"]["properties"]
        takes_values = len([p for p in properties if p not in ("thought", "grounding")]) > 0
        if takes_values and name != "ask_user":
            assert "grounding" in properties, name
            assert properties["grounding"]["enum"] == tool_calling.GROUNDING_VALUES, name
            assert "grounding" in entry.schema()["function"]["parameters"]["required"], name


def test_tools_with_nothing_to_ground_are_not_asked_to():
    # get_current_time takes no values; ask_user composes its own question
    for name in ("get_current_time", "ask_user"):
        properties = tool_calling.REGISTRY[name].schema()["function"]["parameters"]["properties"]
        assert "grounding" not in properties, name


def test_a_call_marked_assumed_is_refused_and_never_executed():
    """The reported failure: asked to 'add 3 numbers' with none given, the model
    invented 5 + 10 + 15 and computed 30."""
    call = tool_calling.parse_call(
        '{"thought": "summing 5, 10 and 15", "expression": "5 + 10 + 15", "grounding": "assumed"}'
    )
    refusal = tool_calling.refuse_if_assumed(call, can_ask=True)
    assert refusal is not None
    assert "invented values" in refusal
    assert "ask_user" in refusal


def test_the_refusal_adapts_when_there_is_nobody_to_ask():
    call = tool_calling.parse_call('{"thought": "t", "expression": "1+1", "grounding": "assumed"}')
    refusal = tool_calling.refuse_if_assumed(call, can_ask=False)
    assert "ask_user" not in refusal
    assert "Say which values you are missing" in refusal


def test_grounded_calls_pass_through_untouched():
    for grounding in ("user", "tool_result"):
        call = tool_calling.parse_call(
            '{"thought": "t", "expression": "4 + 8 + 15", "grounding": "%s"}' % grounding
        )
        assert tool_calling.refuse_if_assumed(call, can_ask=True) is None
        assert tool_calling.execute("calculator", call.kwargs) == "27"


def test_grounding_is_stripped_before_the_tool_runs():
    call = tool_calling.parse_call('{"thought": "t", "query": "Dubai", "grounding": "user"}')
    assert call.kwargs == {"query": "Dubai"}


def test_both_prompts_carry_the_worked_example_of_the_failure():
    for prompt in (
        tool_calling.build_system_prompt(True),
        tool_calling.build_system_prompt(False),
    ):
        assert "add 3 numbers" in prompt
        assert "5 + 10 + 15" in prompt


def test_scratchpad_parses_and_gates_the_values_from_line():
    completion = "Thought: t\nAction: calculator\nAction Input: 5 + 10 + 15\nValues From: assumed"
    match = scratchpad.ACTION_RE.search(completion)
    assert match.group(1).strip() == "calculator"
    assert match.group(2).strip() == "5 + 10 + 15"
    assert match.group(3).strip().lower() == scratchpad.ASSUMED

    # the line is optional in text: the format can ask for it but cannot require it
    without = scratchpad.ACTION_RE.search("Thought: t\nAction: calculator\nAction Input: 2+2")
    assert without.group(3) is None


# ------------------- questions must not end the run ------------------------- #
def test_a_prose_question_is_recognised_rather_than_treated_as_an_answer():
    """The reported failure: the model asked "Could you please provide the seven
    topics?" in plain text, the loop read "no tool calls" as done, and the run
    ended before the user could reply."""
    asked = "I need to clarify. Could you please provide the seven topics you want?"
    assert tool_calling.looks_like_a_question(asked) == asked
    assert tool_calling.looks_like_a_question("The sum of 12, 30 and 7 is 49.") is None
    assert tool_calling.looks_like_a_question("  Which city?  ") == "Which city?"


def test_both_prompts_explain_that_plain_text_ends_the_run():
    for interactive in (True, False):
        prompt = tool_calling.build_system_prompt(interactive)
        assert "ENDS the run" in prompt
    # only the interactive one can promise the tool as the alternative
    assert "goes through the ask_user tool" in tool_calling.build_system_prompt(True)
    assert "ask_user" not in tool_calling.build_system_prompt(False)


def test_scratchpad_prompt_explains_that_a_final_answer_ends_the_run():
    scratchpad.register_tools(interactive=True)
    prompt = scratchpad.build_system_prompt()
    assert "ENDS the run" in prompt
    assert "Questions go through ask_user" in prompt


# ------------------------------- scratchpad --------------------------------- #
def test_scratchpad_prompt_and_tool_list_agree():
    scratchpad.register_tools(interactive=True)
    prompt = scratchpad.build_system_prompt()
    assert "ask_user" in prompt
    assert "the user's own words, or an earlier Observation" in prompt
    assert "5 + 10 + 15" in prompt          # the worked example of the failure

    scratchpad.register_tools(interactive=False)
    prompt = scratchpad.build_system_prompt()
    assert "ask_user" not in prompt
    assert "no one to ask in this session" in prompt


def test_scratchpad_stop_sequence_prevents_self_dealt_observations():
    """Without this the model writes its own Observation, inventing the result of
    a tool that was never called."""
    assert any("Observation:" in s for s in scratchpad.STOP_SEQUENCES)


@pytest.mark.parametrize("module", [scratchpad, tool_calling])
def test_examples_import_nothing_from_the_application(module):
    source = Path(module.__file__).read_text()
    assert "from app." not in source and "import app" not in source
