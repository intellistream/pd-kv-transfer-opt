"""Project visible ATIF history for serving benchmarks."""

from datetime import datetime
from pathlib import PurePosixPath


def request_histories(steps, history_mode="reasoning"):
    """Project observed messages/calls/results without inventing missing content."""
    if history_mode not in ("reasoning", "visible"):
        raise ValueError("Unknown historical reasoning policy")
    messages = []
    for step in steps:
        source = step["source"]
        if source == "agent":
            if not messages:
                raise ValueError("An agent step has no preceding input history")
            yield step["step_id"], list(messages)
            calls = [
                {
                    "id": call["tool_call_id"],
                    "type": "function",
                    "function": {
                        "name": call["function_name"],
                        "arguments": call["arguments"],
                    },
                }
                for call in step.get("tool_calls", [])
            ]
            message = {"role": "assistant", "content": step["message"]}
            if calls:
                message["tool_calls"] = calls
            if history_mode == "reasoning" and step.get("reasoning_content"):
                message["reasoning_content"] = step["reasoning_content"]
            messages.append(message)
            for result in (step.get("observation") or {}).get("results", []):
                content = result["content"]
                if not isinstance(content, str):
                    raise TypeError(
                        "Non-text observations require a separate multimodal profile"
                    )
                call_id = result.get("source_call_id")
                if call_id and call_id in {call["id"] for call in calls}:
                    messages.append(
                        {"role": "tool", "tool_call_id": call_id, "content": content}
                    )
                else:
                    # Do not guess which command an unlabelled result belongs to.
                    messages.append({"role": "user", "content": content})
        elif source in ("user", "system"):
            if not isinstance(step["message"], str):
                raise TypeError("Non-text inputs require a separate multimodal profile")
            messages.append({"role": source, "content": step["message"]})
        else:
            raise ValueError(f"Unsupported ATIF source: {source}")


def recover_tool_intervals(session, trace, trial):
    """Estimate response-to-next-call waits from Harbor's ATIF and API clocks."""
    task = session["task"]
    metadata = trial["agent_result"]["metadata"]
    steps = [step for step in trace["steps"] if step["source"] == "agent"]
    times = metadata["api_request_times_msec"]
    source_task = PurePosixPath(trial["config"]["task"]["path"]).name.removeprefix("terminal-bench__")
    if (source_task != task or metadata["summarization_count"] or not steps or len(times) != len(steps)
            or trial["agent_info"]["name"] not in ("terminus2", "terminus-2")
            or any(ms <= 0 for ms in times)):
        raise ValueError(f"API durations cannot be joined one-to-one in {task}")
    step_ends = [datetime.fromisoformat(step["timestamp"]) for step in steps]
    first_agent_index = trace["steps"].index(steps[0])
    positions = {step["step_id"]: i for i, step in enumerate(steps)}
    intervals = {}
    for parent, child in zip(session["turns"], session["turns"][1:]):
        i, next_i = positions[parent["step_id"]], positions[child["step_id"]]
        if next_i != i + 1 or (i == 0 and first_agent_index == 0):
            raise ValueError(f"Missing surrounding steps for tool clock of {parent['id']}")
        prior = (step_ends[i - 1] if i else
                 datetime.fromisoformat(trace["steps"][first_agent_index - 1]["timestamp"]))
        elapsed_ms = (step_ends[i] - prior).total_seconds() * 1000
        estimated = elapsed_ms - times[i]
        if estimated < 0:
            raise ValueError(f"Negative step residual for {parent['id']}")
        nominal_wait_ms = sum(float(c.get("arguments", {}).get("duration", 0)) * 1000
                              for c in steps[i].get("tool_calls", ()))
        intervals[child["id"]] = {"parent": parent["id"], "gap_ms": round(estimated),
                                   "step_elapsed_ms": elapsed_ms, "same_step_API_ms": times[i],
                                   "parent_configured_tool_wait_ms": nominal_wait_ms}
    return {"task": task, "source_agent_calls": len(steps), "intervals_by_child": intervals}
