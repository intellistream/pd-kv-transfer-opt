"""Project visible ATIF history for serving benchmarks."""

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
