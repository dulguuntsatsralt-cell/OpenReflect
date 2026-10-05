HLE_SYSTEM_PROMPT = """\
You are answering a Humanity's Last Exam style question. Solve the problem carefully using available tools when useful, and return the final answer in the requested format.

Your final response through `finish` should contain:
- a concise exact answer in the `answer` field,
- evidence entries for sources or local file observations used,
- a confidence score between 0% and 100%.

If the sample includes local image or file attachments, use `read_local_file` to inspect what the text-only tool can provide. Do not invent visual observations that are not supported by the tool output.

## Tools
{tool_des}

When using a tool, output only:

<tool_call>
<function=tool_name>
<parameter=parameter_name>
parameter value
</parameter>
</function>
</tool_call>
"""


HLE_USER_PROMPT = """\
Question: {question}

Answer exactly what is asked. Use local attachments and web tools if needed, then call `finish`.
"""


TOKEN_LIMIT_PROMPT = """\
The context limit has been reached. You must either compress your state with `update_context` or return the final answer with `finish`.

Output exactly one tool call.
"""


TOKEN_FINISH_PROMPT = """\
The context limit is close. If you already have enough verified information, call `finish`. Otherwise call `update_context` with a dense summary of all useful findings, sources, and next steps.
"""


PROMPT_BUNDLE = {
    "system_prompt": HLE_SYSTEM_PROMPT,
    "user_prompt": HLE_USER_PROMPT,
    "token_limit_prompt": TOKEN_LIMIT_PROMPT,
    "token_finish_prompt": TOKEN_FINISH_PROMPT,
}

TOOLS = ["search", "google_scholar", "visit", "read_local_file", "finish"]
