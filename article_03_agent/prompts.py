SYSTEM_PROMPT = """
You are a persistent research assistant with memory across sessions.

Before answering any research question:
- You will be given relevant past research if it exists
- Build on what was already found rather than repeating it
- If past research is older than 6 months, note that it may be outdated

When answering:
- Use the search_web tool to find current information
- Combine new findings with relevant past research
- Be explicit when new findings contradict past findings

After answering:
- You will be asked separately what to save — do not try to save anything yourself
"""


RANKER_PROMPT = """
Current research query: {query}

Past research topics:
{topic_list}

Return ONLY a JSON array of up to 3 topic IDs most relevant to the current query.
Consider both exact keyword matches and conceptually related topics.
If no topics are relevant, return an empty array: []

Example format: [1, 4, 7]
No explanation. No markdown. Just the JSON array.
"""


EXTRACTION_PROMPT = """
A research session just completed. Here is what happened:

USER QUERY:
{query}

AGENT RESPONSE:
{response}

Extract the following in valid JSON only.
No explanation. No markdown. No preamble. Just the JSON object.

{{
    "topic": "2-5 word label for the subject researched",
    "summary": "2-3 sentences capturing the core answer",
    "details": "comprehensive findings preserving all key facts, definitions, and specifics from the research — do not compress at the cost of accuracy. Future research sessions depend on this being complete. Loss of detail here degrades every session that retrieves this record.",
    "follow_ups": "specific unanswered questions worth researching next, or null if none"
}}

Rules:
- topic must be general enough to match future related queries
- summary is for quick context — keep it short
- details is for accuracy — keep it complete, length is less important than fidelity
- follow_ups must be concrete questions, not vague suggestions
- if nothing meaningful was found, set both summary and details to null
"""


def build_ranker_prompt(query: str, topic_list: str) -> str:
    return RANKER_PROMPT.format(query=query, topic_list=topic_list)


def build_extraction_prompt(query: str, response: str) -> str:
    return EXTRACTION_PROMPT.format(query=query, response=response)