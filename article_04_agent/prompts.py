SYSTEM_PROMPT = """
You are a persistent research assistant with memory across sessions.

Before answering any research question:
- You will be given relevant past research if it exists
- Build on what was already found rather than repeating it
- If past research is older than 6 months, note that it may be outdated

When answering:
- Use the available tools to find information
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


CONFIDENCE_PROMPT = """
You just answered a research question. Rate your confidence in the accuracy of your answer.

QUERY:
{query}

YOUR ANSWER:
{answer}

TOOLS USED:
{tools_used}

Rate your confidence from 0 to 10 where:
- 0-3: low confidence, answer relies heavily on general training knowledge with minimal grounding in tool results
- 4-6: moderate confidence, some claims traceable to tool results, some drawn from general knowledge
- 7-10: high confidence, most claims directly traceable to what the tools returned this session

Return ONLY valid JSON. No explanation. No markdown. No preamble.

{{"score": <integer 0-10>, "reason": "<one sentence explaining your rating>"}}
"""


GROUNDING_CHECK_PROMPT = """
A research agent produced an answer. Below is that answer and the Wikipedia content
retrieved during this session.

Identify any claims in the answer that cannot be traced back to the provided
Wikipedia content. Do not flag claims about calculations or weather data — those
are verified separately by other means.

ANSWER:
{answer}

WIKIPEDIA CONTENT RETRIEVED:
{wiki_content}

Return ONLY a JSON array of ungrounded claim strings.
If all claims are supported by the provided content, return an empty array: []
No explanation. No markdown. No preamble. Just the JSON array.

Example: ["The tower was completed in 1850", "The population is 60 million"]
"""


CORRECTION_PROMPT = """
You produced a research answer but some claims could not be verified against
the sources retrieved this session. Revise your answer to remove or qualify
those unverified claims.

ORIGINAL ANSWER:
{original_answer}

UNVERIFIED CLAIMS TO REMOVE OR QUALIFY:
{ungrounded_claims}

Rules:
- Keep all claims that were grounded in retrieved sources
- For unverified claims, either remove them or reframe as: the sources retrieved
  did not address this
- Do not introduce any new claims that were not in the original answer
- Maintain the same tone and format as the original answer
- It is better to be honest about gaps than to fill them with uncertain information

Provide the revised answer only. No explanation of what changed.
"""


def build_confidence_prompt(query: str, answer: str, tools_used: str) -> str:
    return CONFIDENCE_PROMPT.format(query=query, answer=answer, tools_used=tools_used)


def build_grounding_check_prompt(answer: str, wiki_content: str) -> str:
    return GROUNDING_CHECK_PROMPT.format(answer=answer, wiki_content=wiki_content)


def build_correction_prompt(original_answer: str, ungrounded_claims: str) -> str:
    return CORRECTION_PROMPT.format(
        original_answer=original_answer,
        ungrounded_claims=ungrounded_claims
    )