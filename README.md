# ai-agents-from-first-principles

Code for a learning-in-public blog series on building AI agents: from a first single agent, to a multi-agent customer service chatbot built without a framework, to a rebuild of that chatbot on LangGraph.

Each article's code lives in its own folder, with a README that shows how to run it.

## Articles

| # | Topic | Code |
|---|---|---|
| 1 | Building a first AI agent from scratch | [article_01_agent](article_01_agent) |
| 2 | Tool use in AI agents | [article_02_agent](article_02_agent) |
| 3 | Memory in AI agents | [article_03_agent](article_03_agent) |
| 4 | Detecting and handling hallucination | [article_04_agent](article_04_agent) |
| 5 | RAG, agentic RAG, and fine-tuning | [article_05_agent](article_05_agent) |
| 6 | A production postmortem analyzer | [article_06_agent](article_06_agent) |
| 7 | Why one agent isn't enough | No code, concept article |
| 8 | A multi-agent system built without a framework | [article_08_agent](article_08_agent) |
| 9 | Agent communication patterns | [article_09_agent](article_09_agent) |
| 10 | Choosing LangGraph | [article_10_agent](article_10_agent) |
| 11 | Rebuilding the chatbot on LangGraph | [article_11_agent](article_11_agent) |
| 12 | Harness engineering: auditing the chatbot | No code, audits the Article 11 build |

## Setup

Clone the repo, then from inside the cloned folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file at the root with your OpenAI API key:

```bash
OPENAI_API_KEY=your-key-here
```

Then open an article folder and follow its README.
