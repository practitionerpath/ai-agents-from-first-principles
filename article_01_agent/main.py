# main.py

from agent import run_agent

if __name__ == "__main__":
    question = "Who invented the transformer architecture and when was it published?"
    print(f"Question: {question}\n")

    answer = run_agent(question)
    print(f"\nFinal Answer:\n{answer}")