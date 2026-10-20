# main.py

from agent import run_agent

if __name__ == "__main__":
    question = "Who invented the transformer architecture and when was it published?"
    print(f"Question: {question}\n")

    answer = run_agent(question)
    print(f"\nFinal Answer:\n{answer}")

    question = "What is the weather in Boston?"
    print(f"Question: {question}\n")

    answer = run_agent(question)
    print(f"\nFinal Answer:\n{answer}")

    question = "How much is (2*3)/0.2?"
    print(f"Question: {question}\n")

    answer = run_agent(question)
    print(f"\nFinal Answer:\n{answer}")

    # Multi agent loop test — more complex question that requires multiple tools and reasoning steps
    question = "What is the current temperature in Boston and London? What is the average of the two? Also, who invented the thermometer and when?"
    print(f"Question: {question}\n")

    answer = run_agent(question)
    print(f"\nFinal Answer:\n{answer}")