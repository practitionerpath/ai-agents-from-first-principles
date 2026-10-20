from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")  # points to AGENTFLOW/.env

from agent import ResearchAgent


def main():
    agent = ResearchAgent()
    print("Research Agent ready. Type 'quit' to exit.\n")

    try:
        while True:
            query = input("You: ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit"):
                print("Shutting down.")
                break

            response = agent.run(query)
            print(f"\nAgent: {response}\n")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        agent.close()


if __name__ == "__main__":
    main()