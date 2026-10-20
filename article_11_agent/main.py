"""Interactive chat loop for the LangGraph triage chatbot.

One thread_id per conversation. Every message in a conversation shares the thread,
so state carries across turns. A clarification pauses the graph; the loop shows
the question, reads the reply, and resumes the same thread.
"""

import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

import llm
from graph import build_graph

# Same messages as the Article 9 build, unchanged, so before/after runs use
# identical input. Test 9 replies for the original Article 8 trace:
#   first reply  -> ORD-9999
#   second reply -> ORD-1006 lisa.martin@email.com
TEST_MESSAGES = {
    "1": "Hi, I placed an order (ORD-1002) last week and wanted to check "
         "when it will arrive. Thanks, Mike",
    "2": "I received the wrong size jacket in my order ORD-1001. "
         "I ordered size M but got size L. My email is sarah.jones@email.com. "
         "What do I do?",
    "3": "Hello, I was charged twice for my order ORD-1003. "
         "I can see two charges of $120 on my card from the same day. "
         "Please refund the extra charge. My email is priya.patel@email.com",
    "4": "This is absolutely unacceptable. I have contacted you THREE times "
         "about my refund for order ORD-1004. The coffee maker was defective, "
         "my return was approved over a week ago and I still have not received "
         "my $75 refund. This is my last attempt before I dispute this with my bank. "
         "james.wilson@email.com",
    "5": "I cancelled my order ORD-1005 almost two weeks ago and I still "
         "haven't received my refund of $55. "
         "My email is lisa.martin@email.com",
    "6": "My standing desk order ORD-1006 was supposed to arrive on January 28th "
         "and it's still not here. That was a $520 purchase. "
         "Please let me know what's happening. lisa.martin@email.com",
    "7": "I'd like to return my bluetooth speaker from order ORD-1008. "
         "I know it's been a while but I think I'm still within the window. "
         "Email: mike.chen@email.com",
    "8": "I returned my laptop stand from order ORD-1007 because it didn't "
         "match the description. I only got $20 back but I paid $45. "
         "Why wasn't I fully refunded? sarah.jones@email.com",
    "9": "My order hasn't arrived and I'm really frustrated. Can someone help me?",
    "10": "I was told my blender (ORD-1009) was delivered on January 30th "
          "but I never received it. There was nothing at my door. "
          "david.kim@email.com",
    "11": "Hi, I'm having trouble with a recent order and I'm not sure what "
          "to do. Can someone help me figure out the right next step?",
    "12": "Hello, I never got my package and I'm not sure who to contact "
          "about it. Can you help me look into this?",
}


def run_turn(app, config: dict, message: str) -> None:
    """One customer message, including any clarification pauses."""
    llm.reset_calls()
    result = app.invoke({"messages": [HumanMessage(content=message)]}, config)

    while "__interrupt__" in result:
        print(f"\n[Support] {result['__interrupt__'][0].value['question']}")
        reply = input("Customer: ").strip()
        result = app.invoke(Command(resume=reply), config)

    print("\n" + "=" * 60)
    print("CUSTOMER RESPONSE")
    print("=" * 60)
    print(result.get("reply_text", "(no reply produced)"))
    print("=" * 60)
    print(f"LLM usage this turn: {llm.call_summary()}")


def new_config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("E-COMMERCE CUSTOMER SERVICE TRIAGE SYSTEM")
    print("Article 11: rebuilt on LangGraph")
    print("=" * 60)
    print("\nType a message, a test number (1-12), 'new' to start a new")
    print("conversation, or 'quit' to exit.")

    app = build_graph()
    config = new_config()

    while True:
        raw = input("\nCustomer: ").strip()
        if raw.lower() in ("quit", "exit", "q"):
            break
        if raw.lower() == "new":
            config = new_config()
            print("[Started a new conversation]")
            continue
        if not raw:
            continue

        message = TEST_MESSAGES.get(raw, raw)
        if raw in TEST_MESSAGES:
            print(f"Using test message {raw}: {message[:70]}...")
        run_turn(app, config, message)