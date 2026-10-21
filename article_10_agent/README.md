# ai-agents-from-first-principles / article_10_agent

## Activate
```bash
source ../.venv/bin/activate
```

## Run Order
```bash
python3 state_and_reducers.py
```
Shows reducers fixing a corrected order ID, failing loudly on concurrent writes, and losing history silently without a reducer.

```bash
python3 routing_guard.py
```
Runs the same scripted supervisor with the identifier guard off and on to show code removing an invalid route.

```bash
python3 interrupt_resume.py start
python3 interrupt_resume.py resume "It's ORD-1011."
python3 interrupt_resume.py history
```
Pauses the conversation at clarification, resumes it in a separate process from a SQLite checkpoint, then prints the checkpoint history.
