
"""Minimal local AgentOS demo. Run: python main.py"""

from datetime import datetime, timezone
import os
from pathlib import Path

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.deepinfra import DeepInfra
from agno.os import AgentOS


def get_utc_time() -> str:
    """Return the current UTC time from the server."""
    return datetime.now(timezone.utc).isoformat()


db = SqliteDb(db_file=str(Path(__file__).with_name("agentos.db")))

agent = Agent(
    id="demo-agent",
    name="Demo Agent",
    model=DeepInfra(
        id="deepseek-ai/DeepSeek-V4-Flash",
        api_key="sk-di-cRv3mQUJ6M9kmXevH1xEPwHSnw0ufQXe",
    ),
    db=db,
    tools=[get_utc_time],
    add_history_to_context=True,
    instructions="Be concise. Use get_utc_time when asked for the current time.",
    markdown=True,
)

agent_os = AgentOS(agents=[agent], db=db, tracing=True)
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="main:app", host="127.0.0.1", port=7777, reload=True)
