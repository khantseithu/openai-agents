import asyncio
from agents import Agent, ModelSettings, Runner
import os
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    name="Math Tutor",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
    model="gpt-4o-mini",
    model_settings=ModelSettings(
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        parallel_tool_calls=True,
    )
)

from agents import Agent

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent]
)

async def main():
    result = await Runner.run(triage_agent, "What is the capital of France?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
