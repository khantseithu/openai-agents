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

async def main():
    result = await Runner.run(agent, "What is the square root of 16?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
