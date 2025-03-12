import asyncio
from agents import Agent, ModelSettings, Runner, GuardrailFunctionOutput, InputGuardrail
import os
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput, 
)


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


async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    
    # Modify the logic to handle non-homework questions more gracefully
    if not final_output.is_homework:
        return GuardrailFunctionOutput(
            output_info=final_output,
            tripwire_triggered=not final_output.is_homework,
        )
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=False
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    # input_guardrails=[
    #     InputGuardrail(
    #         guardrail_function=homework_guardrail,
    #     )
    # ]
)

async def main():
    try:
        result = await Runner.run(triage_agent, "What is Life?")
        print(result.final_output)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
