# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from semantic_kernel import Kernel
from semantic_kernel.agents import (
    ChatCompletionAgent,
    ChatHistoryAgentThread,
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.filters import FunctionInvocationContext

# Auto function invocation filter (for logging)
async def function_invocation_filter(context: FunctionInvocationContext, next):
    if "messages" not in context.arguments:
        await next(context)
        return
    print(f"    Agent [{context.function.name}] called with messages: {context.arguments['messages']}")
    await next(context)
    print(f"    Response from agent [{context.function.name}]: {context.result.value}")

# Chat loop
async def chat(triage_agent: ChatCompletionAgent, thread: ChatHistoryAgentThread = None) -> bool:
    try:
        user_input = input("User:> ")
    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting chat...")
        return False

    if user_input.lower().strip() == "exit":
        print("\n\nExiting chat...")
        return False

    response = await triage_agent.get_response(messages=user_input, thread=thread)

    if response:
        print(f"Agent :> {response}")

    return True

# Main function
async def main() -> None:
    kernel = Kernel()
    kernel.add_filter("function_invocation", function_invocation_filter)

    # Define shared service
    azure_service = AzureChatCompletion(
        deployment_name="gpt-4.1-nano",
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        base_url="https://utkar-mdicyby8-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4.1-nano/chat/completions?api-version=2025-01-01-preview"
    )

    # Billing Agent
    billing_agent = ChatCompletionAgent(
        service=azure_service,
        name="BillingAgent",
        instructions=(
            "You specialize in handling customer questions related to billing issues. "
            "This includes clarifying invoice charges, payment methods, billing cycles, "
            "explaining fees, addressing discrepancies in billed amounts, updating payment details, "
            "assisting with subscription changes, and resolving payment failures."
        )
    )

    # Refund Agent
    refund_agent = ChatCompletionAgent(
        service=azure_service,
        name="RefundAgent",
        instructions=(
            "You specialize in addressing customer inquiries regarding refunds. "
            "This includes evaluating eligibility for refunds, explaining refund policies, "
            "processing refund requests, providing status updates on refunds, handling complaints "
            "related to refunds, and guiding customers through the refund claim process."
        )
    )

    # Triage Agent
    triage_agent = ChatCompletionAgent(
        service=azure_service,
        kernel=kernel,
        name="TriageAgent",
        instructions=(
            "Your role is to evaluate the user's request and forward it to the appropriate agent based on the "
            "nature of the query. Forward requests about charges, billing cycles, payment methods, fees, or "
            "payment issues to the BillingAgent. Forward requests concerning refunds, refund eligibility, "
            "refund policies, or the status of refunds to the RefundAgent. Your goal is accurate identification "
            "of the appropriate specialist to ensure the user receives targeted assistance."
        ),
        plugins=[billing_agent, refund_agent],
    )

    print("Welcome to the chatbot!\nType 'exit' to exit.\nTry asking billing or refund-related questions.")

    thread: ChatHistoryAgentThread = None
    chatting = True
    while chatting:
        chatting = await chat(triage_agent, thread)

# Run the chatbot
if __name__ == "__main__":
    asyncio.run(main())
