import asyncio
import ctypes
import logging
import os
from typing import Annotated
from semantic_kernel.functions import kernel_function
from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from AzureDocPlugin import AzureDocPlugin
from volumePlugin import VolPlugin
from summary import TextSummaryPlugin
from semantic_kernel.prompt_template import PromptTemplateConfig
from WallpaperPlugin import wallpaperPlugin

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from NanoVisionPlugin import NanoVisionPlugin
from VisionPlugin import VisionPlugin
from PdfToImgPlugin import PdfToImgPlugin



async def main():
    # Initialize the kernel
    kernel = Kernel()

    # Add Azure OpenAI chat completion
    chat_completion = AzureChatCompletion(
        deployment_name="gpt-4.1-nano",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    kernel.add_service(chat_completion)


    # chat_completion = AzureChatCompletion(
    #     deployment_name="grok-3",
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY_GROK"),
    #     base_url=os.getenv("AZURE_OPENAI_ENDPOINT_GROK"),
    # )
    # kernel.add_service(chat_completion)




    # Set the logging level for  semantic_kernel.kernel to DEBUG.
    setup_logging()
    logging.getLogger("kernel").setLevel(logging.DEBUG)

    ## adding sematic function
    prompt_text = "rephrase this text to be more professional: \n{{$input}}"

    config = PromptTemplateConfig(
        name="rephrase_text",
        template=prompt_text,
        description="Rephrases the input text to be more professional."
        )

    kernel.add_function(
        function_name="rephrase_text",
        plugin_name="TextRephraser",
        prompt_template_config=config,
    )

    # kernel.add_plugin(
    #     AzureDocPlugin(),
    #     plugin_name="AzureDocumentPlugin",
    # )

    kernel.add_plugin(
        wallpaperPlugin(),
        plugin_name="Wallpaper",
    )
    kernel.add_plugin(
        VolPlugin(),
        plugin_name="Volume",
    )

    kernel.add_plugin(
        TextSummaryPlugin(),
        plugin_name="Summary",
    )

    kernel.add_plugin(
        VisionPlugin(),
        plugin_name="Vision",
    )

    kernel.add_plugin(
        PdfToImgPlugin(),
        plugin_name="PdfToImage",
    )

    # Enable planning
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Create a history of the conversation
    history = ChatHistory()

    # Initiate a back-and-forth chat
    userInput = None
    while True:
        # Collect user input
        userInput = input("User > ")

        # Terminate the loop if the user says "exit"
        if userInput == "exit":
            break

        # Add user input to the history
        history.add_user_message(userInput)

        # Get the response from the AI
        result = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel,
        )

        # Print the results
        print("Assistant > " + str(result))

        # Add the message from the agent to the chat history
        history.add_message(result)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())