import base64
import logging
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments

from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents.chat_history import ChatHistory

class NanoVisionPlugin:
    def __init__(self, chat_service: AzureChatCompletion):
        """
        Accepts AzureChatCompletion instance from the main orchestration layer.
        """
        self.chat_service = chat_service

    def encode_image(self, image_path: str) -> str:
        """
        Reads and base64-encodes image for multimodal prompt.
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @kernel_function(name="analyze_image", description="Analyze image and extract visible text using GPT-4.1 Nano")
    async def analyze_image(self, file_path) -> str:
        """
        Uses KernelArguments to extract 'image_path' and returns KernelResult with response.
        """
        setup_logging()
        logging.getLogger("kernel").setLevel(logging.DEBUG)
        history = ChatHistory()
        
        image_path = file_path
        if image_path is None:
            return ("Missing required argument: 'image_path'")

        base64_image = self.encode_image(image_path)
        image_url = f"data:image/jpeg;base64,{base64_image}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image and extract any visible text."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
        history.add_message(
            messages
        )
        


        settings = AzureChatPromptExecutionSettings()
        reply = await self.chat_service.get_chat_message_content(history, settings)

        result = reply.content if hasattr(reply, "content") else str(reply)
        return result

    
