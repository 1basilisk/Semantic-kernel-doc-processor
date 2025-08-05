
import os
import base64
from openai import AzureOpenAI
from semantic_kernel.functions import kernel_function
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from groq import Groq

from dotenv import load_dotenv
class VisionPlugin:
    @kernel_function(name="extract_info_from_image", description="Extracts structured info from an image")
    def extract_info_from_image(self, image_path: str) -> str:
        load_dotenv()
        # endpoint = os.getenv("ENDPOINT_URL", "https://utkar-mdicyby8-eastus2.openai.azure.com/")
        # deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1-nano")
        # subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
        # model = "grok-3"
        

        #Initialize Azure OpenAI client with key-based authentication
        # client = AzureOpenAI(
        #     azure_endpoint=endpoint,
        #     api_key=subscription_key,
        #     api_version="2025-01-01-preview",
        # )

        client = Groq()
        

        


        encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')

        #client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(subscription_key))

        # Prepare the chat prompt
        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "you are an ai assistant and your job is to extract information from the document provided to you.\n\nclassify the document as \"aadhar\", \"pan\", \"\", or \"policy document\" then extract key information from them and show in organised manner"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "\n"
                    }
                ]
            }
        ]

        # Include speech result if speech is enabled
        messages = chat_prompt

        # Generate the completion
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        return (completion.to_json())
        