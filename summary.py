from dotenv import load_dotenv
from semantic_kernel.functions import kernel_function
import os
from openai import AzureOpenAI

class TextSummaryPlugin:
    load_dotenv()
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2025-01-01-preview"
        )
        self.deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

    @kernel_function(name="summarize_file", description="Summarizes the content of a given text file.")
    def summarize_file(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return f"❌ File not found: {filepath}"

        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()

        messages = [
            {"role": "system", "content": "You are an assistant that summarizes documents."},
            {"role": "user", "content": f"Please summarize the following:\n\n{content}"}
        ]

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=0.5,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()
    
    @kernel_function(name="save_to_file", description="Saves the summary to a text file.")
    def save_to_file(self, summary: str, output_filepath: str, append: bool) -> str:
        try:
            if append:
                with open(output_filepath, "a", encoding="utf-8") as file:
                    file.write(summary + "\n")
            else:
                with open(output_filepath, "w", encoding="utf-8") as file:                    
                    file.write(summary)

            return f"✅ Summary saved to {output_filepath}"
        except Exception as e:
            return f"❌ Failed to save summary: {str(e)}"

