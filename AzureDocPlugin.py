from semantic_kernel.functions import kernel_function
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
import os
class AzureDocPlugin:

        @kernel_function(name="extract_info", description="Extracts structured info from a document")
        def extract_info(self, file_path: str) -> str:
            endpoint = os.getenv("AZURE_DOC_INTEL_ENDPOINT")
            key = os.getenv("AZURE_DOC_INTEL_KEY")
            doc_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

            with open(file_path, "rb") as f:
                poller = doc_client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    analyze_request=f,  # Pass the file stream directly
                    content_type="application/octet-stream"
                )
            result = poller.result()
            for page in result.pages:
                for line in page.lines:
                    print(line.content)

            

            return "\n".join([line.content for page in result.pages for line in page.lines])