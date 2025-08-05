import os
from pdf2image import convert_from_path
from semantic_kernel.functions import kernel_function

class PdfToImgPlugin:
    @kernel_function(
        name="convert_pdf_to_images",
        description="Converts a PDF file to a series of images, one for each page."
    )
    def convert_pdf_to_images(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            return f"❌ PDF file not found: {pdf_path}"
        if not pdf_path.lower().endswith('.pdf'):
            return "❌ The provided file is not a PDF."

        # Set DPI for clarity (300+ recommended)
        images = convert_from_path(pdf_path, dpi=500)

        #clear all images in ./pdfToImg folder

        folder_path = './pdfToImg'

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Save each page as a JPEG
        for i, image in enumerate(images):   
            image.save(f'./pdfToImg/page_{i+1}.jpg', 'JPEG')
        
        return f"✅ Converted {len(images)} pages to images in './pdfToImg' folder."
