from tkinter import Image
from pytesseract import pytesseract
import fitz
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file. Use OCR for image-based PDFs and direct text extraction for native PDFs.
    :param pdf_path: Path to the PDF file.
    :return: Extracted text.
    """
    pdf_text = ""
    pdf_document = fitz.open(pdf_path)

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]

        # Try to extract text directly
        text = page.get_text()
        if text.strip():
            pdf_text += text
        else:
            # If no text is found, use OCR on the page image
            pix = page.get_pixmap()
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(image)
            pdf_text += text

    pdf_document.close()
    return pdf_text


if __name__ == '__main__':
    filename = "/Users/jonathanwang/Desktop/07_01_Bow_AsianFemaleRobot.pdf"
    pdf_text = extract_text_from_pdf(filename)
    print(pdf_text)