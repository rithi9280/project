from docx import Document

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return '\n'.join([p.text for p in doc.paragraphs])

text = extract_text_from_docx('sdnbvc_chatbot.pdf.docx')
