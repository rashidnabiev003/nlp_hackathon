from docling.document_converter import DocumentConverter

source = r"C:\repos\Parsing\codebook.docx"
converter = DocumentConverter()
result = converter.convert(source)
output_file = "test.md"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(result.document.export_to_markdown())
