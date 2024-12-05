import os
import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()

parser = LlamaParse(
    result_type="markdown",  # "markdown" and "text"  and "json"
    show_progress=True,
    parsing_instruction=None,
    # structured_output=True,
    disable_ocr=True,
    disable_image_extraction=True,
    api_key=os.getenv("LLAMA_CLOUD_API_KEY")
)

file_extractor = {".docx": parser}
documents = SimpleDirectoryReader(input_files=['DOCUMENT_NAME'],
                                  file_extractor=file_extractor).load_data(show_progress=True)

all_pages_content = ""
for i, document in enumerate(documents):
    all_pages_content += f"### Page {i + 1}\n\n{document.text}\n\n"

# Сохраняем весь текст в Markdown файл
output_md_path = "all_pages_output.md"
with open(output_md_path, "w", encoding="utf-8") as f:
    f.write(all_pages_content)
