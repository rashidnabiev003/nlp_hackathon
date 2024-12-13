import docx


class DocumentProcessor:
    def __init__(self, file_path, llm):
        """Initialize the DocumentProcessor with a DOCX file and a language
        model.

        :param file_path: Path to the DOCX file.
        :param llm: Language model used for generating table titles.
        """
        self.doc = docx.Document(file_path)
        self.llm = llm

    def find_last_sentence(self, text_list):
        """Find the last non-empty sentence in a list of text elements.

        :param text_list: List of text elements.
        :return: The last non-empty sentence or a default message if
            none found.
        """
        for text in reversed(text_list):
            sentences = text.split('.')
            for sentence in reversed(sentences):
                if sentence.strip():
                    return sentence.strip()
        return 'No preceding text'

    def process_content(self):
        """Process the content of the document, extracting paragraphs and
        tables.

        :return: A tuple containing processed text and a list of table
            titles.
        """
        processed_text = []
        table_titles = []

        for element in self.doc.element.body:
            if element.tag.endswith('p'):
                paragraph = docx.text.paragraph.Paragraph(element, self.doc)
                processed_text.append(paragraph.text)
            elif element.tag.endswith('tbl'):
                last_sentence = self.find_last_sentence(processed_text)
                title = self.llm.invoke(
                    f'Преобразуй описание перед таблицей "{last_sentence}" в название самой таблицы. Без лишних символов и слова Таблица',
                ).content
                table_titles.append(title)
                processed_text.append(f'TABLE_TITLE {title}')

                table = docx.table.Table(element, self.doc)
                table_data = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_data.append(row_data)
                processed_text.append(f'TABLE_START {table_data} TABLE_END')

        return processed_text, table_titles

    def save_to_txt(self, output_file_path):
        """Save processed document content to a text file.

        :param output_file_path: Path to the output text file.
        """
        content, _ = self.process_content()
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for line in content:
                file.write(line + '\n')

    def get_table_titles(self):
        """Get a list of table titles from the document.

        :return: List of table titles.
        """
        _, table_titles = self.process_content()
        return table_titles
