import urllib3
from chromadb.config import Settings
from dtt import DocumentProcessor
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import GigaChat
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import GigaChatEmbeddings
from langchain_community.vectorstores import Chroma

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ChatBot:
    """A chat bot that processes a DOCX file, converts it to text, and uses it
    to answer questions."""

    def __init__(self, docx_file_path):
        """Initializes the ChatBot with a DOCX file path.

        Args:
            docx_file_path (str): The file path of the DOCX document to process.
        """
        self.text_file_path = docx_file_path[:-5] + 'Intxt.txt'
        self.auth_key = (
            'NjMyY2E1MTQtZDg4Ny00MjJjLThmYmUtNjlhOWJhNGVmYzFhOjllZjY4ZDllLTJkN2EtNDg3ZC05ZDc2LTE0ZTZkZjgwZmE3MA=='
        )

        # Initialize GigaChat API client
        self.giga_chat_simple = GigaChat(
            credentials=self.auth_key,
            scope='GIGACHAT_API_PERS',
            model='GigaChat',
            verify_ssl_certs=False,
            streaming=True,
        )

        # Process the DOCX file
        converter = DocumentProcessor(docx_file_path, self.giga_chat_simple)
        converter.save_to_txt(self.text_file_path)
        table_names = converter.get_table_titles()

        # Load and split text documents
        self.loader = TextLoader(self.text_file_path, encoding='utf-8')
        self.documents = self.loader.load()
        chunk_size = 2000
        chunk_overlap = 400

        while chunk_size > 100:
            try:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                self.documents = self.text_splitter.split_documents(self.documents)

                # Create embeddings for the documents
                self.embeddings = GigaChatEmbeddings(credentials=self.auth_key, verify_ssl_certs=False)

                # Create a database from the documents
                self.db = Chroma.from_documents(
                    self.documents,
                    self.embeddings,
                    client_settings=Settings(anonymized_telemetry=False),
                )

                break

            except Exception as e:
                if '"status":413' in str(e):
                    chunk_size = int(chunk_size * 0.9)
                    chunk_overlap = int(chunk_overlap * 0.9)
                else:
                    print(e)
                    break

        # Initialize GigaChat for question answering
        self.giga_chat = GigaChat(
            credentials=self.auth_key,
            model='GigaChat-Max',
            verify_ssl_certs=False,
            temperature=0.9,
        )

        # Set up the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.giga_chat,
            retriever=self.db.as_retriever(search_kwargs={'k': 15}),
            return_source_documents=True,
        )

        self.docx_context = (
            'You are a language model that can respond to the context in docx files, including lists, '
            'tables, and plain text.. When you encounter the markers "TABLE_START" and "TABLE_END", treat the '
            'enclosed content as a table. Each nested list within these markers represents a row of the table, and each '
            'element within a nested list represents a cell. Assume that the table structure is consistent and logical, '
            'but be aware that some rows may have fewer elements due to merged cells. In such cases, understand that the '
            'missing elements are a result of cell merging and adjust your interpretation accordingly. Additionally, '
            'recognize the marker "TABLE_TITLE" as the title or context for the following table, derived from the sentence '
            'preceding the table. Your task is to accurately understand and process this tabular data as if it were '
            'presented in a traditional table format, accounting for potential merged cells and contextual titles.'
        )
        self.message_history = []
        self.response_blacklist = (
            'Что-то в вашем вопросе меня смущает. Может, поговорим на другую тему? '
            'Как у нейросетевой языковой модели у меня не может быть настроения, но почему-то я совсем не хочу говорить на эту тему. '
            'Не люблю менять тему разговора, но вот сейчас тот самый случай.'
        )

    def get_answer(self, user_input):
        """Processes the user's query and returns an answer.

        :param user_input: A string containing the user's question.
        :return: A string with the chatbot's answer or an error message
            if the request is inappropriate.
        """
        # Append a question mark to the user's input for better context
        user_input_with_question_mark = user_input + '?'
        self.message_history.append('Question: ' + user_input_with_question_mark)

        # Gather recent context from the document and message history
        recent_context = self.docx_context + '\n'.join(self.message_history[-4:])

        # Get the answer from the question-answer chain
        answer = self.qa_chain.invoke({'query': recent_context})
        full_answer = answer['result']

        # Shorten the answer for storage in message history
        shortened_answer = self.giga_chat_simple.invoke(
            'Summarize this message into one concise sentence for dialog history, preserving its main point: '
            + full_answer,
        )

        # Check if the answer is in the blacklist
        if full_answer in self.response_blacklist or shortened_answer.content in self.response_blacklist:
            self.message_history.pop()
            return 'Ваш запрос неуместен для этого события. Пожалуйста, переформулируйте его, и я сделаю вид, что не услышал.'

        # Add the shortened answer to the message history
        self.message_history.append('Chatbot AI: ' + shortened_answer.content)
        return full_answer

    def notebook_demo(self):
        """Provides an interactive demo interface for Jupyter notebooks.

        Returns:
            self: Returns the chatbot instance for method chaining
        """
        import ipywidgets as widgets
        from IPython.display import HTML, clear_output, display

        # Create input widget
        text_input = widgets.Text(
            value='',
            placeholder='Type your question here...',
            description='Question:',
            layout=widgets.Layout(width='80%'),
        )

        # Create output widget for displaying responses
        output = widgets.Output()

        def on_submit(b):
            with output:
                clear_output()
                if text_input.value.lower() == 'stop':
                    print('Chat session ended.')
                    return

                response = self.get_answer(text_input.value)
                print(f'Q: {text_input.value}')
                print(f'A: {response}\n')

            text_input.value = ''  # Clear input after submission

        # Create submit button
        submit_button = widgets.Button(description='Ask', button_style='primary', layout=widgets.Layout(width='100px'))
        submit_button.on_click(on_submit)

        # Handle enter key press
        def handle_enter(sender):
            if sender.value and sender.value.strip():
                on_submit(None)

        text_input.on_submit(handle_enter)

        # Display the interface
        display(HTML('<h3>Chat Interface</h3>'))
        display(widgets.HBox([text_input, submit_button]))
        display(output)

        return self
