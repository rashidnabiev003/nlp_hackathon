"""Chat interface for the LLaMA RAG system."""

import logging
import sys
from typing import Optional, Protocol, TextIO

from langchain import chains

logger = logging.getLogger(__name__)


class IOHandler(Protocol):
    """Protocol for handling I/O operations."""

    def read_input(self, prompt: str) -> str:
        """Read input with given prompt.

        Args:
            prompt: Input prompt to display
        """

    def write_output(self, message: str) -> None:
        """Write output message.

        Args:
            message: Message to write
        """


class ConsoleIO(IOHandler):
    """Handles console I/O operations."""

    def __init__(self, input_stream: TextIO = sys.stdin, output_stream: TextIO = sys.stdout):
        """Initialize ConsoleIO with input/output streams.

        Args:
            input_stream: Input stream to read from (default: sys.stdin)
            output_stream: Output stream to write to (default: sys.stdout)
        """
        self.input_stream = input_stream
        self.output_stream = output_stream

    def read_input(self, prompt: str) -> str:
        """Read input from configured input stream.

        Args:
            prompt: Input prompt to display

        Returns:
            str: User input
        """
        if self.input_stream.isatty():
            return sys.stdin.readline().strip()

        # For non-interactive streams (testing, automation)
        line = self.input_stream.readline().strip()
        if line:
            self.write_output(f'{prompt}{line}')
        return line

    def write_output(self, message: str) -> None:
        """Write output to configured output stream.

        Args:
            message: Message to write
        """
        self.output_stream.write(f'{message}\n')
        self.output_stream.flush()


class LoggingIO(IOHandler):
    """Handles I/O operations with logging."""

    def __init__(self, console_io: IOHandler):
        """Initialize LoggingIO with console I/O handler.

        Args:
            console_io: Underlying console I/O handler
        """
        self.console_io = console_io

    def read_input(self, prompt: str) -> str:
        """Read and log input.

        Args:
            prompt: Input prompt to display

        Returns:
            str: User input
        """
        user_input = self.console_io.read_input(prompt)
        logger.debug('User input: %s', user_input)
        return user_input

    def write_output(self, message: str) -> None:
        """Write and log output.

        Args:
            message: Message to write
        """
        logger.info(message)
        self.console_io.write_output(message)


def handle_user_input(io_handler: IOHandler) -> Optional[str]:
    """Get and process user input.

    Args:
        io_handler: I/O handler

    Returns:
        Optional[str]: Processed user input or None if user wants to exit
    """
    query = io_handler.read_input('Question: ')
    return None if query.lower() == 'exit' else query


def process_question(query: str, qa_chain: chains.RetrievalQA) -> str:
    """Process a single question through the QA chain.

    Args:
        query: User's question
        qa_chain: Configured QA chain

    Returns:
        str: Model's response or error message
    """
    try:
        return qa_chain.run(query).strip()
    except Exception as error:
        logger.error('Error processing question: %s', error)
        return 'Sorry, I encountered an error processing your question.'


def run_chat_session(qa_chain: chains.RetrievalQA, io_handler: IOHandler) -> None:
    """Run the chat session loop.

    Args:
        qa_chain: Configured QA chain
        io_handler: I/O handler
    """
    while True:
        query = handle_user_input(io_handler)
        if query is None:
            logger.info('Chat session ended')
            break

        response = process_question(query, qa_chain)
        io_handler.write_output(f'Bot: {response}')


def chat(qa_chain: chains.RetrievalQA) -> None:
    """Interactive chat interface for the QA system.

    Args:
        qa_chain: Configured RetrievalQA chain instance
    """
    logger.info("Chat session started. Type 'exit' to end.")
    console_io = ConsoleIO()
    io_handler = LoggingIO(console_io)

    try:
        run_chat_session(qa_chain, io_handler)
    except KeyboardInterrupt:
        logger.info('Chat session interrupted')
    except Exception as error:
        logger.error('Error in chat session: %s', error)
