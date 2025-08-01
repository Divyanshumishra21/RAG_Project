import os
import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        """
        Tidies up the input text while keeping paragraph structure intact.
        Removes unnecessary spacing but leaves logical breaks.
        """
        # Replace multiple spaces or tabs with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Convert multiple blank lines to double newlines (clean paragraph breaks)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Trim each line and remove leading/trailing whitespace
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines).strip()

    def load_document(self, file_path: str) -> str:
        """
        Loads a text document from the given path.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def chunk_document(self, text: str) -> List[str]:
        """
        Breaks a cleaned document into smaller chunks.
        Logic: preserves paragraphs and tries not to break context abruptly.
        """
        cleaned = self.clean_text(text)
        paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
        
        chunks = []
        buffer = ""

        for paragraph in paragraphs:
            # If a paragraph looks like a section title, treat it as a hard split
            if paragraph.startswith(tuple(f"{i}." for i in range(1, 20))):
                if buffer:
                    chunks.append(buffer.strip())
                buffer = paragraph
            else:
                buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph

                if len(buffer) > self.chunk_size:
                    chunks.append(buffer.strip())
                    buffer = ""

        if buffer:
            chunks.append(buffer.strip())

        # Remove extremely short chunks (noise)
        return [chunk for chunk in chunks if len(chunk.strip()) > 30]

    def process_documents(self, data_dir: str) -> List[str]:
        """
        Walks through a directory and processes all .txt/.md files into chunks.
        Returns a list of cleaned, meaningful text blocks.
        """
        chunks = []

        for fname in os.listdir(data_dir):
            if fname.endswith(('.txt', '.md')):
                path = os.path.join(data_dir, fname)
                content = self.load_document(path)
                chunks.extend(self.chunk_document(content))

        return chunks
