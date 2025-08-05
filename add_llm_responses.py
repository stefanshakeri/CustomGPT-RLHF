#!/usr/bin/env python3
"""
Script to convert 'llm_response_1' and 'llm_response_2' columns in a CSV to
vector embeddings and save them in an existing Chroma vector database. 
"""

import os

import openai
import pandas as pd
from dotenv import load_dotenv

from langchain.schema import Document

from create_database import DATA_PATH, CHROMA_PATH, add_to_chroma_db

# load environment variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_llm_responses() -> list[Document]:
    """
    Reads the CSV file and extracts 'llm_response_1' and 'llm_response_2' columns into a list.

    :return: list of LLM responses (Document objects)
    """
    documents = []
    df = pd.read_csv(DATA_PATH)

    if 'llm_response_1' not in df.columns or 'llm_response_2' not in df.columns:
        raise ValueError(f"'llm_response_1' or 'llm_response_2' columns not found in {DATA_PATH}")
    
    # extract 'llm_response_1' column into a list
    responses = df['llm_response_1'].dropna().tolist()

    # convert to Document objects with metadata of question number and response type
    for i, response in enumerate(responses):
        doc = Document(
            page_content=response,
            metadata={"question_id": i, "type": "llm_1"}
        )
        documents.append(doc)

    # extract 'llm_response_2' column into a list
    responses = df['llm_response_2'].dropna().tolist()

    # convert to Document objects with metadata of question number and response type
    for i, response in enumerate(responses):
        doc = Document(
            page_content=response,
            metadata={"question_id": i, "type": "llm_2"}
        )
        documents.append(doc)
    
    return documents


def main():
    """
    Main function to read LLM responses from CSV and add them to the Chroma vector database.
    """

    documents = get_llm_responses()

    if not documents:
        print("No LLM responses found in the CSV file.")
        return
    
    add_to_chroma_db(documents, CHROMA_PATH)

    if os.path.exists(CHROMA_PATH):
        print(f"{len(documents)} LLM responses added to Chroma database at {CHROMA_PATH}.")
    else:
        print(f"Failed to add LLM responses to Chroma database at {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
