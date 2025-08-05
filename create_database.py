#!/usr/bin/env python3
"""
Script to create a Chroma vector database from the expert_response feature in the CSV
being used. This script reads a CSV file, extracts the 'expert_response' column,
and creates a Chroma vector database from the text data.
"""

import os
import shutil

import openai
import pandas as pd
from dotenv import load_dotenv

from langchain.schema import Document

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# load environment variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

DATA_PATH = "data/" + os.getenv("INPUT_FILE")

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")


def get_expert_responses() -> list[Document]:
    """
    Reads the CSV file and extracts the 'expert_response' column into a list. 

    :return: list of expert responses (Document objects)
    """
    df = pd.read_csv(DATA_PATH)

    if 'expert_response' not in df.columns:
        raise ValueError(f"'expert_response' column not found in {DATA_PATH}")
    
    # extract the 'expert_response' column into a list
    responses = df['expert_response'].dropna().tolist()

    # convert to Document objects with metadata of question number and response type
    documents = []

    for i, response in enumerate(responses):
        doc = Document(
            page_content=response,
            metadata={"question_id": i, "type": "expert"}
        )
        documents.append(doc)

    return documents


def create_chroma_db(responses: list[Document]):
    """
    Save the responses to a Chroma vector database. Each expert response is labeled as an expert response in the database.

    :param responses: list of expert responses (Document objects)
    """

    # clear out existing Chroma database
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # create a new Chroma database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OpenAIEmbeddings()
    )

    # add documents to the database
    db.add_documents(responses)


def add_to_chroma_db(responses: list[Document], chroma_path: str = CHROMA_PATH):
    """
    Add new responses to an existing Chroma vector database.
    
    :param responses: list of expert responses (Document objects)
    """
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=OpenAIEmbeddings()
    )

    # add documents to the database
    db.add_documents(responses)


def main():
    """
    Main function to create the Chroma vector database.
    """
    responses = get_expert_responses()

    if not responses:
        print("No expert responses found.")
        return

    create_chroma_db(responses)
    print(f"Chroma vector database created with {len(responses)} documents.")


if __name__ == "__main__":
    main()