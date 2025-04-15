import os
import pathlib
import pyodbc
from sqlalchemy import create_engine
import urllib

import pandas as pd
from promptflow.tracing import start_trace, trace
import json

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.contents import ChatMessageContent, TextContent, ImageContent
from semantic_kernel.functions import KernelArguments, kernel_function
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings, OpenAIChatPromptExecutionSettings
)

from azure.identity import DefaultAzureCredential
from sqlalchemy import create_engine
from sqlalchemy import Index, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
import urllib

start_trace(collection="recipe_data_insights")
from dotenv import load_dotenv

from typing import TypedDict
class Result(TypedDict):
    data: dict
    error: str
    query: str
    execution_time: float

# Callable class with @trace decorator on the __call__ method
class RecipeDataInsights:
    """
    RecipeDataInsights tool. You can use this tool as a standalone flow to retrieve recipe data
    with natural language queries. 
    """

    def __init__(self, model_type="azure_openai"):
        self.model_type = model_type
        print("Initializing RecipeDataInsights...")
        self.kernel = Kernel()
        load_dotenv(override=True)
        service_id = "default"
        self.kernel.add_service(
            AzureChatCompletion(
                service_id=service_id,
            ),
        )

        # Create a chat history object
        self.chat_history = ChatHistory()


    @trace
    @kernel_function(description="recipe Data Insights tool to retrieve recipe data with natural language queries.")
    async def __call__(self, query: str) -> Result:
        print("Calling RecipeDataInsights with input:", query, flush=True)
        return await self.query_db(query)
    
    @trace
    @kernel_function(description="Query recipe database using sql statement")
    async def query_db(self,query: str) -> dict:
        try:
            print("Authenticating to Azure Database for PostgreSQL using Azure Identity...")
            print(f"Query: {query}")
            # Connect to the database based on environment variables
            POSTGRES_HOST = os.environ["POSTGRES_HOST"]
            POSTGRES_USERNAME = os.environ["POSTGRES_USERNAME"]
            POSTGRES_DATABASE = os.environ["POSTGRES_DATABASE"]
            POSTGRES_SSL = os.environ.get("POSTGRES_SSL")

            azure_credential = DefaultAzureCredential()
            token = azure_credential.get_token("https://ossrdbms-aad.database.windows.net/.default")
            POSTGRES_PASSWORD = token.token

            DATABASE_URI = f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DATABASE}"
            # Specify SSL mode if needed
            DATABASE_URI += f"?sslmode={POSTGRES_SSL}"

            engine = create_engine(DATABASE_URI, echo=False)

            # Run query
            results = []
            with Session(engine) as session:
                    most_similars = session.execute(text(query)) #.scalars()
                    #print(f"Most similar recipes to 'chicken and peanut':")
                    #print(f"--------------------------------------------------")
                    for Recipe in most_similars:
                            results.append({"recipe": Recipe[1], "score": Recipe[2]})
                            #print(f"INPUT: {Recipe[0]}: \nOUTPUT: {Recipe[1]} SCORE: ({Recipe[2]})")
                            #print(f"--------------------------------------------------")
        
            return results
        except Exception as e:
            print("*******Error in query_db*******", e)
            raise e


