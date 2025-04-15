from semantic_kernel.agents.open_ai import AzureAssistantAgent
from semantic_kernel.contents import StreamingFileReferenceContent, FileReferenceContent
import base64
import json
import io


async def convert_image_to_BytesIO(agent: AzureAssistantAgent, file_ids: str,save_image=False):
    outputs = []
    if file_ids:
        for file_id in file_ids:
            print(f"File ID: {file_id}")
            # Fetch the content of the file using the provided method
            response_content = await agent.client.files.content(file_id)    
            output = io.BytesIO(response_content.content)      
            # Save image
            if save_image:
                with open(f"./{file_id}", "wb") as f:
                    f.write(response_content.content)      
            
            outputs.append(output)
    
    return outputs

########################
import urllib
from PIL import Image as Image
from semantic_kernel.functions import kernel_function
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

class RecipesPlugin:
    """
    RecipesPlugin is a plugin that provides a function to retrieve recipe data from a postefeSQL database.
    """

    @kernel_function(description="Recipes function to return recipe data")
    def __call__(self, query: str) -> str:
        print("Authenticating to Azure Database for PostgreSQL using Azure Identity...")
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
        
        
        return json.dumps(results)
    
########################
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from sqlalchemy import create_engine
from sqlalchemy import Index, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
import urllib

from dotenv import load_dotenv
import os

load_dotenv(override=True)

async def main():

    client, model = AzureAssistantAgent.setup_resources()

    assistant_definition = await client.beta.assistants.retrieve(os.getenv("AZURE_ASSISTANT_ID"))
    print(assistant_definition)

    agent_assistant = AzureAssistantAgent(
        client=client,
        name="assistant_recipes_test",
        definition=assistant_definition,
        #arguments=KernelArguments(settings=self.settings),
        plugins=[RecipesPlugin()],
    )

    thread = await agent_assistant.client.beta.threads.create()

    chat_request = "I am looking for recipe that contains chicken. Can you suggest me few similar recipes? Show me a graph"
    await agent_assistant.add_chat_message(thread_id=thread.id,message=chat_request)

    file_ids = []
    image_outputs = []
    final_response = None
    async for response in agent_assistant.invoke(thread_id=thread.id):#, history=history):
        file_ids.extend([
            item.file_id for item in response.items
            if isinstance(item, FileReferenceContent) and item.file_id not in file_ids
        ])
        image_outputs = await convert_image_to_BytesIO(agent_assistant, file_ids)
        for image_output in image_outputs:
            if isinstance(image_output, io.BytesIO):
                image = Image.open(image_output)
                image.show()
        print(f"# {response.name.upper()}")
        print(response.content, end="", flush=True)

    
#Run asyn main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())