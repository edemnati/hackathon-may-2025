# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
import io

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents import ChatHistoryTruncationReducer
from semantic_kernel.functions import KernelFunctionFromPrompt

# Agent collaboration
from semantic_kernel.agents.open_ai import AzureAssistantAgent
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)

from dotenv import load_dotenv
import os

load_dotenv(override=True)

from recipe_data_insights import RecipeDataInsights

SERVICE_ID = "default"

# Define agent names
REVIEWER_NAME = "Reviewer"
CODER_NAME = "Coder"

def query_db(query: str) -> dict:
    try:
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
    
        return results
    except Exception as e:
        print("*******Error in query_db*******", e)
        raise e


def create_kernel() -> Kernel:
    """Creates a Kernel instance with an Azure OpenAI ChatCompletion service."""
    kernel = Kernel()
    kernel.add_service(service=AzureChatCompletion(service_id="default"))

    # Get the AI Service settings
    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=SERVICE_ID)

    # Configure the function choice behavior to auto invoke kernel functions
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Add the Plugin to the Kernel
    #kernel.add_plugin(RecipeDataInsights, plugin_name="RecipeDataInsights")

    kernel.add_plugin(RecipeDataInsights(), plugin_name="RecipeDataInsights")
    return kernel, settings


from semantic_kernel.contents import StreamingFileReferenceContent,FileReferenceContent
from semantic_kernel.functions import KernelArguments

async def main():

    # Create a single kernel instance for all agents.
    print("Creating kernel...")
    kernel, settings = create_kernel()

    # Create ChatCompletionAgents using the same kernel.
    
    # Create the client using Azure OpenAI resources and configuration
    client, model = AzureAssistantAgent.setup_resources()
    # Get the code interpreter tool and resources
    file_ids = []
    code_interpreter_tools, code_interpreter_tool_resources = AzureAssistantAgent.configure_code_interpreter_tool(
        file_ids=file_ids
    )

    #Create reviewer agent
    print("Creating reviewer agent...")
    reviewer_definition = await client.beta.assistants.retrieve(os.getenv("AZURE_ASSISTANT_ID"))

    '''   
    reviewer_definition = await client.beta.assistants.create(
        model=model,
        instructions="""
Your responsibility is to review the provided data and present the results to the user using markdown tables and charts.
                You don't have access to the data directly, always call the Coder agent to get the data.
                
                Follow these instructions to assist the user:
                STEPS:
                1. Review the user question and identify the data needed to answer it.
                2. Call the Coder agent to get the data. Reformulate the user question to ask the Coder agent for the data.
                3. Review the data returned by the Coder agent and format it to return the appropriate response.
                4. If the data is not satisfactory, provide specific suggestions to the Coder agent to improve the data.
                5. If the data is satisfactory, return it to the user.

            
                RULES:
                - Only identify suggestions that are specific and actionable.
                - Verify previous suggestions have been addressed.
                - Never try to simulate the data, always call the Coder agent to get the data.
                - Never repeat previous suggestions.
                - Never generate SQL query, let the coder agent do that.
                - **MUST** nerver make up data or use data from other tables. Use only result returned by the Coder agent.
""",
        name=REVIEWER_NAME,
        tools=[{"type": "code_interpreter"}]
        )
'''
    agent_reviewer = AzureAssistantAgent(
            client=client,
            name=REVIEWER_NAME,
            definition=reviewer_definition,
        )

    # Create the assistant definition
    system_message = """
You are expert in PostgreSQL queries. Given an input question, create a syntactically correct SQL query AND use the RecipeDataInsights function to execute the query.
If the user is asking to generate a graph,  you should only return the data in markdown table format.
 PostgreSQL "recommend_recipe_by_description" table has properties:
    #
    #  in_recipedescription character varying "input recipe description"
    #  out_recipename character varying "similar recipe name"
    #  out_similarityscore real "similarity score between 0 and 1"
    #  

        
Examples:
Question: What recipes are similar to 'chicken and peanut'?

Query:
SELECT in_recipedescription, out_recipename, out_similarityscore 
FROM recommend_recipe_by_description('chicken and peanut', 3)
WHERE out_similarityscore !=0
ORDER BY 2 DESC
; 

Answer:
The top 3 most similar recipes to 'chicken and peanut' are in markdown format: 
markdown table example:
| Recipe Name | Similarity Score |
| ------------- | ----------------- |
| Recipe 1 | 0.95 |
| Recipe 2 | 0.90 |
| Recipe 3 | 0.85 |


    
If the user is asking you for data that is not in the table, you should answer with "Error: <description of the error>":
	
Follow these Instructions:
- Generate a valid SQL query to execute on a postgreSQL database.
- Order results by most similar recipe name.
- **MUST** Call the RecipeDataInsights function to execute the query.
- **MUST** return the results in a markdown table format.
"""
       
    
    
    print("Creating coder agent...")
    agent_coder = ChatCompletionAgent(
        kernel=kernel,
        name=CODER_NAME,
        instructions=system_message,
        arguments=KernelArguments(settings=settings),
        #plugins=[query_db]
        )
    
    # Define a selection function to determine which agent should take the next turn.
    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
Examine the provided RESPONSE and choose the next participant.
State only the name of the chosen participant without explanation.
Never choose the participant named in the RESPONSE.

Choose only from these participants:
- {REVIEWER_NAME}: The reviewer agent who reviews and formats the data to return the appropriate response.
- {CODER_NAME}: The coder agent who writes SQL queries.

Rules:
- If RESPONSE is user input, it is {CODER_NAME}'s turn.
- If RESPONSE is by {REVIEWER_NAME}, it is {CODER_NAME}'s turn.
- If RESPONSE is by {CODER_NAME}, it is {REVIEWER_NAME}'s turn.

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    # Define a termination function where the reviewer signals completion with "yes".
    termination_keyword = "yes"

    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt=f"""
Examine the RESPONSE and determine whether the content has been deemed satisfactory.
If the content is satisfactory, respond with a single word without explanation: {termination_keyword}.
If specific suggestions are being provided, it is not satisfactory.
If no correction is suggested, it is satisfactory.

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    history_reducer = ChatHistoryTruncationReducer(target_count=5)

    # Create the AgentGroupChat with selection and termination strategies.
    chat = AgentGroupChat(
        agents=[agent_reviewer, agent_coder],
        selection_strategy=KernelFunctionSelectionStrategy(
            initial_agent=agent_coder,
            function=selection_function,
            kernel=kernel,
            result_parser=lambda result: str(result.value[0]).strip() if result.value[0] is not None else CODER_NAME,
            history_variable_name="lastmessage",
            history_reducer=history_reducer,
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[agent_reviewer],
            function=termination_function,
            kernel=kernel,
            result_parser=lambda result: termination_keyword in str(result.value[0]).lower(),
            history_variable_name="lastmessage",
            maximum_iterations=10,
            history_reducer=history_reducer,
        ),
        
    )

    print(
        "Ready! Type your input, or 'exit' to quit, 'reset' to restart the conversation."
    )

    is_complete = False
    while not is_complete:
        print()
        user_input = input("User > ").strip()
        if not user_input:
            continue

        if user_input.lower() == "exit" or user_input.lower() == "quit":
            is_complete = True
            break

        if user_input.lower() == "reset":
            await chat.reset()
            print("[Conversation has been reset]")
            continue

        # Add the current user_input to the chat
        await chat.add_chat_message(message=user_input)

        try:
            last_agent = None
            file_ids_count=0
            #current_response = ""
            async for response in chat.invoke_stream():
                if response.content is not None:
                    if last_agent != response.name:
                        # Process beginning of agent response
                        last_agent = response.name
                        # Process streamed content
                        print(f"# {response.name.upper()}", flush=True)
                    
                    if response is None or not response.name:
                        continue
                    #print(f"{response.content}")
                    #current_response += response.content
                    print(response.content, end="", flush=True)
                    
                file_ids.extend([
                    item.file_id for item in response.items if isinstance(item, StreamingFileReferenceContent) and item not in file_ids
                ])
                #print(f"File IDs: {file_ids}")
                if len(file_ids)>file_ids_count:
                    image_outputs = await print_image(agent_reviewer, file_ids)
                    for image_output in image_outputs:
                        if isinstance(image_output, io.BytesIO):
                            image = Image2.open(image_output)
                            image.show()
                    file_ids=[]
                    
        except Exception as e:
            print(f"Error during chat invocation: {e}")

        # Reset the chat's complete flag for the new conversation round.
        chat.is_complete = False

from PIL import Image as Image2  
async def print_image(agent: AzureAssistantAgent, file_ids: str):   
    outputs = []
    #img.save(output, format='JPEG')
    if file_ids:
        
        for file_id in file_ids:
            print(f"File ID: {file_id}")
            # Fetch the content of the file using the provided method
            response_content = await agent.client.files.content(file_id)    
            output = io.BytesIO(response_content.content)      
            #with open(f"./{file_id}", "wb") as f:
            #    f.write(response_content.content)      
            #image = Image2.open(output)#.to_base64(with_type=True)
            #image.show()
            outputs.append(output)
    
    return outputs
    
if __name__ == "__main__":

    #Sample questions
    #Q1: I am looking for recipe that contains chicken and peanut. Can you suggest me few similar recipes? Show me a graph
    #Q2: can you use a log scale to show the results ?
    asyncio.run(main())
