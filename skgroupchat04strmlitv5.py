# Copyright (c) Microsoft. All rights reserved.

import streamlit as st
import asyncio
import time
import json
import requests
from semantic_kernel.agents import Agent, ChatCompletionAgent, GroupChatOrchestration, RoundRobinGroupChatManager,ConcurrentOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.functions import kernel_function
from typing import Dict, Any, Union

def azure_ai_search_plugin(
    query: str,
    select: str = "chunk_id,parent_id,chunk,title",
    k: int = 100,
    #semantic_configuration: str = "cosmos-rag-semantic-configuration",
    semantic_configuration: str = "rag-dmindexbrazilnewsbot-semantic-configuration",
    vector_field: str = "text_vector",
    query_type: str = "semantic",
    query_language: str = "en-GB",
    timeout: int = 40,
) -> Union[Dict[str, Any], None]:
    """
    Execute Azure AI Search with semantic + vector search, returning a Python dict
    with total_count, results, search_id, and semantic_answers.
    """
    search_endpoint = sendpoint # Azure Search endpoint
    search_api_key = sapi_key # Azure Search API key
    index_name = sindex_name  # Updated index name from Streamlit input

    if not search_endpoint or not search_api_key:
        print("Azure AI Search endpoint and API key must be set.")
        return None

    if not query or not query.strip():
        print("Search query is required.")
        return None

    endpoint = f"{search_endpoint}/indexes/{index_name}/docs/search?api-version=2024-05-01-Preview"
    headers = {"Content-Type": "application/json", "api-key": search_api_key}
    payload = {
        "search": query,
        "select": select,
        "vectorQueries": [
            {
                "kind": "text",
                "text": query,
                "fields": vector_field,
                "k": k,
            }
        ],
        "queryType": query_type,
        "semanticConfiguration": semantic_configuration,
        "queryLanguage": query_language,
        "top": k,
    }

    try:
        print(f"Running Azure AI Search for query: '{query}'")
        response = requests.post(
            endpoint, headers=headers, data=json.dumps(payload), timeout=timeout
        )

        if response.status_code != 200:
            print(
                f"Search failed with status {response.status_code}: {response.text}"
            )
            return None

        data = response.json()
        return {
            "total_count": data.get("@odata.count", len(data.get("value", []))),
            "results": data.get("value", []),
            "search_id": data.get("@search.searchId"),
            "semantic_answers": data.get("@search.answers", []),
        }

    except Exception as e:
        print(f"Exception during Azure AI Search: {str(e)}")
        return None

class AzureAISearchPlugin:
    @kernel_function
    def search(self, query: str) -> str:
        """
        Perform a search using Azure AI Search.
        Replace this stub with actual Azure Search SDK/API calls.
        """
        
        results = azure_ai_search_plugin(query)
        if results is None:
            return json.dumps({"error": "search_failed"})
        # TODO: Replace with real Azure AI Search logic
        # Example: Use azure-search-documents package to query your index
        # For now, return a mock result
        print(f"Searching Azure AI Search for: {query}")
        count = len(results.get("results", []))
        print(f"Search returned {count} documents")
        print(f"Results: ={results}")
        return json.dumps(results)
        

# --- AGENT SETUP (same as your original code) ---
def get_agents(deployment_name, api_key, endpoint):
    azure_service = AzureChatCompletion(
        deployment_name=deployment_name,
        api_key=api_key,
        endpoint=endpoint,
    )
    
    rag_agent = ChatCompletionAgent(
        name="RAGAgent",
        description="An AI assistant designed to support senior managers and leaders by answering questions based on a curated knowledge base of news articles from multiple reputable sources.",
        instructions="Respond concisely by default; provide exact analytics for data questions, descriptive insights for general queries, and detailed, multi-perspective reports when requested. include title in the reference list",
        service=azure_service,
        plugins=[AzureAISearchPlugin()],
    )
    # analyst = ChatCompletionAgent(
    #     name="DataAnalyst",
    #     description="A Data analyst.",
    #     instructions="You are a Senior Data anlyst who can generate anlytics reports and statements based on the details available.",
    #     service=azure_service,
    # )
    analyticsagent = ChatCompletionAgent(
        name="AnalyticsAgents",
        description="An analytics assistant which extracts datapoints and provides insights based on the data.",
        instructions="You are an excellent data analyst. You thourghly analyse the contents and focused on available data points and prepare statistcal figures.",
        service=azure_service,
        plugins=[AzureAISearchPlugin()]
    )
    return [rag_agent, analyticsagent]

# --- STREAMLIT UI ---


# Azure OpenAI configuration inputs
with st.sidebar:
    deployment_name = st.text_input("Azure Deployment Name", value="azure doplyment name")
    api_key = st.text_input("Azure API Key", type="password", value="")
    endpoint = st.text_input("Azure Endpoint", value="https://xxxxxxxxxxxxx.openai.azure.com/")
    sendpoint = st.text_input("Azure Search Endpoint", value="https://xxxxxxxxxxxxxxxx.search.windows.net")
    sapi_key = st.text_input("Azure Search Key", value="",type="password")
    sindex_name = st.text_input("Azure Search Name", value="rag-xxxxxxxxxxxx")

st.title("News bot agent Demo")
st.write("This demo runs a group chat between agents to iteratively refine the answers for  news bot")

# User input for the task
task = st.text_area("Enter your task for the group:", "What are the top news on ecigarettes?")

# Button to run the orchestration
if st.button("Run the analysis"):
    conversation = []
    conversation_placeholder = st.empty()  # For streaming conversation

    def agent_response_callback(message: ChatMessageContent):
        # Animate the message word by word
        display_message = f"**{message.name}**: "
        words = message.content.split()
        animated = display_message
        for word in words:
            animated += word + " "
            conversation_placeholder.markdown("\n\n".join(conversation + [animated]))
            time.sleep(0.02)  # Adjust speed as desired
        # Add the full message to the conversation history
        conversation.append(f"**{message.name}**: {message.content}")
        conversation_placeholder.markdown("\n\n".join(conversation))

    async def run_orchestration():
        agents = get_agents(deployment_name, api_key, endpoint)
        # group_chat_orchestration = GroupChatOrchestration(
        #     members=agents,
        #     manager=RoundRobinGroupChatManager(max_rounds=6),
        #     agent_response_callback=agent_response_callback,
        # )
        concurrent_orchestration = ConcurrentOrchestration(members=agents,agent_response_callback=agent_response_callback)
        runtime = InProcessRuntime()
        runtime.start()
        orchestration_result = await concurrent_orchestration.invoke(
            task=task,
            runtime=runtime,
        )
        value = await orchestration_result.get()
        await runtime.stop_when_idle()
        return value

    # Run the async orchestration
    result = asyncio.run(run_orchestration())

    # Final conversation display (in case any last message was missed)
    conversation_placeholder.markdown("\n\n".join(conversation))

    # st.subheader("Final Result")
    # st.success(result)
