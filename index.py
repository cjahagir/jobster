import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# Get API keys from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input model
class JobQuery(BaseModel):
    current_position: str
    current_location: str
    expected_position: str
    expected_location: str
    expected_salary: str

# Create the agent
memory = MemorySaver()
model = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)
search = TavilySearchResults(max_results=10, tavily_api_key=TAVILY_API_KEY)
agent_executor = create_react_agent(model, [search], checkpointer=memory)

# Improved regex pattern for job extraction
job_pattern = re.compile(r'\{\s*"title":\s*"([^"]+)",\s*"company":\s*"([^"]+)",\s*"location":\s*"([^"]*)",\s*"link":\s*"([^"]+)"')

def extract_job_listings(text):
    job_details = []
    matches = job_pattern.findall(text)
    for match in matches:
        job_title, company, location, job_link = match
        job_details.append({
            "title": job_title.strip(),
            "company": company.strip(),
            "location": location.strip(),
            "link": job_link.strip()
        })

    # Ensure 10 entries (fill empty ones with placeholder)
    while len(job_details) < 10:
        job_details.append({
            "title": "No structured job listings found",
            "company": "",
            "location": "",
            "link": ""
        })

    return job_details

@app.post("/search_jobs")
def search_jobs(query: JobQuery):
    prompt = (
        f"I am a {query.current_position}, working in {query.current_location}. "
        f"I want to find **exactly 10 job listings** for a {query.expected_position} in {query.expected_location}. "
        f"My expected salary is {query.expected_salary}. "
        f"Please provide the results in **valid JSON format** with the following structure:\n\n"
        f'{{\n'
        f'    "job_listings": [\n'
        f'        {{ "title": "Job Title", "company": "Company Name", "location": "Location", "link": "Job Link" }},\n'
        f'        {{ "title": "Job Title", "company": "Company Name", "location": "Location", "link": "Job Link" }},\n'
        f'        ...\n'
        f'    ]\n'
        f'}}\n\n'
        f"Important Instructions:\n"
        f"- Ensure the JSON is **well-formed** with no extra text or comments.\n"
        f"- The **company name** should be the actual employer, not the job board.\n"
        f"- The **location** should match the specified city/region (e.g., Bengaluru, Karnataka).\n"
        f"- The **job link** should be a **direct URL** to the job posting.\n"
        f"- Return **exactly 10 job listings**. DO NOT GIVE ANY ADITIONAL COMMENTS"
    )

    config = {"configurable": {"thread_id": "abc123"}}

    # Send prompt to LLM
    response = ""
    for step in agent_executor.stream(
        {"messages": [HumanMessage(content=prompt)]},
        config,
        stream_mode="values",
    ):
        response = step["messages"][-1].content

    print(response)  # Debugging: Print raw response

    # Extract job listings from response
    job_listings = extract_job_listings(response)

    # If fewer than 10 results, attempt a second query
    if len(job_listings) < 10:
        additional_prompt = (
            f"Please provide additional job listings in the same JSON format."
        )

        for step in agent_executor.stream(
            {"messages": [HumanMessage(content=additional_prompt)]},
            config,
            stream_mode="values",
        ):
            additional_response = step["messages"][-1].content
            additional_jobs = extract_job_listings(additional_response)
            job_listings.extend(additional_jobs)

    # Ensure 10 unique entries
    job_listings = job_listings[:10]

    return {"job_listings": job_listings}