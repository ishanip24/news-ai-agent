from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
from duckduckgo_search import DDGS
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

@tool
def get_news_articles(
    keywords: str,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: str | None = None
) -> list[dict[str, str]]:
    """A tool that gets 3 news articles (title, url, date) for a given set of keywords.
    
    Args:
        keywords: keywords for query.
        region: us-en, uk-en, ru-ru, etc. Defaults to "us-en".
        safesearch: on, moderate, off. Defaults to "moderate".
        timelimit: d, w, m. Defaults to None.
    
    Returns:
        List of dictionaries with 'title', 'url', and 'date' keys.
    """
    results = DDGS().news(
        keywords=keywords,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
        max_results=3
    )
    articles = []
    for result in results:
        articles.append({
            "title": result.get("title", "No title"),
            "url": result.get("url", "No URL"),
            "date": result.get("date", "No date")
        })
    return articles

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_news_articles, get_current_time_in_timezone], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()