"""Pre-built task templates for complex multi-step research tasks."""

from dataclasses import dataclass


@dataclass
class TaskTemplate:
    name: str
    goal: str
    max_steps: int
    description: str
    example_output: str


TEMPLATES = {
    "news_aggregation": TaskTemplate(
        name="News Aggregation",
        description="Collect and summarize recent news articles on a topic",
        goal="""Search Google News for '{topic}' news from the last 7 days.
Open the top 3 results one by one.
For each article extract: headline, date, key facts, source name.
After reading all 3 articles, compile a summary of the main themes and findings.
Goal complete when you have extracted content from at least 2 articles.""",
        max_steps=25,
        example_output="3 articles summarized with themes, dates, sources",
    ),

    "multi_site_research": TaskTemplate(
        name="Multi-Site Research",
        description="Research a topic across multiple authoritative sources",
        goal="""Research '{topic}' across multiple sources.
Step 1: Search Google for '{topic}' and identify 3-4 relevant results from different domains.
Step 2: Open each result and extract key facts, data points, and quotes.
Step 3: Note any conflicting information between sources.
Step 4: Compile findings into a structured summary with source attribution.
Goal complete when you have visited at least 3 different domains and extracted content from each.""",
        max_steps=30,
        example_output="Structured report with findings from 3+ sources",
    ),

    "competitive_intelligence": TaskTemplate(
        name="Competitive Intelligence",
        description="Gather competitor information including pricing, features, and recent news",
        goal="""Gather competitive intelligence on '{company}'.
Step 1: Search Google for '{company} pricing 2024' — extract any pricing tiers found.
Step 2: Search for '{company} features' — note key product capabilities.
Step 3: Search for '{company} news 2024' — find recent announcements or funding.
Step 4: Visit {company}'s homepage and extract their value proposition.
Compile everything into a competitive profile.
Goal complete when you have pricing, features, and recent news.""",
        max_steps=30,
        example_output="Competitive profile with pricing, features, recent news",
    ),
}


def get_template(template_key: str, **kwargs) -> TaskTemplate:
    """Get a template with variables filled in."""
    template = TEMPLATES.get(template_key)
    if not template:
        raise ValueError(f"Unknown template: {template_key}. Available: {list(TEMPLATES.keys())}")
    filled_goal = template.goal.format(**kwargs)
    return TaskTemplate(
        name=template.name,
        description=template.description,
        goal=filled_goal,
        max_steps=template.max_steps,
        example_output=template.example_output,
    )
