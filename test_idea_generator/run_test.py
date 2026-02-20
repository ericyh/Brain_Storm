from agents_vs2 import run_supervised_generation
from dotenv import load_dotenv

load_dotenv()

profile = {
    "location": "UK",
    "capital_available": 15000,
    "risk_tolerance": "moderate",
    "time_available_per_week": 20,
}

skills_text = """
Python developer.
Experience with automation, data scraping, and small business operations.
Comfortable with B2B sales.
"""

result = run_supervised_generation(
    profile=profile,
    query="boring B2B businesses in logistics",
    skills_text=skills_text,
    worker_count=4,     # keep small for testing
    critic_count=3,
    top_k=3,
)

print(result["shortlist"])