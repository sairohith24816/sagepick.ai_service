"""
Answer generation agents (direct and final).
"""
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


def generate_direct_answer(question: str, assessment: dict) -> dict:
    """
    Generate direct answer using fast LLM (high confidence path).
    
    Args:
        question: User's query
        assessment: Assessment data with suggested movies
        
    Returns:
        Dict with answer and movie list
    """
    try:
        system_prompt = """You are a movie recommendation expert. The user asked about movies and we have high confidence in our answer.

Provide a helpful response that:
1. Briefly acknowledges their query
2. Lists 5-10 recommended movies with short descriptions
3. Is concise and friendly

Format your response naturally, don't use numbered lists unless necessary."""

        human_prompt = """User Query: {question}

Suggested Movies: {suggested_movies}
Themes: {themes}

Provide your recommendation."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        llm = ChatOpenAI(
            model=settings.FAST_LLM,
            temperature=0.3,
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
        )
        
        chain = prompt | llm
        
        response = chain.invoke({
            "question": question,
            "suggested_movies": ", ".join(assessment.get("suggested_movies", [])),
            "themes": ", ".join(assessment.get("themes", []))
        })
        
        return {
            "answer": response.content,
            "movies": assessment.get("suggested_movies", [])
        }
        
    except Exception as e:
        logger.error(f"Direct answer generation error: {e}")
        return {
            "answer": "I apologize, but I encountered an error generating recommendations.",
            "movies": []
        }


def generate_final_answer(question: str, assessment: dict, research_data: dict) -> dict:
    """
    Generate final answer using smart LLM with vector research data.
    
    Args:
        question: User's query
        assessment: Initial assessment
        research_data: Results from semantic vector search
    
    Returns:
        Dict with answer and movie list
    """
    try:
        system_prompt = """You are an expert movie recommendation system. You have researched the user's query using a curated semantic vector database of movies.

Your task:
1. Synthesize the vector search findings
2. Recommend 5-10 movies that best match the query
3. Provide brief, engaging descriptions grounded in the provided metadata
4. Be confident and helpful

Focus on quality over quantity. Only recommend movies present in the research batch."""

        human_prompt = """User Query: {question}

== RESEARCH RESULTS ==

Vector Search Summary:
{vector_results}

Structured Movie Hits:
{movie_details}

====================

Based on this research, provide your movie recommendations."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        llm = ChatOpenAI(
            model=settings.SMART_LLM,
            temperature=0.4,
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
        )
        
        chain = prompt | llm
        
        # Format research data
        vector_results = research_data.get("vector_summary", "No vector results")
        movie_details = research_data.get("movie_details", [])
        
        formatted_details = []
        for movie in movie_details:
            title = movie.get("title", "Unknown")
            overview = movie.get("overview", "No overview available")
            genres = movie.get("genres", "")
            year = movie.get("release_date", "")[:4]
            score = movie.get("score", 0)
            formatted_details.append(
                f"- {title} ({year or 'N/A'}) | genres: {genres} | relevance: {score:.2f}\n  {overview[:200]}..."
            )
        structured_context = "\n".join(formatted_details) if formatted_details else "No structured movie hits available."
        
        response = chain.invoke({
            "question": question,
            "vector_results": vector_results,
            "movie_details": structured_context
        })
        
        # Extract movie names from research
        movies = research_data.get("movie_names", [])
        
        return {
            "answer": response.content,
            "movies": movies
        }
        
    except Exception as e:
        logger.error(f"Final answer generation error: {e}")
        return {
            "answer": "I apologize, but I encountered an error generating recommendations.",
            "movies": []
        }
