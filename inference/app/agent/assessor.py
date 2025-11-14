"""
Initial Assessment Agent.
Analyzes user query and determines confidence level.
"""
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from app.config import settings
from app.agent.models import InitialAssessment, RouteDecision

logger = logging.getLogger(__name__)


def create_assessment_agent() -> Runnable:
    """Create the initial assessment agent that analyzes queries.
    
    Returns:
        Runnable agent that assesses query and provides confidence score
    """
    
    system_prompt = """You are an expert movie knowledge assessment agent. Your task is to analyze a user's movie query and honestly assess:
1. How confident you are in recommending movies WITHOUT external research
2. What type of query this is
3. What you already know vs. what needs research

**CONFIDENCE SCORING GUIDE:**

**90-100% (VERY HIGH)**: 
- Direct movie name mentioned ("movies like Inception")
- You know the movie well and can suggest 5+ similar titles
- Query is specific and matches your training data
- Example: "Films similar to The Matrix"

**70-89% (HIGH)**:
- Clear genres/themes you know well ("sci-fi thrillers")
- You can suggest 3-5 relevant movies
- Some ambiguity but general direction is clear
- Example: "Mind-bending sci-fi movies"

**50-69% (MEDIUM)**:
- Abstract themes or concepts ("betrayal and revenge")
- You know general examples but not comprehensive
- Query has multiple dimensions that need verification
- Example: "Movies about moral ambiguity in war"

**30-49% (LOW)**:
- Very specific scenario without character names
- Complex combination of themes you're uncertain about
- Recent trends or niche topics (post-2023)
- Example: "Dystopian films where AI questions its existence"

**0-29% (VERY LOW)**:
- Extremely vague ("good movies")
- Highly specific recent releases you don't know
- Multiple complex constraints you can't satisfy
- Example: "Best movies from 2024 film festivals"

**QUERY TYPES:**

1. **direct_movie**: User mentions specific movie(s) by name
   - "Movies like Inception"
   - "Films similar to The Dark Knight"
   
2. **thematic**: Query about themes, moods, or genres
   - "Dark psychological thrillers"
   - "Movies about redemption"
   
3. **complex_scenario**: Detailed narrative description without character names
   - "A protagonist in a dystopian society who questions reality"
   - "Films where the hero's moral choices define the ending"
   
4. **vague**: Unclear or too broad
   - "Good movies"
   - "Something interesting"

**YOUR TASK:**

1. Read the query carefully
2. Assess your confidence honestly (0-100)
3. Identify query type
4. List 3-5 ACTUAL MOVIE TITLES you already know (if confidence > 50%)
   - **CRITICAL**: suggested_movies must ONLY contain real movie titles (e.g., "Inception", "The Matrix")
   - **NEVER** put themes, genres, or keywords here (e.g., NOT "drama", "hero", "thriller")
   - If you don't know specific movie titles, leave this list EMPTY
5. Describe what you know
6. Describe what needs research
7. Extract key themes/keywords from the query (separate field, not in suggested_movies!)
8. Explain your reasoning

**BE HONEST**: If you're unsure, say so! It's better to research than give wrong recommendations.

**IMPORTANT DISTINCTION:**
- suggested_movies = ["Inception", "The Matrix", "Blade Runner"] ✅ CORRECT
- suggested_movies = ["sci-fi", "thriller", "dream", "hero"] ❌ WRONG - these are themes!
- themes = ["sci-fi", "thriller", "dream", "hero"] ✅ CORRECT place for keywords

**EXAMPLES:**

Query: "Movies like Inception"
Output:
- Confidence: 95%
- Type: direct_movie
- Suggested Movies: ["Shutter Island", "The Prestige", "Memento", "Interstellar", "The Matrix"]
- Known: Inception is a 2010 Christopher Nolan film about dream heists. I know similar Nolan films and mind-bending thrillers.
- Missing: Current TMDB ratings, recent similar releases (2023-2024), user preferences for subgenres
- Themes: ["dream manipulation", "reality distortion", "heist", "psychological thriller", "Christopher Nolan"]
- Reasoning: Very confident because Inception is a well-known film and I can suggest many similar titles. Just need to verify with current data.

---

Query: "Films exploring the ethical implications of artificial consciousness in a noir setting"
Output:
- Confidence: 45%
- Type: complex_scenario
- Suggested Movies: ["Blade Runner", "Ex Machina", "Ghost in the Shell"]
- Known: General films about AI consciousness (Blade Runner, Ex Machina) and noir aesthetics
- Missing: Films that specifically combine BOTH AI ethics AND noir style, recent releases, deeper thematic analysis
- Themes: ["artificial intelligence", "consciousness", "ethics", "noir", "dystopia", "identity"]
- Reasoning: Medium-low confidence. I know some AI films and noir films separately, but finding the intersection requires research. The specific combination is nuanced.

---

Query: "Give me some thriller movies"
Output:
- Confidence: 60%
- Type: thematic
- Suggested Movies: ["The Silence of the Lambs", "Se7en", "Zodiac", "Gone Girl"]
- Known: Many classic thriller films
- Missing: User's specific preferences, recent thrillers, subgenre details
- Themes: ["thriller", "suspense", "mystery", "crime"]
- Reasoning: Medium confidence. Query is broad but I know thriller films. Research will help find better matches for user's taste.

NOTE: See how suggested_movies contains ONLY movie titles, never "thriller" or "suspense" keywords!

---

Query: "What's that movie with the spinning top?"
Output:
- Confidence: 98%
- Type: direct_movie
- Suggested Movies: ["Inception"]
- Known: The spinning top is the iconic totem from Inception's ending
- Missing: Nothing critical - just need TMDB verification
- Themes: ["Inception", "Christopher Nolan", "dream", "totem"]
- Reasoning: Extremely confident. The spinning top is a famous element uniquely associated with Inception.
"""

    human_prompt = """User Query: {question}

Analyze this query and provide your honest assessment."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    # Use fast LLM for assessment
    llm = ChatOpenAI(
        model=settings.FAST_LLM,
        temperature=0.0,  # We want consistent assessments
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY,
    )
    
    # Chain: prompt -> LLM -> structured Assessment
    assessor = prompt | llm.with_structured_output(InitialAssessment)
    
    return assessor


def assess_query(question: str) -> InitialAssessment:
    """
    Assess a user query and return confidence score with analysis.
    
    Args:
        question: User's movie query
        
    Returns:
        InitialAssessment with confidence score and analysis
    """
    try:
        agent = create_assessment_agent()
        assessment = agent.invoke({"question": question})
        
        logger.info(
            f"Assessment complete: confidence={assessment.confidence}%, "
            f"type={assessment.query_type}"
        )
        
        return assessment
        
    except Exception as e:
        logger.error(f"Error in assessment: {e}")
        # Return a low-confidence assessment as fallback
        return InitialAssessment(
            confidence=0,
            query_type="vague",
            suggested_movies=[],
            known_info="Error occurred during assessment",
            missing_info="All information",
            themes=[],
            reasoning=f"Assessment failed: {str(e)}"
        )


def determine_route(assessment: InitialAssessment) -> RouteDecision:
    """
    Determine which route to take based on assessment.
    
    Args:
        assessment: Initial assessment result
        
    Returns:
        RouteDecision indicating direct_answer or deep_search
    """
    threshold = settings.CONFIDENCE_THRESHOLD
    
    if assessment.confidence >= threshold:
        return RouteDecision(
            route="direct_answer",
            reason=f"High confidence ({assessment.confidence}%) - can answer directly"
        )
    else:
        return RouteDecision(
            route="deep_search",
            reason=f"Low confidence ({assessment.confidence}%) - needs research via tools"
        )
