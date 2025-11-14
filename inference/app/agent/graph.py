"""
LangGraph workflow for movie recommendation agent.
Simple and fast architecture with minimal nodes.
"""
import logging
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

from app.agent.assessor import assess_query, determine_route
from app.agent.answer_agent import generate_direct_answer, generate_final_answer
from app.agent.research_agent import deep_search_research

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the agent graph."""
    question: str
    messages: Annotated[Sequence[BaseMessage], "messages"]
    assessment: dict
    route: str
    research_data: dict
    answer: str
    recommended_movies: list


def assessment_node(state: AgentState) -> AgentState:
    """Assess the query and determine route."""
    logger.info("=== ASSESSMENT NODE ===")
    
    assessment = assess_query(state["question"])
    route_decision = determine_route(assessment)
    
    state["assessment"] = assessment.dict()
    state["route"] = route_decision.route
    
    logger.info(f"Confidence: {assessment.confidence}%, Route: {route_decision.route}")
    
    return state


def direct_answer_node(state: AgentState) -> AgentState:
    """Generate direct answer using fast LLM (high confidence)."""
    logger.info("=== DIRECT ANSWER NODE ===")
    
    result = generate_direct_answer(
        question=state["question"],
        assessment=state["assessment"]
    )
    
    state["answer"] = result["answer"]
    state["recommended_movies"] = result["movies"]
    
    return state


async def research_node(state: AgentState) -> AgentState:
    """Deep search using the vector search tool."""
    logger.info("=== RESEARCH NODE ===")
    
    research_data = await deep_search_research(
        question=state["question"],
        themes=state["assessment"]["themes"]
    )
    
    state["research_data"] = research_data
    
    return state


def final_answer_node(state: AgentState) -> AgentState:
    """Generate final answer using smart LLM with research data."""
    logger.info("=== FINAL ANSWER NODE ===")
    
    result = generate_final_answer(
        question=state["question"],
        assessment=state["assessment"],
        research_data=state["research_data"]
    )
    
    state["answer"] = result["answer"]
    state["recommended_movies"] = result["movies"]
    
    return state


def route_after_assessment(state: AgentState) -> str:
    """Route based on assessment confidence."""
    if state["route"] == "direct_answer":
        return "direct_answer"
    else:
        return "research"


def create_agent_graph():
    """
    Create the LangGraph workflow.
    
    Flow:
    1. Assessment (fast LLM) -> confidence score
    2. If confidence >= 70: Direct Answer (fast LLM) -> END
    3. If confidence < 70: Research (vector search) -> Final Answer (smart LLM) -> END
    """
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("assessment", assessment_node)
    workflow.add_node("direct_answer", direct_answer_node)
    workflow.add_node("research", research_node)
    workflow.add_node("final_answer", final_answer_node)
    
    # Set entry point
    workflow.set_entry_point("assessment")
    
    # Add conditional routing after assessment
    workflow.add_conditional_edges(
        "assessment",
        route_after_assessment,
        {
            "direct_answer": "direct_answer",
            "research": "research"
        }
    )
    
    # Direct answer goes to END
    workflow.add_edge("direct_answer", END)
    
    # Research goes to final answer
    workflow.add_edge("research", "final_answer")
    
    # Final answer goes to END
    workflow.add_edge("final_answer", END)
    
    return workflow.compile()


# Global graph instance
agent_graph = create_agent_graph()


async def run_agent(question: str) -> dict:
    """
    Run the agent graph with a user question.
    
    Args:
        question: User's movie query
        
    Returns:
        Dict with answer and recommended movies
    """
    try:
        initial_state = {
            "question": question,
            "messages": [],
            "assessment": {},
            "route": "",
            "research_data": {},
            "answer": "",
            "recommended_movies": []
        }
        
        # Run the graph
        final_state = await agent_graph.ainvoke(initial_state)
        
        return {
            "answer": final_state["answer"],
            "movies": final_state["recommended_movies"],
            "confidence": final_state["assessment"].get("confidence", 0),
            "route_taken": final_state["route"]
        }
        
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        return {
            "answer": f"I encountered an error: {str(e)}",
            "movies": [],
            "confidence": 0,
            "route_taken": "error"
        }
