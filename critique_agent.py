"""
Critic Agent for the multi-agent system using Groq.
Evaluates research reports for quality, accuracy, and completeness.
"""

import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from base_agent import BaseAgent

class CritiqueAgent(BaseAgent):
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="CritiqueAgent",
            description="Evaluates and provides feedback on research reports using Groq",
            config=config or {}
        )
        
        self.api_key=api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment variables.")

        model_name=self.config.get("model_name", "llama-3.3-70b-versatile")
        temperature=self.config.get("temperature", 0.3)  # Lower temperature for objective criticism
        
        self.llm=ChatGroq(
            groq_api_key=self.api_key,
            model_name=model_name,
            temperature=temperature
        )

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return "topic" in input_data and "report" in input_data

    def execute(self, input_data: Dict[str, Any])-> Dict[str, Any]:
        if not self.validate_input(input_data):
            return {"success": False, "error": "Missing topic or report content"}

        topic=input_data.get("topic")
        report_content=input_data.get("report")
        
        prompt=ChatPromptTemplate.from_template("""
        You are a Senior Editor and Fact-Checker. Your task is to evaluate the following research report on: {topic}
        
        REPORT CONTENT:
        {report}
        
        CRITIQUE CRITERIA:
        1. Accuracy: Are there any logical fallacies or questionable claims?
        2. Completeness: Are there major aspects of the topic missing?
        3. Structure: Is the report easy to follow and professional?
        4. Citations: Are the references integrated correctly?
        
        OUTPUT FORMAT:
        - Score (1-10)
        - Strengths
        - Weaknesses
        - Suggestions for Improvement
        
        Tone: Critical, objective, and constructive.
        """)

        try:
            chain=prompt | self.llm
            print(f"--- {self.name} is reviewing the report for: {topic} ---")
            
            response=chain.invoke({"topic": topic, "report": report_content})
            
            if hasattr(self, 'update_execution_time'):
                self.update_execution_time()
            
            return{
                "success": True,
                "data": {
                    "feedback": response.content,
                    "is_approved": "Score: 8" in response.content or "Score: 9" in response.content or "Score: 10" in response.content
                }
            }
        except Exception as e:
            return {"success": False, "error": f"Groq execution failed: {str(e)}"}
