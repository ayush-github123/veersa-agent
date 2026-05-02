"""
Writer Agent for the multi-agent system using Groq.
Synthesizes research content into a structured, professional report.
"""

import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from base_agent import BaseAgent 

class WriterAgent(BaseAgent):
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="WriterAgent",
            description="Drafts structured research reports from gathered data using Groq",
            config=config or {}
        )
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment variables.")

        model_name = self.config.get("model_name", "llama-3.3-70b-versatile")
        temperature = self.config.get("temperature", 0.5)
        
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=model_name,
            temperature=temperature
        )

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return "topic" in input_data and "research_data" in input_data

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_input(input_data):
            return {"success": False, "error": "Missing topic or research_data"}

        topic = input_data.get("topic")
        research_results = input_data.get("research_data", [])
        
        context_blocks = [
            f"Source: {item.get('url')}\nContent: {item.get('content')}" 
            for item in research_results
        ]
        full_context = "\n\n---\n\n".join(context_blocks)

        prompt = ChatPromptTemplate.from_template("""
        You are a Professional Research Analyst. Write a comprehensive report on: {topic}
        
        RESEARCH DATA:
        {context}
        
        STRUCTURE:
        1. Introduction
        2. Key Findings
        3. Technical Insights
        4. Conclusion
        5. References (URLs only)
        
        Tone: Professional and objective. Respond in Markdown format.
        """)

        try:
            chain = prompt | self.llm
            print(f"--- {self.name} is drafting the report for: {topic} ---")
            
            response = chain.invoke({"topic": topic, "context": full_context})
            
            if hasattr(self, 'update_execution_time'):
                self.update_execution_time()
            
            return {
                "success": True,
                "data": {"report": response.content}
            }
        except Exception as e:
            return {"success": False, "error": f"Groq execution failed: {str(e)}"}