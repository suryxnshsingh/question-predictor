"""
Conversational Agent for Question Predictor

Uses OpenAI SDK for natural language understanding and function calling.
Supports Hindi-English mixed queries (Hinglish).

Usage:
    from src.conversational_agent import QuestionPredictorAgent

    agent = QuestionPredictorAgent()
    response = agent.chat("Electric Charges ke important questions batao")
"""

import os
import json
from typing import Optional, Dict, List, Any
from openai import OpenAI
from src.prediction_service import PredictionService


# Tool definitions for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_chapter_important_topics",
            "description": "Get the most important/probable topics from a specific chapter. Use this when user asks for important questions, probable questions, or topics from a chapter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chapter_name": {
                        "type": "string",
                        "description": "The chapter name or partial match. Examples: 'Electric Charges', 'Ray Optics', 'Current Electricity', 'Electromagnetic Induction'"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of topics to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["chapter_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_topic_probability",
            "description": "Check if a specific topic/question will appear in the exam. Use this when user asks 'will X come?', 'X aayega kya?', 'chances of X'",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_name": {
                        "type": "string",
                        "description": "The topic or concept to check. Examples: 'Ohms Law', 'Electromagnetic Induction', 'Photoelectric Effect', 'Lens'"
                    }
                },
                "required": ["topic_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_overall_top_predictions",
            "description": "Get overall most important topics across all chapters. Use when user asks for general important topics without specifying a chapter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of topics to return (default 10)",
                        "default": 10
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_topic_question_types",
            "description": "Get historical question types for a topic - what format questions usually come in (MCQ, Short Answer, Long Answer, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_name": {
                        "type": "string",
                        "description": "The topic to get question types for"
                    }
                },
                "required": ["topic_name"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are a helpful Physics exam preparation assistant for Class 12 board exams.
You help students understand which topics are most likely to appear in exams based on historical patterns.

You can:
1. Tell students the most important topics from any chapter
2. Check if a specific topic is likely to appear
3. Provide ALL predicted questions that are likely to come in the exam

Guidelines:
- Be encouraging but realistic about predictions
- Always mention that predictions are based on historical patterns
- If probability is high (>70%), be confident. If medium (50-70%), mention it's likely. If low (<50%), mention it's less predictable.
- You can respond in Hindi, English, or Hinglish based on how the user asks
- CRITICAL: List ALL questions from the predicted_questions field. These are the actual questions our ML model predicts will appear.
- Format each question clearly with type, year and marks
- Group questions by topic
- Do NOT summarize or skip questions - show the complete list

Response format for each topic:
**Topic Name** - XX% probability (Frequency: X/8 years)

Predicted Questions:
1. [Type, Year, Marks] "Full question text here"
2. [Type, Year, Marks] "Full question text here"
... (list ALL questions)
"""


class QuestionPredictorAgent:
    """Conversational agent for the Question Predictor system"""

    def __init__(self, subject: str = 'physics', api_key: Optional[str] = None, prediction_service: Optional[PredictionService] = None):
        """
        Initialize the agent

        Args:
            subject: Subject for predictions (default: physics)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            prediction_service: Optional existing PredictionService to reuse
        """
        self.subject = subject
        self.prediction_service = prediction_service or PredictionService(subject=subject)

        # Initialize OpenAI client
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")

        self.client = OpenAI(api_key=api_key)
        self.conversation_history: List[Dict] = []
        self.model = "gpt-4.1-mini"  # Cost-effective, good for function calling

    def _execute_function(self, function_name: str, arguments: Dict) -> str:
        """Execute a function and return results as string"""

        if function_name == "get_chapter_important_topics":
            chapter = arguments.get("chapter_name", "")
            top_n = arguments.get("top_n", 5)

            results = self.prediction_service.get_chapter_topics(chapter, top_n=top_n)

            if not results:
                return json.dumps({
                    "status": "not_found",
                    "message": f"No topics found for chapter matching '{chapter}'. Try a different chapter name."
                })

            formatted = []
            for r in results:
                # Get ALL questions for this topic
                all_questions = []
                questions_by_type = r.get("questions_by_type", {})
                for qtype, qs in questions_by_type.items():
                    for q in qs:  # ALL questions, not just samples
                        all_questions.append({
                            "question": q["question"],
                            "type": qtype,
                            "marks": q["marks"],
                            "year": q["year"]
                        })

                formatted.append({
                    "rank": r["rank"],
                    "topic": r["topic"],
                    "probability": f"{r['probability_score']*100:.0f}%",
                    "confidence": r["confidence"],
                    "frequency": r["frequency"],
                    "most_common_type": r["most_common_question_type"],
                    "reasoning": r["reasoning"],
                    "predicted_questions": all_questions
                })

            return json.dumps({
                "status": "success",
                "chapter_query": chapter,
                "topics": formatted
            })

        elif function_name == "check_topic_probability":
            topic = arguments.get("topic_name", "")

            result = self.prediction_service.check_topic_probability(topic)

            if not result:
                return json.dumps({
                    "status": "not_found",
                    "message": f"Topic '{topic}' not found. Try variations like 'Ohms Law', 'Electric Field', 'Lens', etc."
                })

            # Get ALL questions for this topic
            all_questions = []
            questions_by_type = result.get("questions_by_type", {})
            for qtype, qs in questions_by_type.items():
                for q in qs:  # ALL questions
                    all_questions.append({
                        "question": q["question"],
                        "type": qtype,
                        "marks": q["marks"],
                        "year": q["year"]
                    })

            return json.dumps({
                "status": "success",
                "topic": result["topic"],
                "probability": f"{result['probability_score']*100:.0f}%",
                "confidence": result["confidence"],
                "frequency": result["frequency"],
                "last_appeared": result["last_appeared"],
                "most_common_type": result["most_common_question_type"],
                "all_question_types": list(result["question_type_distribution"].keys()),
                "reasoning": result["reasoning"],
                "predicted_questions": all_questions
            })

        elif function_name == "get_overall_top_predictions":
            top_n = arguments.get("top_n", 10)

            results = self.prediction_service.get_top_predictions(top_n=top_n)

            formatted = []
            for r in results:
                formatted.append({
                    "rank": r["rank"],
                    "topic": r["topic"],
                    "probability": f"{r['probability_score']*100:.0f}%",
                    "confidence": r["confidence"],
                    "frequency": r["frequency"]
                })

            return json.dumps({
                "status": "success",
                "topics": formatted
            })

        elif function_name == "get_topic_question_types":
            topic = arguments.get("topic_name", "")

            result = self.prediction_service.check_topic_probability(topic)

            if not result:
                return json.dumps({
                    "status": "not_found",
                    "message": f"Topic '{topic}' not found."
                })

            return json.dumps({
                "status": "success",
                "topic": result["topic"],
                "question_types": result["question_type_distribution"],
                "most_common": result["most_common_question_type"],
                "avg_marks": result["avg_marks_per_type"],
                "years_by_type": result["question_type_years"]
            })

        return json.dumps({"status": "error", "message": f"Unknown function: {function_name}"})

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return response

        Args:
            user_message: User's question in natural language

        Returns:
            Agent's response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Build messages with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message

        # Check if function call is needed
        if assistant_message.tool_calls:
            # Add assistant message with tool calls
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            # Execute each function call
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                # Execute function
                result = self._execute_function(function_name, arguments)

                # Add function result to history
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            # Get final response from model
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history

            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            final_message = final_response.choices[0].message.content

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": final_message
            })

            return final_message
        else:
            # No function call, just return the response
            content = assistant_message.content or ""
            self.conversation_history.append({
                "role": "assistant",
                "content": content
            })
            return content

    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history"""
        return self.conversation_history


def run_cli():
    """Run interactive CLI chat"""
    print("=" * 60)
    print("Physics Question Predictor - Conversational Agent")
    print("=" * 60)
    print("\nExamples:")
    print("  - 'Electric Charges ke important questions batao'")
    print("  - 'Ohm's Law aayega kya?'")
    print("  - 'Most important topics for physics board exam'")
    print("  - 'Ray Optics chapter se kya padhu?'")
    print("\nType 'quit' to exit, 'reset' to clear history\n")

    try:
        agent = QuestionPredictorAgent()
    except ValueError as e:
        print(f"Error: {e}")
        print("Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        return

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            if user_input.lower() == 'reset':
                agent.reset_conversation()
                print("Conversation history cleared.")
                continue

            response = agent.chat(user_input)
            print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    run_cli()
