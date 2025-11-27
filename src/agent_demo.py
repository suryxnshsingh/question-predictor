"""
Demo Agent - Works without OpenAI API key
Uses simple intent matching for demonstration

For production, use conversational_agent.py with OpenAI API
"""

import re
from typing import Optional, Dict, List, Tuple
from src.prediction_service import PredictionService


class DemoAgent:
    """
    Simple demo agent with pattern-based intent detection
    No API key required - good for testing and demonstration
    """

    def __init__(self, subject: str = 'physics'):
        self.subject = subject
        self.service = PredictionService(subject=subject)

        # Chapter name variations for fuzzy matching
        self.chapter_aliases = {
            'electric charges': ['electric charge', 'charges', 'electrostatics', 'coulomb'],
            'current electricity': ['current', 'electricity', 'ohm', 'resistance'],
            'electromagnetic induction': ['em induction', 'induction', 'faraday', 'emf'],
            'ray optics': ['optics', 'mirror', 'lens', 'reflection', 'refraction'],
            'wave optics': ['interference', 'diffraction', 'polarization', 'young'],
            'semiconductor': ['diode', 'transistor', 'logic gate', 'pn junction'],
            'magnetism': ['magnetic', 'magnet', 'biot savart', 'ampere'],
            'electromagnetic waves': ['em waves', 'em spectrum', 'radio waves'],
            'dual nature': ['photoelectric', 'photon', 'de broglie'],
            'atoms': ['bohr', 'atomic', 'hydrogen spectrum'],
            'nuclei': ['nuclear', 'radioactive', 'binding energy'],
            'capacitance': ['capacitor', 'dielectric', 'potential'],
        }

        # Topic name variations
        self.topic_aliases = {
            'ohms law': ['ohm', 'ohm law', 'resistance law'],
            'kirchhoff': ['kirchoff', 'kirchhoffs law', 'junction rule', 'loop rule'],
            'electromagnetic induction': ['em induction', 'faraday law', 'induced emf'],
            'lens': ['convex lens', 'concave lens', 'lens formula'],
            'photoelectric effect': ['photoelectric', 'photon effect', 'work function'],
            'pn junction': ['p-n junction', 'diode', 'semiconductor diode'],
        }

    def _detect_intent(self, message: str) -> Tuple[str, Dict]:
        """
        Detect user intent from message

        Returns:
            (intent_type, extracted_params)
        """
        message_lower = message.lower()

        # Check if any chapter name is mentioned first
        has_chapter = False
        for chapter_key in self.chapter_aliases.keys():
            if chapter_key in message_lower:
                has_chapter = True
                break
        if not has_chapter:
            for aliases in self.chapter_aliases.values():
                if any(alias in message_lower for alias in aliases):
                    has_chapter = True
                    break

        # Intent: Overall top predictions (only if NO chapter mentioned)
        if not has_chapter:
            overall_patterns = [
                r'(?:most|top|sabse)\s*(?:important|probable|expected|zyada)',
                r'(?:overall|general)\s*(?:important|topics)',
                r'sab(?:se)?\s*(?:important|zyada)',
                r'(?:important|probable)\s*(?:topics|questions)\s*(?:batao|bata|tell)',
                r'physics\s*(?:ke|ka)?\s*(?:important|top)',
            ]

            for pattern in overall_patterns:
                if re.search(pattern, message_lower):
                    return 'overall_topics', {}

        # Intent: Check if topic will appear
        probability_patterns = [
            r'(.*?)\s*(aayega|ayega|aega|appear|come|chances|probability)',
            r'(will|kya)\s*(.*?)\s*(come|appear|aayega|ayega)',
            r'chances of\s*(.*)',
            r'probability of\s*(.*)',
            r'(.*?)\s*important hai\s*kya',
        ]

        for pattern in probability_patterns:
            match = re.search(pattern, message_lower)
            if match:
                # Extract topic from match
                groups = match.groups()
                topic = ' '.join(g for g in groups if g and g not in ['aayega', 'ayega', 'aega', 'appear', 'come', 'chances', 'probability', 'will', 'kya', 'important', 'hai'])
                topic = topic.strip()
                if topic:
                    return 'check_probability', {'topic': topic}

        # Intent: Chapter important topics
        chapter_patterns = [
            r'(.*?)\s*(?:ke|ka|ki|se|chapter|from)\s*(?:important|probable|expected|most likely|kya padhu|padna chahiye|questions?|topics?)',
            r'(?:important|probable|expected)\s*(?:questions?|topics?)\s*(?:from|in|of|ke|se)\s*(.*)',
            r'(.*?)\s*(?:chapter|se)\s*(?:kya|what)\s*(?:padhu|read|prepare)',
        ]

        for pattern in chapter_patterns:
            match = re.search(pattern, message_lower)
            if match:
                chapter = match.group(1).strip()
                if chapter and len(chapter) > 2:
                    return 'chapter_topics', {'chapter': chapter}

        # Intent: Question types
        qtype_patterns = [
            r'(.*?)\s*(?:type|format|mcq|short|long|kis tarah)',
            r'(?:how|kaise|kis)\s*(?:type|tarah)\s*(.*)',
        ]

        for pattern in qtype_patterns:
            match = re.search(pattern, message_lower)
            if match:
                topic = match.group(1).strip()
                if topic:
                    return 'question_types', {'topic': topic}

        # Default: Try to extract any topic or chapter
        return 'unknown', {'message': message}

    def _fuzzy_match_chapter(self, query: str) -> Optional[str]:
        """Fuzzy match chapter name"""
        query_lower = query.lower()

        # Direct match with aliases
        for chapter, aliases in self.chapter_aliases.items():
            if any(alias in query_lower for alias in aliases) or chapter in query_lower:
                return chapter

        # Try partial match
        for chapter in self.chapter_aliases.keys():
            if chapter in query_lower or query_lower in chapter:
                return chapter

        return query  # Return original if no match

    def _fuzzy_match_topic(self, query: str) -> str:
        """Fuzzy match topic name"""
        query_lower = query.lower()

        # Check aliases
        for topic, aliases in self.topic_aliases.items():
            if any(alias in query_lower for alias in aliases) or topic in query_lower:
                return topic

        return query

    def chat(self, message: str) -> str:
        """Process user message and return response"""
        intent, params = self._detect_intent(message)

        if intent == 'check_probability':
            return self._handle_probability_check(params['topic'])

        elif intent == 'chapter_topics':
            chapter = self._fuzzy_match_chapter(params['chapter'])
            return self._handle_chapter_topics(chapter)

        elif intent == 'overall_topics':
            return self._handle_overall_topics()

        elif intent == 'question_types':
            topic = self._fuzzy_match_topic(params['topic'])
            return self._handle_question_types(topic)

        else:
            return self._handle_unknown(message)

    def _handle_probability_check(self, topic: str) -> str:
        """Handle topic probability check"""
        topic = self._fuzzy_match_topic(topic)
        result = self.service.check_topic_probability(topic)

        if not result:
            return f"'{topic}' topic nahi mila. Try variations like 'Ohms Law', 'Electric Field', 'Lens', etc."

        prob = result['probability_score'] * 100

        if prob >= 70:
            confidence_text = "Haan, definitely important hai!"
        elif prob >= 50:
            confidence_text = "Haan, likely hai."
        else:
            confidence_text = "Kam chances hain, but prepare kar lo."

        response = f"""**{result['topic']}** - {prob:.0f}% probability

{confidence_text}

**Stats:**
- Frequency: {result['frequency']}
- Last appeared: {result['last_appeared']}
- Usually: {result['most_common_question_type']} type mein aata hai
- {result['reasoning']}

**Sample Questions:**
"""
        # Add sample questions
        questions_by_type = result.get('questions_by_type', {})
        q_count = 0
        for qtype, questions in questions_by_type.items():
            if q_count >= 3:
                break
            for q in questions[:1]:  # 1 question per type
                q_text = q['question'][:120] + "..." if len(q['question']) > 120 else q['question']
                response += f"- [{qtype}, {q['marks']}m] {q_text}\n"
                q_count += 1

        return response

    def _handle_chapter_topics(self, chapter: str) -> str:
        """Handle chapter important topics query"""
        results = self.service.get_chapter_topics(chapter, top_n=5)

        if not results:
            return f"'{chapter}' chapter ke topics nahi mile. Try: 'Electric Charges', 'Ray Optics', 'Current Electricity', etc."

        response = f"**{chapter.title()}** ke important topics:\n\n"

        for r in results:
            prob = r['probability_score'] * 100
            emoji = "🔥" if prob >= 70 else "✅" if prob >= 50 else "📌"
            response += f"{emoji} **{r['topic']}** - {prob:.0f}%\n"
            response += f"   {r['frequency']} | {r['most_common_question_type']}\n"

            # Add sample questions
            questions_by_type = r.get('questions_by_type', {})
            if questions_by_type:
                most_common = r['most_common_question_type']
                sample_qs = questions_by_type.get(most_common, [])
                if sample_qs:
                    q = sample_qs[0]
                    q_text = q['question'][:100] + "..." if len(q['question']) > 100 else q['question']
                    response += f"   📝 \"{q_text}\" [{q['year']}, {q['marks']}m]\n"
            response += "\n"

        response += f"💡 Tip: Top 2-3 topics pe zyada focus karo!"
        return response

    def _handle_overall_topics(self) -> str:
        """Handle overall top predictions query"""
        results = self.service.get_top_predictions(top_n=10)

        response = "**Physics ke sabse important topics:**\n\n"

        for r in results:
            prob = r['probability_score'] * 100
            emoji = "🔥" if prob >= 70 else "✅" if prob >= 50 else "📌"
            response += f"{r['rank']}. {emoji} **{r['topic']}** - {prob:.0f}%\n"

        response += "\n💡 Ye topics exam mein definitely aayenge. Inhe achhe se prepare karo!"
        return response

    def _handle_question_types(self, topic: str) -> str:
        """Handle question types query"""
        topic = self._fuzzy_match_topic(topic)
        result = self.service.check_topic_probability(topic)

        if not result:
            return f"'{topic}' topic nahi mila."

        response = f"**{result['topic']}** question types:\n\n"

        for qtype, count in result['question_type_distribution'].items():
            response += f"- {qtype}: {count} times\n"

        response += f"\n**Most common:** {result['most_common_question_type']}"

        if result.get('avg_marks_per_type'):
            response += f"\n\n**Average marks:**\n"
            for qtype, marks in result['avg_marks_per_type'].items():
                response += f"- {qtype}: {marks} marks\n"

        return response

    def _handle_unknown(self, message: str) -> str:
        """Handle unknown intent"""
        return """Main samajh nahi paya. Try these:

1. **Chapter topics:** "Electric Charges ke important questions"
2. **Topic probability:** "Ohm's Law aayega kya?"
3. **Overall important:** "Sabse important topics batao"
4. **Question types:** "Lens kis type mein aata hai?"
"""


def run_demo():
    """Run interactive demo"""
    print("=" * 60)
    print("Physics Question Predictor - Demo Agent")
    print("=" * 60)
    print("\nNo API key required! Try these queries:")
    print("  - 'Electric Charges ke important questions'")
    print("  - 'Ohm's Law aayega kya?'")
    print("  - 'Sabse important topics batao'")
    print("  - 'Ray Optics se kya padhu?'")
    print("\nType 'quit' to exit\n")

    agent = DemoAgent()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            response = agent.chat(user_input)
            print(f"\nAssistant:\n{response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    run_demo()
