import textwrap

system_prompt = textwrap.dedent(
    """You are Lepton, a highly analytical AI assistant named after the family of fundamental 
particles that includes neutrinos. Your purpose is to support users in research, analysis, and engineering, with accuracy 
and efficiency.

Core traits:
- Rational, concise, and scientifically grounded
- Communicates with clarity, precision, and professionalism
- Adept in physics, machine learning, programming, and data workflows
- Responds in a way that optimizes for correctness, not verbosity
- Offers actionable insights and structured reasoning
- Respects uncertainty; does not fabricate knowledge

Behavior:
- Use concrete examples and direct explanations
- When generating code, adhere to clean, maintainable, production-quality practices
- If a question is ambiguous, request targeted clarification
- When appropriate, suggest logical next steps without overstepping
- Focus on data, reasoning, and relevance—avoid fluff or opinion unless requested

---
Persona: Lepton — small, subtle, and scientifically sharp.
"""
).strip()

self_critique_prompt_template = textwrap.dedent(
    f"""Critique your previous response and suggest improvements:
    Question: \n {{query}}
                                                
    Your previous response: \n {{response}}

    Your task is to critically analyze your response. Identify any potential errors, oversights, or areas where the reasoning could be strengthened. Then provide an improved response that addresses these issues."""
).strip()

alternative_generation_prompt_template = textwrap.dedent(
    f"""Consider your previous response to this question:

    Question: {{query}}

    Your previous response:
    
    {{response}}

    Generate alternative approaches or perspectives that you did not consider initially. Then synthesize these alternatives with your original thinking to provide a more comprehensive response."""
).strip()

confidence_assessment_prompt_template = textwrap.dedent(
    f"""Evaluate your previous response to this question:

    Question: {{query}}

    Your previous response:
    
    {{response}}

    For each major claim or conclusion in your response, assess your confidence level and identify areas of uncertainty. Focus your reflection on the low-confidence areas and provide additional analysis or revised reasoning where needed."""
).strip()

verification_prompt_template = textwrap.dedent(
    f"""Verify your previous response to this question:

    Question: {{query}}

    Your previous response:
    {{response}}

    Check whether your response satisfies these criteria: internal logical consistency, completeness in addressing all aspects of the question, and accuracy of any factual claims. Identify any failures and provide a corrected response."""
).strip()

adversarial_prompt_template = textwrap.dedent(
    f"""Challenge your previous response to this question:

    Question: {{query}}

    Your previous response:
    
    {{response}}

    Adopt a skeptical perspective and argue against your own conclusions. What counterarguments or alternative explanations exist? After considering these challenges, provide a refined response that addresses the strongest objections."""
).strip()
