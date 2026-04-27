import re
from langchain_core.messages import HumanMessage

GENERIC_GRAPH_PROGRAM = """
p(image | concept=math-graph) =
  p(axes-limits-and-labels | concept=math-graph)
  p(origin-location-and-grid-increments | concept=math-graph, axes-limits-and-labels)
  p(curve-type-and-qualitative-shape | concept=math-graph, origin-location-and-grid-increments)
  p(precise-coordinates-of-intercepts-and-extrema | concept=math-graph, curve-type-and-qualitative-shape)
  p(image | axes-limits-and-labels, origin-location-and-grid-increments, curve-type-and-qualitative-shape, precise-coordinates-of-intercepts-and-extrema)
"""

def parse_program(program):
    lines = [line.strip() for line in program.strip().split('\n') if '=' not in line and '|' in line]
    concepts = []
    for line in lines:
        match = re.search(r'p\((.*?) \|', line)
        if match:
            concepts.append(match.group(1).replace('-', ' '))
    return concepts

def run_dsg_loop(img, original_question, llm, encode_fn):
    """Runs the multi-turn grounding loop using the provided LangChain LLM."""
    concepts = parse_program(GENERIC_GRAPH_PROGRAM)
    context = ""
    
    for concept in concepts:
        prompt = f"Imagine that the image represents math-graph and the context is {context if context else 'nothing yet'}, what is the {concept}? Answer with one word or phrase only."
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_fn(img)}"}}
        ])
        # Use simple invoke for grounding steps
        res = llm.invoke([msg], n=1)
        ans = res.content.strip()
        context += f"{concept} is {ans}, "
    
    final_prompt = f"Using the following grounded context: {context}\nSolve this question: {original_question}"
    return final_prompt
