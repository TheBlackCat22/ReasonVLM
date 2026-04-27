import os
import base64
import pandas as pd
from io import BytesIO
from langchain_core.messages import HumanMessage

from viser_utils import apply_viser_scaffolding, get_viser_prompt
from scaffold_utils import apply_scaffold_coordinates, get_scaffold_prompt
from dsg_utils_vasu import run_dsg_loop


def encode_image(img):
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


GUIDE = """
## Answer Instruction 
Please provide an answer to the question outlined above. Your response should adhere to the following JSON format, which includes two keys: 'solution' and 'short answer'. The 'solution' key can contain detailed steps needed to solve the question, and the 'short answer' key should provide a concise response. {INST}

Example of expected JSON response format:
{{
    "solution": "[Detailed step-by-step explanation]",
    "short answer": "[Concise Answer]"
}}
"""


class BaselineMethod:
    def __call__(self, question_id, question, answer_type, subject, img, llm):
        # Determine instruction suffix
        if answer_type == 'multiple choice':
            inst = "Provide the corresponding choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'."
        elif answer_type == 'float':
            inst = "Format the answer as a three-digit floating-point number and provide it in the 'short answer' key."
        else:
            inst = "Float numbers in the answer should be formatted as three-digit floating-point numbers."
            
        # Construct multimodal message
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": f"## Question\n {question}\n" + GUIDE.format(INST=inst)
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"}
                }
            ]
        )
        return message


class SushilOracleMethod:
    def __init__(self):
        self.oracle_data = pd.read_json("src/sushiloracle_data.jsonl", lines=True)

    def __call__(self, question_id, question, answer_type, subject, img, llm):
        if answer_type == 'multiple choice':
            inst = "Provide the corresponding choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'."
        elif answer_type == 'float':
            inst = "Format the answer as a three-digit floating-point number and provide it in the 'short answer' key."
        else:
            inst = "Float numbers in the answer should be formatted as three-digit floating-point numbers."

        plot_code = self.oracle_data.loc[self.oracle_data['question_id'] == int(question_id), 'plot_function'].values[0]
        plot_code_prompt = f"\n## Plot Code \nHere is the python code used to generate the image. \n```python\n{plot_code}\n```"

        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": f"## Question\n {question}\n" + GUIDE.format(INST=inst) + plot_code_prompt
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"}
                }
            ]
        )
        return message


class SushilMethod:
    def __call__(self, question_id, question, answer_type, subject, img, llm):
        if answer_type == 'multiple choice':
            inst = "Provide the corresponding choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'."
        elif answer_type == 'float':
            inst = "Format the answer as a three-digit floating-point number and provide it in the 'short answer' key."
        else:
            inst = "Float numbers in the answer should be formatted as three-digit floating-point numbers."

        plot_code = self.generate_plot_code(question_id, img)
        if plot_code != "":
            plot_code_prompt = f"\n## Plot Code \nHere is the python code used to generate the image. \n```python\n{plot_code}\n```"
        else:
            plot_code_prompt = ""

        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": f"## Question\n {question}\n" + GUIDE.format(INST=inst) + plot_code_prompt
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"}
                }
            ]
        )
        return message    

    def generate_plot_code(self, question_id, img):
        message = HumanMessage(
            content =[
                {
                    "type": "text",
                    "text": "Analyze this image and write a complete Python script using the matplotlib library to recreate it visually. Break the image down into basic shapes, lines, and colors. Output ONLY valid, executable Python code. Do not include any markdown formatting, explanations, or conversational text."
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"}
                }
            ]
        )

        try:
            response = llm.generate([[message]], n=1)
            generation = response.generations[0] 
        except Exception as e:
            print(f"\nPlot Generation failed for ID {question_id}: {e}")
            generation = ""
        
        return generation


class AkibMethod:
    def __call__(self, question_id, question, answer_type, subject, img, llm):
        pass


class VasudevMethod:
    """
    Orchestrates DSG, VISER, and SCAFFOLD methods.
    Mode toggled via environment variable 'VASUDEV_MODE' (options: 'dsg', 'viser', 'scaffold').
    """
    def __init__(self, sub_method=None):
        # Default to dsg for now
        self.sub_method = sub_method or os.getenv("VASUDEV_MODE", "dsg")

    def __call__(self, question_id, question, answer_type, subject, img, llm):
        # 1. Determine instruction suffix (standard DynaMath format)
        if answer_type == 'multiple choice':
            inst = "Provide the corresponding choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'."
        elif answer_type == 'float':
            inst = "Format the answer as a three-digit floating-point number and provide it in the 'short answer' key."
        else:
            inst = "Float numbers in the answer should be formatted as three-digit floating-point numbers."

        processed_img = img.copy()
        processed_question = question

        # 2. Apply the selected logic
        if self.sub_method == "viser":
            processed_img = apply_viser_scaffolding(processed_img)
            processed_question = get_viser_prompt(question)
        
        elif self.sub_method == "scaffold":
            processed_img = apply_scaffold_coordinates(processed_img)
            processed_question = get_scaffold_prompt(question)
            
        elif self.sub_method == "dsg":
            # Multi-turn DSG grounding loop
            processed_question = run_dsg_loop(img, question, llm, encode_image)

        # 3. Build and return the multimodal message
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": f"## Question\n {processed_question}\n" + GUIDE.format(INST=inst)
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{encode_image(processed_img)}"}
                }
            ]
        )
        return message