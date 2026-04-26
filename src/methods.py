import base64
from io import BytesIO
from langchain_core.messages import HumanMessage


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
    def __call__(self, question_id, question, answer_type, subject, img, llm):
        pass


class SushilMethod:
    def __call__(self, question_id, question, answer_type, subject, img, llm):
        pass


class AkibMethod:
    def __call__(self, question_id, question, answer_type, subject, img, llm):
        pass


class VasudevMethod:
    def __call__(self, question_id, question, answer_type, subject, img, llm):
        pass