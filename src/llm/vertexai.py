import vertexai
from vertexai.language_models import TextGenerationModel


class VertexAIClient:
    def __init__(self, project_id: str, location: str):
        vertexai.init(project=project_id, location=location)

    def generate(self, prompt: str) -> str:
        return self.model.generate(prompt)