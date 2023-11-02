class InvalidModelResponseError(Exception):
    """Raised when the model returns a response that cannot be evaluated"""

    def __init__(self, model: str, response: str):
        super().__init__(f"Model {model} returned an invalid response: {response}")
        self.response = response
