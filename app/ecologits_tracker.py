from ecologits import EcoLogits
from mistralai import Mistral
import os


class EcoMistralTracker:
    def __init__(self):
        EcoLogits.init(providers=["mistralai"])

        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def tracked_inference(self, prompt: str):
        """
        Makes a real tracked Mistral call and returns (text, impacts)
        """
        response = self.client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content, response.impacts
