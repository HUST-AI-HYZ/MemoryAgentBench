import argparse
import json
import yaml
import requests
from datasets import load_dataset


class MemoryClient:
    def __init__(self, base_url: str = "http://localhost:8000", memory_system: str = "long_context_gpt-4o-mini"):
        self.base_url = base_url.rstrip("/")
        self.memory_system = memory_system

    def initialize(self, user_id: str):
        resp = requests.post(f"{self.base_url}/memory/initialize", json={"user_id": user_id, "memory_system": self.memory_system})
        resp.raise_for_status()
        return resp.json()

    def add(self, user_id: str, chunk: str):
        resp = requests.post(f"{self.base_url}/memory/add", json={"user_id": user_id, "chunk": chunk, "memory_system": self.memory_system})
        resp.raise_for_status()
        return resp.json()

    def wrap_user_prompt(self, user_id: str, question: str):
        resp = requests.post(f"{self.base_url}/memory/wrap_user_prompt", json={"user_id": user_id, "question": question, "memory_system": self.memory_system})
        resp.raise_for_status()
        return resp.json()

    def act(self, user_id: str, prompt: str):
        resp = requests.post(f"{self.base_url}/agent/act", json={"user_id": user_id, "prompt": prompt, "memory_system": self.memory_system})
        resp.raise_for_status()
        return resp.json()

def main():
    parser = argparse.ArgumentParser(description="Send chunks/questions to a memory agent server.")
    parser.add_argument("--memory_system", default="long_context_gpt-4o-mini", type=str)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    client = MemoryClient(args.url, args.memory_system)
    dataset = load_dataset("ai-hyz/MemoryAgentBench")

    for category in ['Accurate_Retrieval', 'Test_Time_Learning', 'Long_Range_Understanding', 'Conflict_Resolution']:
        category_dataset = dataset[category]
        results = []
        for item in category_dataset:

            # TODO: randomly generate an instance_id for each item
            instance_id = item['instance_id']

            # TODO: Now item has the key `context`
            # TODO: Can we split them and wrap into conversation formats to get the key `chunks`?
            chunks = item['chunks']
            questions = item['questions']

            # chunk should be [{'role': "user", 'content': chunk}, {'role': "assistant", 'content': "Sure, I'll help you with that."}]

            client.initialize(instance_id)
            for chunk in chunks:
                client.add(instance_id, chunk)
            
            for question in questions:
                wrapped = client.wrap_user_prompt(instance_id, question)
                answered = client.act(instance_id, wrapped.get("prompt", question))
                results.append({"instance_id": instance_id, "predicted_answer": answered.get("answer", answered)})

        with open(f"results_{category}.jsonl", "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main()
