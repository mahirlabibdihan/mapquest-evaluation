from BenchmarkDataset import BenchmarkDataset
from transformers import Trainer, TrainingArguments
from LLM import LLM
import requests


def extract(s):
    for char in s:
        if char.isdigit():
            return char
    return None  # Return None if no numeric character is found


class Evaluator:
    def __init__(self, model: LLM, dataset: BenchmarkDataset):
        self.model = model
        self.dataset = dataset
        self.results = []

    def evaluate(self):
        self.model.load_model()
        data = self.dataset.load_data()

        for i in range(0, len(data)):
            item = data[i]
            print("Running", i + 1, "/", len(data), ":", item["id"])
            if item["context"] == "":
                self.results.append(
                    {
                        "prompt": "",
                        "id": item["id"],
                    }
                )
                continue

            prompt = (
                "Context: "
                + item["context"]
                + "Question: "
                + item["question"]
                + "Choose the answer from the following options (1/2/3/4). And give explanation in bracket. So, the output format will be \"Option_Number (Explanation). If there is no answer in the options, then return 0 first and explain the reason. Remember you need to answer the question only from the context, not using any of your own knowledge. If the question can't be answered from the context notify it. Also return 0 if the correct answer is not present in the options.)"
            )
            for i in range(len(item["answer"]["options"])):
                if(item["answer"]["options"][i] == ""):
                    break
                prompt = (
                    prompt
                    + "Option"
                    + str(i + 1)
                    + ": "
                    + item["answer"]["options"][i]
                    + ", "
                )

            print("Prompt is created. Now passing to the model.")
            response = self.model.generate(prompt)
            print(response, extract(response))
            try:
                self.results.append(
                    {
                        "id": item["id"],
                        "prompt": prompt,
                        "response": response,
                        "ground_truth": item["answer"]["correct"] + 1,
                        # "data": item,
                    }
                )
                # print(
                #     {
                #         # "prompt": prompt,
                #         "response": int(response.split()[0].strip(":")[-1]),
                #         "ground_truth": item["answer"]["correct"] + 1,
                #         "verdict": int(response.split()[0].strip(":")[-1])
                #         == item["answer"]["correct"] + 1,
                #     }
                # )
            except ValueError:
                print("Error: The response could not be converted to an integer.")
            # break

    def compute_metrics(self):
        correct_answers = 0
        total_questions = len(self.results)
        invalid_questions = 0
        invalid_answers = 0

        list = []

        for result in self.results:
            # print(result)
            if result["prompt"] == "":
                invalid_questions += 1
                # print(result)
                list.append(
                    {
                        "query_id": result["id"],
                        "model_id": self.model.id,
                        "answer": "",
                        "verdict": "invalid",
                    }
                )
            else:
                try:
                    option = extract(result["response"])
                    # response = int(result["response"].split()[0].strip(":.")[-1])
                    response = int(option)
                    if result["ground_truth"] == 0:
                        invalid_questions += 1
                        list.append(
                            {
                                "query_id": result["id"],
                                "model_id": self.model.id,
                                "answer": response,
                                "verdict": "invalid",
                            }
                        )

                    elif response == result["ground_truth"]:
                        correct_answers += 1
                        list.append(
                            {
                                "query_id": result["id"],
                                "model_id": self.model.id,
                                "answer": response,
                                "verdict": "right",
                            }
                        )
                    elif response == 0:
                        invalid_answers += 1
                        list.append(
                            {
                                "query_id": result["id"],
                                "model_id": self.model.id,
                                "answer": result["response"],
                                "verdict": "invalid",
                            }
                        )
                    else:
                        list.append(
                            {
                                "query_id": result["id"],
                                "model_id": self.model.id,
                                "answer": response,
                                "verdict": "wrong",
                            }
                        )

                except Exception:
                    # print("Error: The response could not be converted to an integer.")
                    invalid_answers += 1
                    list.append(
                        {
                            "query_id": result["id"],
                            "model_id": self.model.id,
                            "answer": result["response"],
                            "verdict": "invalid",
                        }
                    )
        # print(list)
        response = requests.post(
            "https://mapquest-app.onrender.com/api/evaluation/", json=list
        )
        # print(response)
        accuracy = correct_answers * 100 / (total_questions - invalid_questions)
        accuracy = "{:.2f}".format(accuracy)

        # Open the file in write mode ('w')
        print(f"Accuracy: {accuracy}%\n")
        print(f"{invalid_questions} invalid questions\n")
        print(f"{invalid_answers} invalid responses\n")

    def print_results(self):
        # for result in self.results:
        # print(f"Prompt: {result['prompt']}")
        # print(f"Response: {result['response']}")
        # print(f"Ground Truth: {result['ground_truth']}\n")
        pass
