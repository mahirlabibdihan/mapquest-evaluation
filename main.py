from BenchmarkDataset import BenchmarkDataset
from Evaluator import Evaluator
from Llama3 import Llama3
from Mistral import Mistral
from Phi3 import Phi3
from Qwen2 import Qwen2
from Mixtral import Mixtral
import torch
import argparse

# def main():
#     # Load and preprocess dataset
#     dataset = BenchmarkDataset(filepath="dataset.json")
#     dataset.preprocess_data()

#     # Initialize models
#     models = [Qwen2(), Phi3(), Mistral(), Llama3()]

#     # Evaluate each model
#     for model in models:
#         print(f"Evaluating model: {model.__class__.__name__}")
#         evaluator = Evaluator(model=model, dataset=dataset)
#         evaluator.evaluate()
#         evaluator.print_results()
#         print(model.__class__.__name__, "metrics")
#         evaluator.compute_metrics()
#         torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument("model", type=str, help="Name of the model to evaluate")
    args = parser.parse_args()

    # Load and preprocess dataset
    dataset = BenchmarkDataset(filepath="dataset.json")
    dataset.preprocess_data()

    # Initialize models
    if args.model == "Phi3":
        model = Phi3()
    elif args.model == "Mistral":
        model = Mistral()
    elif args.model == "Llama3":
        model = Llama3()
    elif args.model == "Qwen2":
        model = Qwen2()
    elif args.model == "Mixtral":
        model = Mixtral()
    else:
        raise ValueError(f"Model {args.model} not recognized.")

    # Evaluate each model
    print(f"Evaluating model: {model.__class__.__name__}")
    evaluator = Evaluator(model=model, dataset=dataset)
    evaluator.evaluate()
    evaluator.print_results()
    print(model.__class__.__name__, "metrics")
    evaluator.compute_metrics()


if __name__ == "__main__":
    main()
