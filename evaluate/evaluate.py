"""Evaluation tool for the Airbus AI Hackathon 2024"""


import sys
import getopt
import json
import torch
import numpy as np

from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


class HackathonMetrics():
    """Management of the metrics computation"""
    def __init__(
        self,
        reference_dataset_path,
        generated_dataset_path,
        model_path='../all-MiniLM-L6-v2/'
    ):
        super(HackathonMetrics, self).__init__()
        # Load semantic model.
        self.semantic_model = SentenceTransformer(model_path)
        if torch.cuda.is_available():
            self.semantic_model = self.semantic_model.cuda()
        # Load dataset
        self.reference_dataset = self.load_dataset(reference_dataset_path)
        self.generated_dataset = self.load_dataset(generated_dataset_path)
        self.reference_uid = set(self.reference_dataset.keys())
        self.generated_uid = set(self.generated_dataset.keys())
        
        if len(self.generated_uid) < len(self.reference_dataset):
            print(
                "Warning: Reference dataset contain "
                "more example than Generated dataset"
            )
        self.scores = None
        self.rouge_scores = None
        self.summary_similarity_scores = None
        self.original_similarity_scores = None

    def load_dataset(self, dataset_path) -> dict[str, dict[str, str]]:
        with open(dataset_path, 'r', encoding="utf8") as json_file:
            dataset = json.load(json_file)
        if dataset is None:
            print(f'Error when loading dataset {dataset_path}')
            sys.exit(1)
        print(f"Loaded {len(dataset)} examples from {dataset_path}")
        return dataset

    def compute_rouge(self) -> dict[str, dict[str, str]]:
        scores = []
        metrics = ['rouge1', 'rouge2', 'rougeL']
        scorer = rouge_scorer.RougeScorer(
            metrics,
            use_stemmer=True
        )
        for text_uid in self.generated_dataset.keys():
            scores.append(
                scorer.score(
                    self.reference_dataset.get(text_uid).get(
                        "reference_summary"
                    ),
                    self.generated_dataset.get(text_uid).get(
                        "generated_summary"
                    )
                )
            )
        final_scores = {
            metric: {
                "precision": str(np.mean(
                    [score.get(metric).precision for score in scores]
                )),
                "recall": str(np.mean(
                    [score.get(metric).recall for score in scores]
                )),
                "fmeasure": str(np.mean(
                    [score.get(metric).fmeasure for score in scores]
                )),
                }
            for metric in metrics
        }
        return final_scores

    def compute_similarity_scores(self) -> dict[str, dict[str, str]]:
        scores_with_reference = []
        scores_with_original_text = []
        for text_uid in self.generated_dataset.keys():
            reference_embeddings = self.semantic_model.encode(
                self.reference_dataset.get(text_uid).get("reference_summary"),
                convert_to_tensor=True
            )
            original_text_embeddings = self.semantic_model.encode(
                self.reference_dataset.get(text_uid).get("original_text"),
                convert_to_tensor=True
            )
            generated_embeddings = self.semantic_model.encode(
                self.generated_dataset.get(text_uid).get("generated_summary"),
                convert_to_tensor=True
            )
            scores_with_reference.append(
                util.cos_sim(
                    reference_embeddings,
                    generated_embeddings).cpu()
            )
            scores_with_original_text.append(
                util.cos_sim(
                    original_text_embeddings,
                    generated_embeddings).cpu()
            )
        final_scores = {
            "similarity_with_reference_summary": {
                "mean":   str(np.mean(scores_with_reference)),
                "median": str(np.median(scores_with_reference)),
                "std":    str(np.std(scores_with_reference))
            },
            "similarity_with_original_text": {
                "mean":   str(np.mean(scores_with_original_text)),
                "median": str(np.median(scores_with_original_text)),
                "std":    str(np.std(scores_with_original_text))
            }
        }
        return final_scores

    def evaluate(self) -> None:
        if self.scores is not None:
            return
        # Compute similarity scores between original text and generated summary
        self.scores = {
            "Rouge": self.compute_rouge(),
            "Similarity": self.compute_similarity_scores()
        }

    def print_scores(self) -> None:
        print(json.dumps(self.scores, indent=4, sort_keys=True))

    def save_report(self) -> None:
        with open("../evaluations reports/evaluation_report.json", "w", encoding='UTF-8') as f_out:
            f_out.write(json.dumps(self.scores, indent=4, sort_keys=True))


def print_usage() -> None:
    print(
        "python evaluate.py "
        "-r <path/reference_dataset.json> "
        "-g <path/generated_dataset.json>"
    )


def main(argv) -> None:
    generated_dataset_path = ''
    reference_dataset_path = ''
    opts, _ = getopt.getopt(
        argv,
        "hr:g:",
        ["help", "reference=", "generated="]
    )
    if len(opts) != 2:
        print_usage()
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', "help"):
            print_usage()
            sys.exit()
        elif opt in ("-r", "--reference"):
            reference_dataset_path = arg
        elif opt in ("-g", "--generated"):
            generated_dataset_path = arg
    hackathon_metrics = HackathonMetrics(
        reference_dataset_path,
        generated_dataset_path
    )
    hackathon_metrics.evaluate()
    hackathon_metrics.print_scores()
    hackathon_metrics.save_report()


if __name__ == "__main__":
    main(sys.argv[1:])
