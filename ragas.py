from datasets import load_dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextRecall,
    SemanticSimilarity,
)
from ragas.utils import evaluate

dataset = load_dataset(
    "explodinggradients/amnesty_qa",
    "english_v3",
    trust_remote_code=True
)
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

metrics = [
    LLMContextRecall(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=evaluator_embeddings)
]

eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])
results = evaluate(dataset=eval_dataset, metrics=metrics)
