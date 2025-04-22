import asyncio
import random
from typing import Any, Sequence

import pydantic
from pydantic import TypeAdapter

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.prompt.base import ChatCompletionPrompt
from evals.record import Event, RecorderBase, record_error, record_metrics  # type: ignore

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import UnexpectedModelBehavior


class Sample(pydantic.BaseModel):
    input: str
    criteria_questions: list[str]


# Define Pydantic model for the expected structure within Event.data for metrics
class MetricsData(pydantic.BaseModel):
    score: float
    all_correct: bool
    num_yes: int


MetricsDataList = TypeAdapter(list[MetricsData])


class StructuredCriteriaEval(evals.Eval):
    """
    Eval that uses pydantic-ai to evaluate a completion against multiple
    structured Y/N criteria questions.
    """

    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        evaluator_model_id: str,
        *args: Any,
        evaluator_completion_fn_options: dict[str, Any] | None = None,
        evaluator_retries: int = 10,
        max_tokens: int = 500,
        **kwargs: Any,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        self.samples_jsonl = samples_jsonl
        self.evaluator_retries = evaluator_retries
        self.evaluator_model_id = evaluator_model_id
        self.max_tokens = max_tokens

        assert len(completion_fns) == 1, "StructuredCriteriaEval requires exactly one CompletionFn."

        self.evaluator_completion_fn = self.completion_fn
        self.evaluator_completion_fn_options = evaluator_completion_fn_options or {}
        self.evaluator_agent = PydanticAgent[None, bool](
            self.evaluator_model_id,  # Needs the actual model identifier string
            result_type=bool,
            system_prompt="You are an evaluator assessing an AI response based on a specific criterion.",
            retries=self.evaluator_retries,
        )

    async def get_eval_result(self, input: str, completion: str, criterion_question: str) -> bool | None:
        evaluator_prompt = self._build_evaluator_prompt(original_input=input, completion=completion, criterion_question=criterion_question)
        try:
            return (await self.evaluator_agent.run(evaluator_prompt)).output
        except UnexpectedModelBehavior as e:
            record_error(f"Pydantic-AI evaluation failed for question '{criterion_question}': {e}")

    async def get_all_eval_results(self, input: str, completion: str, criterion_questions: list[str]) -> list[bool | None]:
        return await asyncio.gather(*[self.get_eval_result(input=input, completion=completion, criterion_question=question) for question in criterion_questions])

    def eval_sample(self, sample: dict[str, Any], rng: random.Random) -> None:
        """Evaluates a single sample."""
        del rng
        _sample = Sample.model_validate(sample)

        # 1. Get completion from the model under test
        #    (Using internal _get_completion for potential reuse/caching if needed)
        test_completion_result = self.completion_fn(sample["input"])
        completion_text = test_completion_result.get_completions()[0]  # Assuming one completion

        results = asyncio.run(self.get_all_eval_results(input=_sample.input, completion=completion_text, criterion_questions=_sample.criteria_questions))
        criteria_results = dict(zip(_sample.criteria_questions, results))

        # 3. Calculate score and record results for this sample
        num_yes = sum(bool(r) for r in criteria_results.values())
        num_evaluated = len(criteria_results)
        score = (num_yes / num_evaluated) if num_evaluated > 0 else 0.0
        all_correct = num_yes == num_evaluated

        record_metrics(
            all_correct=all_correct,
            score=score,
            num_yes=num_yes,
        )

    def _build_evaluator_prompt(self, original_input: str | ChatCompletionPrompt, completion: str, criterion_question: str) -> str:
        """Builds the prompt for the evaluator model."""
        input_str = str(original_input)
        prompt = f"""
You are assessing a submitted answer based on a specific criterion question.

[BEGIN DATA]
***
[Task Prompt]:
{input_str}
***
[Submitted Answer]:
{completion}
***
[Criterion Question]:
{criterion_question}
***
[END DATA]

Carefully consider the Submitted Answer in relation to the Task Prompt and the Criterion Question.
Answer the Criterion Question with either "True" or "False".
"""
        return prompt.strip()

    def run(self, recorder: RecorderBase) -> dict[str, Any]:
        """Runs the eval across all samples."""
        samples: list[dict[str, Any]] = self.get_samples()
        self.eval_all_samples(recorder, samples)  # type: ignore

        # Aggregate final metrics
        events: Sequence[Event] = recorder.get_events("metrics")

        ev_data: list[Any] = [ev.data for ev in events]  # type: ignore
        data: list[MetricsData] = MetricsDataList.validate_python(ev_data)

        num_samples_recorded = len(data)  # Total events recorded (might include errors)
        average_score = sum(d.score for d in data) / len(data)
        all_correct = sum(d.all_correct for d in data) / len(data)

        return {
            "average_score": average_score,
            "all_correct": all_correct,
            "num_samples_recorded": num_samples_recorded,
        }
