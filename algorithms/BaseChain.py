

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from utils.log.time import timing
from algorithms.BaseLLM import BaseLLM

class BaseChain(BaseLLM):
    def __init__(self, config, dir_system, dir_human, dir_ai=None, train_examples=None, adapt_examples=None,format_instructions=None):
        super(BaseChain, self).__init__(config, dir_system, dir_human, dir_ai, train_examples, adapt_examples)

        self.format_instructions = format_instructions if format_instructions is not None else self.detect_pydantic_parser.get_format_instructions()

        ########################
        self.ParserMessage = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """For the final answer, wrap the final output in <json>...</json> tags strictly adhering to this schema:\n{format_instructions}""",
                ),
            ]
        ).partial(format_instructions=self.format_instructions)

        # - zero shot
        # For ChatPromptTemplate, it must have a message in the format e.g, ("human", "xxx") for effective invoking
        if train_examples is None:
            self.prompt_chat = ChatPromptTemplate(
                [self.SystemMessage,
                 self.ParserMessage,
                 ("human",self.message_human_content)])

        # - few shot
        else:
            example_prompt = ChatPromptTemplate(
                [("human",self.message_human_content),
                 ("ai", self.message_ai_content)])

            self.FewShotMessage = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=train_examples)

            self.prompt_chat = ChatPromptTemplate(
                [self.SystemMessage,
                 self.ParserMessage,
                 self.FewShotMessage,
                 ("human",self.message_human_content)])

        # -- chain runnables up
        self.model =  self.prompt_chat | self.llm

    # @timing
    def llm_batch(self, x, parser=None, error_return=False):
        # -- initial run internally
        # debug
        # templated_prompts = self.prompt_chat.invoke(x[0])

        ai_msgs = self.model.batch(x)

        if parser == "json_tag_parser" and error_return:
            # -- process initial run
            rsp_content_list = [self.parse_response(r, parser, error_return) for r in ai_msgs]
        else:
            rsp_content_list = [self.parse_response(r, parser) for r in ai_msgs]

        return rsp_content_list

    def predict(self, x):
        """For prediction of fault detection"""
        # ---------- Run the model to get predictions
        if not isinstance(x, (list, tuple)):
            x = [x]

        final_results = [{"label": None, "reason": None}] * len(x)
        retries = 0

        while retries < self.config["max_runs"] and any(r["label"] is None for r in final_results):
            incomplete_indices = [i for i, r in enumerate(final_results) if r["label"] is None]
            x_batch = [x[i] for i in incomplete_indices]

            batch_results = self.llm_batch(x_batch)
            for idx, result in zip(incomplete_indices, batch_results):
                if result["label"] is not None:
                    final_results[idx] = result
            retries += 1

        if any(r["label"] is None for r in final_results):
            raise TimeoutError("Run out of max_runs but still cannot get valid labels for all the signals")
        else:
            return [
                (r["label"], r["reason"]) if self.config["with_reason"] else r["label"]
                for r in final_results
            ]


