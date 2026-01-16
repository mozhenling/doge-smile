from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from utils.output.parser import json_tag_parser, detect_pydantic_parser

class BaseLLM():
    """Initialize a large language model and set some basic variables"""
    def __init__(self, config,
                 dir_system=None,
                 dir_human=None,
                 dir_ai=None,
                 train_examples=None, adapt_examples=None):

        # use to differentiate LLM and non-LLM algorithms
        self.is_llm = True
        self.train_required = False

        self.json_tag_parser = json_tag_parser
        self.detect_pydantic_parser = detect_pydantic_parser
        
        self.dir_system = dir_system
        self.dir_human = dir_human
        self.dir_ai = dir_ai

        self.train_examples = train_examples  if adapt_examples is None else train_examples + adapt_examples
        self.config = config

        self.local_llms = ["llama3.2", "llama3.2:1b", "llama3-groq-tool-use",
                           "qwen3:8b", "qwen2.5:14b","qwen2.5:7b", "qwen2.5:3b",
                           "smollm2", "phi4-mini",
                           "granite3.3:8b", "granite3.3:2b",
                           "hermes3:8b", "mistral:7b",
                           "cogito"]
        """
                   "llama3.2": https://ollama.com/library/llama3.2 (3b)
                   "llama3-groq-tool-use": 8b, https://ollama.com/library/llama3-groq-tool-use
                   "smollm2": https://ollama.com/library/smollm2:1.7b (1.7b)
                   "phi4-mini": 3.8b, https://ollama.com/library/phi4-mini
                    See models that support tool calling: https://ollama.com/search?c=tools
        """

        # -- initialize LLM
        if self.config["model_provider"] is not None and self.config["api_key"] is not None:

            self.llm = init_chat_model(model_provider=self.config["model_provider"],
                                       api_key=self.config["api_key"], **self.config["model_params"])
        elif self.config["model_params"]["model"] in self.local_llms:

            self.llm = ChatOllama(**self.config["model_params"])
        else:
            raise NotImplementedError

        # -- read messages from files
        if self.dir_system is not None:
            with open(self.dir_system, "r", encoding="utf-8") as f:
                self.message_system_content = f.read()
                self.SystemMessage = SystemMessage(content=self.message_system_content)
        else:
            self.message_system_content = None
            self.SystemMessage = None
        if self.dir_human is not None:
            with open(self.dir_human, "r", encoding="utf-8") as f:
                self.message_human_content = f.read()
                self.HumanMessage = HumanMessage(content=self.message_human_content)
        else:
            self.message_human_content = None
            self.HumanMessage = None

        if self.dir_ai is not None:
            with open(self.dir_ai, "r", encoding="utf-8") as f:
                self.message_ai_content= f.read()
                self.AIMessage = AIMessage(content=self.message_ai_content)
        else:
            self.message_ai_content= None
            self.AIMessage = None
    
    def parse_response(self, r, parser=None, error_return=False):
        parser = self.config["parser"] if parser is None else parser
        # -- parse results
        if parser == "detect_pydantic_parser":
            parsed = self.detect_pydantic_parser.invoke(r)
            r_dict = parsed.model_dump(mode="json")
            label = r_dict["label"]
            reason = r_dict["reason"]
            return {"label":label, "reason":reason}

        elif parser in ["detect_json_parser"]:
            if len(r.content) > 0:
                r_list_dict = self.json_tag_parser(r, error_return)
                try:
                    if len(r_list_dict)==1:
                        label = r_list_dict[0]["label"]
                        reason = r_list_dict[0]["reason"]
                    else:
                        label = [d["label"] for d in r_list_dict]
                        reason = [d["reason"] for d in r_list_dict]
                except:
                    label = None
                    reason = "Retry needed! Unable to parse AIMessage using detect_json_parser with the content:\n " + r.content + "\n"
                    # print(reason)
            else:
                label = None
                reason = "Retry needed! Unable to parse AIMessage using detect_json_parser with the content:\n" + r.content + "\n"
                # print(reason)
            return {"label":label, "reason":reason}

        elif parser == "raw_pydantic_parser":
            # Often too strict for small models. Larger LLMs are recommended.
            parsed = self.detect_pydantic_parser.invoke(r)
            r_dict = parsed.model_dump(mode="json")
            return r_dict

        elif parser in ["json_tag_parser", "raw_json_parser"]:
            r_dict = self.json_tag_parser(r, error_return)
            return r_dict
        else:
            raise NotImplementedError("Paser not found!")


