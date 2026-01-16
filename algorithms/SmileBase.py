
from algorithms.BaseChain import BaseChain
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from algorithms.Smile import Smile
from utils.output.seed import seed_hash

import json
import warnings
from utils.log.time import timing


# ----------- Main LLM Class -----------
class SmileBase(BaseChain):
    def __init__(self, config,
                 dir_system,
                 dir_human,
                 dir_ai,
                 examples=None, agent_part=True):

        super(SmileBase, self).__init__(config, dir_system, dir_human, dir_ai, examples)

        self.tool_call_records = {"tool_ai": [], "tool_default": []}
        self.toolkit_ai = {"smile":Smile(config, agent_part=agent_part)}
        self.toolkit_default = {}
        self.toolkit_all = {**self.toolkit_ai, **self.toolkit_default}
        self.tools = list(self.toolkit_all.values())

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.chain_with_tools = self.prompt_chat | self.llm_with_tools

    # For other usages of the llm, just use the self.llm, e.g., as follows
    def invoke(self, message):
        # for general response
        return self.llm.invoke(message)
    def batch(self, messages):
        # for general response
        return self.llm.batch(messages)
    # For fault detection, use the predict method
    def predict(self, x):
        #f for fault detection
        # ----------- Internal utility functions
        # @timing
        def _force_final_answer(msg, tpred):
            if tpred:
                return AIMessage(content=json.dumps({
                    "label": tpred[0][0],
                    "reason": f"Obtained from tool: {tpred[0][1]}"
                }))
            return AIMessage(content=json.dumps({
                "label": None,
                "reason": "No tool calls. AIMessage.content = " + msg.content
            }))

        @timing
        def _run_with_tool(xi, ai_msg, raw=False):
            messages = []
            stats_classifier_preds = []
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                for tool_call in getattr(ai_msg, "tool_calls", []):
                    tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
                    tool_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
                    args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})

                    if not tool_name:
                        warnings.warn("tool_name is empty.")
                        continue

                    tool_name = tool_name.lower()
                    if tool_name not in self.toolkit_all:
                        warnings.warn(f"Tool '{tool_name}' not in toolkit!")
                        continue

                    selected_tool = self.toolkit_all[tool_name]


                    tool_output = selected_tool.invoke(args)
                    if tool_name in ["auto_args_raw_fault_detect",
                                     "ai_args_raw_fault_detect",
                                     "auto_args_file_fault_detect",
                                     "ai_args_file_fault_detect"]:
                        stats_classifier_preds.append((tool_output, tool_name))
                        tool_content = json.dumps({
                                "tool_output": tool_output,
                                "tool_name": tool_name,
                                "tool_aim": "classify the machine signal into one of the two labels: 1 (faulty) or 0 (healthy)",
                                "tool_run": "successful!" if tool_output is not None else "unsuccessful!"
                            })
                    else:
                        if tool_name == "smile":
                            output_dict = json.loads(tool_output)
                            stats_classifier_preds.append((output_dict["tool_output"], output_dict["tool_name"]))
                        tool_content = tool_output
                    if raw:
                        tool_msg = tool_content
                    else:
                        tool_msg = ToolMessage(
                            tool_call_id=tool_id,
                            content=tool_content
                        )
                    messages.append(tool_msg)
            elif len(ai_msg.content)>0:
                # if ai meassge content is not empty, try to parse tool call information as a fallback
                rsp_dict_list = self.parse_response(ai_msg, parser="json_tag_parser")
                for rsp_dict in rsp_dict_list:
                    if "name" in rsp_dict:
                        tool_name = rsp_dict["name"].lower()
                    else:
                        tool_name = None
                    if "parameters" in rsp_dict:
                        parameters = rsp_dict["parameters"]
                    elif "args" in rsp_dict:
                        parameters = rsp_dict["args"]
                    else:
                        parameters=None

                    if tool_name in self.toolkit_all and parameters is not None:
                        selected_tool = self.toolkit_all[tool_name]
                        # implementation for the tool 'smile'
                        if tool_name == "smile" and "signal_path" in parameters:
                            tool_output = selected_tool.invoke(parameters["signal_path"])
                            output_dict = json.loads(tool_output)
                            stats_classifier_preds.append((output_dict["tool_output"], output_dict["tool_name"]))
                            tool_content = tool_output
                            tool_call_id = seed_hash(tool_content)
                            # reset tool call information of the AI message
                            ai_msg.tool_calls=[{"name":tool_name, "args":parameters, "id":tool_call_id, "type":"tool_call"}]
                            if raw:
                                tool_msg = tool_content
                            else:
                                tool_msg = ToolMessage(
                                    tool_call_id=seed_hash(tool_call_id),
                                    content=tool_content
                                )
                            messages.append(tool_msg)
            return messages, stats_classifier_preds

        @timing
        def _llm_batch(x):
            # start = time.perf_counter()
            # templated_prompts = self.prompt_chat.invoke(x[0])
            # Step 1: Run initial LLM batch
            ai_msgs_init = self.chain_with_tools.batch(inputs=x)
            tool_run_outputs = [_run_with_tool(xi, msg) for xi, msg in zip(x, ai_msgs_init)]
            # print(f"Step 1: {time.perf_counter() - start:.4f} seconds")

            # start = time.perf_counter()
            # Step 2: Determine which messages need tool use
            tool_msgs = [msg for msg, _ in tool_run_outputs]
            with_tool_indices = [i for i, msg in enumerate(tool_msgs) if msg]
            tool_stats_preds = [tool_with_pred for _, tool_with_pred in tool_run_outputs]
            # print(f"Step 2: {time.perf_counter() - start:.4f} seconds")

            # start = time.perf_counter()
            # Step 3: Second pass only for those with tool usage
            ai_msgs_end = ai_msgs_init.copy()
            if with_tool_indices:
                # Send new messages (Human + AI + Tool responses) back to the model
                human_message_content = ChatPromptTemplate.from_template(self.message_human_content)
                format_instructions = self.detect_pydantic_parser.get_format_instructions()
                egmsg = AIMessage(content='''```json
                         { "label": 0,
                          "reason": "The tool, smile, is used successfully, and the tool output is, 0, indicating a healthy state."}
                          ```''')
                messages_batch = [[
                    SystemMessage(content=f"""Respond with the tool feedback strictly adhering to this schema:
                                     {format_instructions}. And wrap your final output into Json, e.g. {egmsg}"""),
                    HumanMessage(content=human_message_content.format(**x[i])),
                    ai_msgs_init[i],
                    *tool_msgs[i]
                ] for i in with_tool_indices]
                print("Process tool feedback.")
                updated_msgs = self.llm_with_tools.batch(messages_batch)

                for idx, msg in zip(with_tool_indices, updated_msgs):
                    ai_msgs_end[idx] = msg
            # print(f"Step 3: {time.perf_counter() - start:.4f} seconds")

            # start = time.perf_counter()
            # Step 4: Final response parsing
            rsp_content_list = []
            for i, (msg, tpred) in enumerate(zip(ai_msgs_end, tool_stats_preds)):
                # Record tool calling info.
                tool_type = (
                    "tool_ai" if tpred and tpred[0][1] in self.toolkit_ai else
                    "tool_default" if tpred and tpred[0][1] in self.toolkit_default else
                    None
                )
                self.tool_call_records["tool_ai"].append(int(tool_type == "tool_ai"))
                self.tool_call_records["tool_default"].append(int(tool_type == "tool_default"))

                # If the tool is called, use the tool output. If the tool is not called, try to parse the AI message directly.
                msg_end = msg if not hasattr(msg, "tool_calls") or not msg.tool_calls else _force_final_answer(msg,
                                                                                                               tpred)

                rsp_dict = self.parse_response(msg_end)
                label, reason = rsp_dict["label"], rsp_dict["reason"]

                # # Validation of the parsed result
                # if label is not None:
                #     if label != tpred[0][0]:
                #         warnings.warn(
                #             f"AI parsing error:\n\tTool pred={tpred[0][0]}\n\tAI parsed pred={label}\n\tAI parsed reason={reason}")
                #     # label = tpred[0][0]
                # else:
                #     print(f"[Parser Error]:\n\tAI parsed pred={label}\n\tAI parsed reason={reason}")

                rsp_content_list.append({"label": label, "reason": reason})
            # print(f"Step 4: {time.perf_counter() - start:.4f} seconds")
            return rsp_content_list

        # ----------- Run the model to get predictions
        if not isinstance(x, (list, tuple)):
            x = [x]

        final_results = [{"label": None, "reason": None}] * len(x)
        retries = 0
        while retries < self.config["max_runs"] and any(r["label"] is None for r in final_results):
            incomplete_indices = [i for i, r in enumerate(final_results) if r["label"] is None]
            x_batch = [x[i] for i in incomplete_indices]
            print(f"Try to run _llm_batch with retries = {retries}.")
            batch_results = _llm_batch(x_batch)
            print(f"Progress: {len(final_results)-len(incomplete_indices)}/{len(final_results)}")
            for idx, result in zip(incomplete_indices, batch_results):
                if result["label"] is not None:
                    final_results[idx] = result
            retries += 1
        print(f"Progress: {len(final_results)}/{len(final_results)}")
        if any(r["label"] is None for r in final_results):
            raise TimeoutError("Run out of max_runs but still cannot get valid labels for all the signals")
        else:
            return [
                (r["label"], r["reason"]) if self.config["with_reason"] else r["label"]
                for r in final_results
            ]


