from algorithms.BaseChain import BaseChain

class InContext(BaseChain):
    def __init__(self, config, train_examples=None, adapt_examples=None):
        if config["with_reason"]:
            dir_human = "./messages/incontext_human.txt"
        else:
            dir_human = "./messages/incontext_human_without_reason.txt"

        dir_system = "./messages/incontext_system.txt"
        dir_ai = "./messages/incontext_ai.txt"

        super(InContext, self).__init__(config, dir_system, dir_human, dir_ai, train_examples, adapt_examples)