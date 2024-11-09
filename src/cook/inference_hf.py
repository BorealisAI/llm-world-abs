# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

_TASK_DESCRIPTION_RAW = '''
In TextWorld, You are an intelligent agent tasked with arranging food ingredients in a grid of rooms to match a specific desired configuration. \
Structure of the TextWorld: TextWorld consists of a grid-based map with multiple rooms. You start in one of these rooms.
Goals & Strategies: You start with an initial configuration of ingredients in rooms. \
Your job is to reach the desired state, a different configuration, through a sequence of operations. \
If there is any ingredient not mentioned in the desired state, that implies you should hold it in the end. If you think the desired state is reached, terminate the process.
The legal operations include moving and manipulation(take or drop) of ingredients: 
1. Moving: You can go north, south, west or east to enter adjoining rooms.
2. manipulation: Interact with ingredients inside the rooms by either taking or dropping them. You must be physically present in a room to interact with its contents.
Constraints: For each step, you can choose only one operation: either move or manipulate one ingredient. You can hold multiple ingredients at once.
'''


def reform_tmpl_for_mistral(msges):
    return [{
        'role': "user", 'content': msges[0]['content'] + msges[1]['content']
    }] + msges[2:]


class HFPrompter():

    def __init__(self, prompt_mode, tokenizer, for_mistral=False, task_prompt=_TASK_DESCRIPTION_RAW):
        self.prompt_mode = prompt_mode
        self.tokenizer = tokenizer
        self.task_descript = task_prompt
        self.for_mistral = False
        if self.prompt_mode in ['cot', 'icl', 'inst-ft']:
            self.msges = [{"role": "system", "content": task_prompt}]
        else:
            self.msges = []

    def make_query_prompt_fix(self, query, ret='str'):
          
        user = query['nl_loc_info']+"\n"
        user += f"Initial state: {query['init_state']}\n"
        user += f"Desired state: {query['final_state']}\n"
        if len(query['operations_hist']) > 1:
            user += f"Operations applied: {query['operations_hist']}\n"
        if self.prompt_mode == 'ft':
            user += 'operations: '
        else:    
            user += 'What are the operations to achieve the desired state?'
        msges = self.msges + [{'role': "user", "content": user.strip()}]
        if self.for_mistral:
            msges = reform_tmpl_for_mistral(msges)
        if ret == 'str':
            prompt = self.tokenizer.apply_chat_template(msges, tokenize=False, add_generation_prompt=True)  # + ":"


            return prompt
        else:
            return msges

    def make_query_and_ans_fix(self, input):
        msges = self.make_query_prompt_fix(input, ret='dict')
        msges += [{"role": "assistant", "content": input['actions'].strip()}]
        if self.for_mistral:
            msges = reform_tmpl_for_mistral(msges)

        ret = self.tokenizer.apply_chat_template(msges, tokenize=False, add_generation_prompt=False)

        return ret

    def proc_demos(self, demos):
        for x in demos:
            if self.prompt_mode == "icl":
                self.msges += self.make_icl_demo(x["text"])
            else:
                self.msges += self.make_cot_demo(x["text"])
        print('self.msges', self.msges)


class HFPrompterICL():

    def __init__(self, prompt_mode, tokenizer, for_mistral=False, task_prompt=_TASK_DESCRIPTION_RAW):
        self.prompt_mode = prompt_mode
        self.tokenizer = tokenizer
        self.task_descript = task_prompt
        self.for_mistral = for_mistral
        if self.prompt_mode in ['cot', 'icl', 'inst-ft']:
            self.msges = [{"role": "system", "content": task_prompt}]
        else:
            self.msges = []

    def make_icl_demo(self, inp, first_demo=False):
        user = ""
        user += inp['nl_loc_info']+"\n"
        user += f"Initial state: {inp['init_state']}\n"
        user += f"Desired state: {inp['final_state']}\n"
        if len(inp['operations_hist']) > 1:
            user += f"Operations applied: {inp['operations_hist']}\n"
        user += 'What are the operations to achieve the desired state?\n'
        return [
                {"role": "user", "content": user},
                {"role": "assistant", "content": inp['actions'].strip()}
            ]

    def make_query_prompt_fix(self, query, ret='str'):

        user = query['nl_loc_info']+"\n"
        user += f"Initial state: {query['init_state']}\n"
        user += f"Desired state: {query['final_state']}\n"
        if len(query['operations_hist']) > 1:
            user += f"Operations applied: {query['operations_hist']}\n"
        user += 'What are the operations to achieve the desired state?'
        msges = self.msges + [{'role': "user", "content": user.strip()}]

        if ret == 'str':
            ret = self.tokenizer.apply_chat_template(msges, tokenize=False, add_generation_prompt=True)
            print('ret')
            print(ret)
            return ret
        else:
            return msges

    def proc_demos(self, demos):
        for x in demos:
            if self.prompt_mode == "icl":
                self.msges += self.make_icl_demo(x["text"])
            else:
                self.msges += self.make_cot_demo(x["text"])
