# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

_TASK_DESCRIPTION_RAW = '''You are an intelligent agent in a fictional environment with a sequence of containers arranged in a straight line. \
The environment involves movement, interaction with objects, and strategic planning to reach a particular state.
Structure of the Environment: There are containers arranged in a line from left to right. You are at one of them in the beginning.
Goals & Strategies: You start with an initial configuration of objects in containers. \
Your job is to reach the desired state, a different configuration, through a sequence of operations. \
If there is any object not mentioned in the desired state, that implies you should hold it in the end. If you think the desired state is reached, terminate the process.
The legal operations include moving and manipulation(grab or put) of objects: 
1. Moving: You can only walk left or right. Each time you take a step, you'll be right next to the container on your left or right side.
2. manipulation: Inside some containers, there are objects. To grab an object or put one inside, you need to be at that container. If you see an object in a container far away, you can't grab it. You need to walk over to that container first.
Constraints: For each step, you can choose only one operation: either move or manipulate one object. You can hold multiple objects at once.\
'''

class HFPrompter():

    def __init__(self, prompt_mode, tokenizer):
        self.prompt_mode = prompt_mode
        self.tokenizer = tokenizer
        self.task_descript = _TASK_DESCRIPTION_RAW
        
        if self.prompt_mode in ['cot', 'icl', 'inst-ft']:
            self.msges = [{"role": "system", "content": _TASK_DESCRIPTION_RAW}]
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
        if ret == 'str':
            return self.tokenizer.apply_chat_template(msges, tokenize=False, add_generation_prompt=True)
        else:
            return msges

    def make_query_and_ans_fix(self, input):
        msges = self.make_query_prompt_fix(input, ret='dict')
        msges += [{"role": "assistant", "content": input['actions']}]
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

    def __init__(self, prompt_mode, tokenizer):
        self.prompt_mode = prompt_mode
        self.tokenizer = tokenizer
        self.task_descript = _TASK_DESCRIPTION_RAW
        
        if self.prompt_mode in ['cot', 'icl', 'inst-ft']:
            self.msges = [{"role": "system", "content": _TASK_DESCRIPTION_RAW}]
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
            return self.tokenizer.apply_chat_template(msges, tokenize=False, add_generation_prompt=True)
        else:
            return msges

    def make_query_and_ans(self, input):
        msges = self.make_query_prompt(input, ret='dict') 
        msges += [{"role": "assistant", "content": input['actions']}]
        ret = self.tokenizer.apply_chat_template(msges, tokenize=False, add_generation_prompt=True)
        return ret

    def make_query_and_ans_fix(self, input):
        msges = self.make_query_prompt_fix(input, ret='dict')
        msges += [{"role": "assistant", "content": input['actions']}]
        ret = self.tokenizer.apply_chat_template(msges, tokenize=False, add_generation_prompt=False)
        return ret

    def proc_demos(self, demos):
        for x in demos:
            if self.prompt_mode == "icl":
                self.msges += self.make_icl_demo(x["text"])
            else:
                self.msges += self.make_cot_demo(x["text"])
    


