import json
import os
import openai
import sys
import json
import re
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
    RetryError
)
import copy
import ast
import numpy as np
sys.path.append("../EnvironmentWebs/environments/")
from textcraft.env import TextCraft

openai.api_key = open('KEY.txt').readlines()[0].rstrip()
LM = 'gpt-3.5-turbo-instruct'
max_runs = 40
max_depth = 4
num_games = 200
verbose = False
env = TextCraft(minecraft_dir="../EnvironmentWebs/environments/textcraft/")

environment_context = '''You can perform the following actions to interact with the environment: 
- craft [target count] [target item] using [count] [item]
- get [count] [item]
- inventory
Here [count] is a place holder for number of object, and [item] is placeholder for name of object.

'''

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))
def llm(prompt, stop=["\n"], max_tokens=150):
    if 'davinci' in LM or 'instruct' in LM:
        response = openai.Completion.create(
        model=LM,
        prompt='You are a helpful assistant playing Minecraft\n' + environment_context + prompt,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
        )
        return response["choices"][0]["text"]
    elif 'gpt-3.5-turbo' in LM or 'gpt-4' in LM:
        response = openai.ChatCompletion.create(
        model=LM,
        messages=[
          {"role": "system", "content": 'You are a helpful assistant playing Minecraft.' + environment_context},
          {"role": "user", "content": prompt}
	      ],
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
        )
        choices = response["choices"]
        completion_objs = [choice.message for choice in choices]
        completions = [completion.content for completion in completion_objs]
        return completions[0]

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(25))
def plan_llm(prompt, stop=["\n\n"]):
    if 'davinci' in LM or 'turbo-instruct' in LM:
      if isinstance(prompt, list): prompt = prompt[0]
      response = openai.Completion.create(
        model=LM,
        prompt= 'You are an helpful assistant helping me play a simple version of Minecraft. ' + prompt,
        temperature=0,
        max_tokens=800,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        # stop=stop
      )
      return response["choices"][0]["text"]
    elif 'gpt-3.5-turbo' in LM or 'gpt-4' in LM:
      if isinstance(prompt, list): prompt = prompt[0]
      init_prmpt = 'You are an helpful assistant helping me play a simple version of Minecraft.'
      response = openai.ChatCompletion.create(
      model=LM,
      messages=[
        {"role": "system", "content": init_prmpt},
        {"role": "user", "content": prompt}
      ],
      temperature=0,
      max_tokens=800,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      # stop=stop
      )
      choices = response["choices"]
      completion_objs = [choice.message for choice in choices]
      completions = [completion.content for completion in completion_objs]
      return completions[0]

    
def parse_expression(expression):
    stack = []
    current = {}
    for token in re.findall(r'Step \d+|AND|OR|\(|\)', expression):
        if token.startswith('Step'):
            if 'steps' not in current:
                current['steps'] = []
            current['steps'].append(int(token.split()[1]))
        elif token in ('AND', 'OR'):
            current['logic'] = token
        elif token == '(':
            stack.append(current)
            current = {}
        elif token == ')':
            closed = current
            current = stack.pop()
            if 'steps' not in current:
                current['steps'] = []
            current['steps'].append(closed)
    return current

def plan_to_args(plan, keyword = 'Step', lkey = 'execution order'):
    args = []
    lines = plan.split('\n')
    for line in lines:
        if line.startswith(keyword): args.append(re.sub(r'{} \d+: '.format(keyword), '', line))
        if lkey in line.lower():
            logic = line.split(': ')[-1]
    args_lookup = {i+1: args[i] for i in range(len(args))}
    try:
        return fetch_args(args_lookup, parse_expression(logic))
    except: 
        return {'steps': args, 'logic': 'AND'}

def print_completion(completion):
    out_lines = ['\t-----']
    lines = completion.split('\n')
    lines = ['\t' + line for line in lines]
    out_lines.extend(lines)
    out_lines.append('\t-----')
    out_text = "\n".join(out_lines)
    print(out_text)
    return

def fetch_args(args_lookup, logic_exp):
    out = copy.deepcopy(logic_exp)
    assert 'steps' in logic_exp.keys()
    for s, step in enumerate(logic_exp['steps']):
        if isinstance(step, int):
            out['steps'][s] = args_lookup[step]
        elif isinstance(step, dict):
            out['steps'][s] = fetch_args(args_lookup, step)
    return out

def textcraft_run(prompt, to_print=True, ob='', env=env, max_runs=max_runs, output_term=True):
    if isinstance(prompt, list): 
        init_prompt = copy.copy(prompt)
        init_prompt.append({'type': 'env', 'content': ob})
    else:
        init_prompt = prompt + '\n' + ob + '\n>'
    prompt = ''
    action_history = []
    max_patience = 5
    pat_ctr = 0
    success = False
    terminate = False
    num_runs = 0
    if to_print:
        print(ob)
        sys.stdout.flush()
    for i in range(1, max_runs):
        
        action = llm(init_prompt + prompt, stop=['\n']).strip()
        num_runs += 1
        action = action.lstrip('> ')
        
        observation, reward, done, _,  info = env.step('> ' + action)

        if action.startswith('think:'):
            observation = 'OK.'
            if 'task completed' in action.lower(): done = True; success = True
            if 'task failed' in action.lower(): done = True; success = False
        else: action_history.append(action)
        if observation.startswith("Could not") or observation == "OK.": 
            pat_ctr += 1
            if pat_ctr == max_patience: terminate = True; break
        else: pat_ctr = 0
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        prompt += f' {action}\n{observation}\n>'
        if reward: success = True; terminate = False
        if done:
            return reward, success, terminate, prompt, action_history, num_runs
    if not done: success = False; terminate = True
    return 0, success, terminate,  prompt, action_history, num_runs


def textcraft_run(prompt, to_print=True, ob='', env=env, max_runs=max_runs, output_term=True):
    if isinstance(prompt, list): 
        init_prompt = copy.copy(prompt)
        init_prompt.append({'type': 'env', 'content': ob})
    else:
        init_prompt = prompt + '\n' + ob + '\n>'
    prompt = ''
    action_history = []
    max_patience = 8
    pat_ctr = 0
    success = False
    terminate = False
    num_runs = 0
    if to_print:
        print(env.step('inventory'))
        print(ob)
        sys.stdout.flush()
    for i in range(1, max_runs):
        
        action = llm(init_prompt + prompt, stop=['\n']).strip()

        num_runs += 1
        action = action.lstrip('> ')
        
        observation, reward, done, _,  info = env.step(action)
        
        if 'task completed' in action.lower(): done = True; success = True
        if 'task failed' in action.lower(): done = True; success = False
        if action.startswith('think'):
            observation = 'OK.'
            if 'task completed' in action.lower(): done = True; success = True
            if 'task failed' in action.lower(): done = True; success = False
        else: action_history.append(action)
        if "Could not" in observation or observation == "OK.": 
            pat_ctr += 1
            if pat_ctr == max_patience: terminate = True; break
        else: pat_ctr = 0
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            # print(pat_ctr)
            sys.stdout.flush()
        prompt += f' {action}\n{observation}\n>'
        if reward > 0: success = True; terminate = False
        if done:
            return reward, success, terminate, prompt, action_history, num_runs
    if not done: success = False; terminate = True
    return 0, success, terminate,  prompt, action_history, num_runs

plan_prompt = '''Your task is to come up with a short plan to help me accomplish my goal in a couple of steps using at most ONE of the provided crafting commands. You can take the help of crafting commands below to create new objects. 
Craft command can be understood as follows: craft [target] using [ingredients], where target is item/object generated by the craft command as output and ingredient are the inputs. You are given an agent that can "craft" or "fetch" objects.

Here is are some examples.

Crafting commands:
craft 3 dark oak sign using 6 dark oak planks, 1 stick
craft 4 dark oak planks using 1 dark oak log
craft 1 stick using 1 planks
craft 4 stick using 2 bamboo
craft 4 oak planks using 1 oak log
craft 1 dark oak fence using 2 stick, 4 dark oak planks
craft 1 warped stairs using 6 warped planks
craft 3 oak sign using 6 oak planks, 1 stick

Goal: craft dark oak sign.

# Think: My target is a dark oak sign. From the list of crafting commands, only 1 command generates my target: craft 3 dark oak sign using 6 oak planks, 1 stick. I will use this command to devise a plan. My ingredients are: 6 dark oak planks, 1 stick. I should first get all the ingredients and then use the crafting command.
Step 1: fetch 6 dark oak planks
Step 2: fetch 1 stick
# Think: Now that I have collected the input ingredients, I can craft the dark oak sign using given command.
Step 3: craft dark oak sign using 6 dark oak planks, 1 stick
# Think: To succeed, I need to perform all these steps, one after the other. So I need to use the "AND" operator.
Execution Order: (Step 1 AND Step 2 AND Step 3)

Goal: fetch 6 dark oak planks.

# Think: My target is 6 dark oak planks. From the list of crafting commands, only 1 command generates my target: craft 4 dark oak planks using 1 dark oak log. My ingredients are: 1 dark oak log. To successfully accomplish the goal, I should first get all the ingredients and then use the crafting command.
Step 1: fetch 1 dark oak log
# Think: Now that I have collected the input ingredients, I can craft dark oak planks using given command. I know that I cannot use a partial recipe.
Step 2: craft 4 dark oak planks using 1 dark oak log
# Think: This gives me 4 dark oak planks which is less than my desired 6 dark oak planks. I know that I cannot use a partial recipe. So my goal is not satisfied, I need to craft more dark oak planks by repeating Step 2 one more time.
Step 3: craft 4 dark oak planks using 1 dark oak log
# Think: To succeed, I need to perform all these steps, one after the other. So I need to use the "AND" operator.
Execution Order: (Step 1 AND Step 2 AND Step 3)

Here is a different goal with different craft commands. Your task is to come up with a short plan to help me accomplish my goal in a couple of steps using at most ONE of the provided crafting commands. You can take the help of crafting commands below to create new objects. Keep in mind that:
- It is okay to generate more target objects than your goal.
- Be very careful with the count of objects, SAME object counts mentioned in the input crafting command. 
- You cannot use a partial crafting command recipe, i.e. if the recipe generates 2 objects you CANNOT alter it to produce just 1. 
- Also, you can use ONLY 1 crafting command in your plan.
'''

atomic_examples = {
'craft_with_ingredients':'''Crafting commands:
craft 3 dark oak sign using 6 dark oak planks, 1 stick
craft 4 dark oak planks using 1 dark oak log
craft 1 stick using 1 planks
craft 4 stick using 2 bamboo
craft 4 oak planks using 1 oak log
craft 1 dark oak fence using 2 stick, 4 dark oak planks
craft 1 warped stairs using 6 warped planks
craft 3 oak sign using 6 oak planks, 1 stick

Goal: craft dark oak sign

> think: I should check if I can fetch dark oak sign directly from the environment or the inventory.
OK.

> inventory: 
Inventory: [stick] (1) [dark oak planks] (8)

> get dark oak sign
Could not find dark oak sign

> think: I cannot get dark oak sign directly, I need to craft it. From the crafting commands, I can use: craft dark oak sign using 6 dark oak planks, 1 stick. Ingredients needed: 6 dark oak planks, 1 stick. Input assumption: I have all the neccessary ingredients in my inventory. Let me verify this first.
OK.

> inventory
Inventory: [stick] (1) [dark oak planks] (8)

> think: I found my ingredients: 6 dark oak planks, 1 stick in my inventory. My assumption is true, I can proceed. I will use the crafting command: craft dark oak sign using 6 dark oak planks
OK.

> craft 1 dark oak sign using 6 dark oak planks, 1 stick
Crafted 1 minecraft:dark_oak_sign

> inventory 
Inventory: [dark oak sign] (1)

> think: I now have dark oak sign in my inventory. Task Completed!
OK.
''', 
'craft_with_ingredients_gen':'''Goal: fetch 2 dark oak logs.

> think: I should check my inventory first, to see if I already have dark oak sign. Otherwise, I will directly try to get it from the environment.
OK.

> inventory
Inventory: [stick] (1)

> get 2 dark oak logs.
Got 2 dark oak logs

> inventory
Inventory: [dark oak log] (2) [stick] (1)

> think: I have 2 dark oak logs in my inventory. Task Completed!
OK.
'''
}

react_trajectory = '''Goal: craft 2 oak planks

> think: I should check if I can fetch dark oak sign directly from the environment or the inventory.
OK.

> inventory: 
Inventory: [stick] (1) 

> get 2 oak planks
Could not find 2 oak planks

> think: I cannot get oak planks directly, I need to craft it. From the crafting commands, I can use: craft 4 oak planks using 1 oak log. Ingredients needed: 1 dark oak log. Input assumption: I have the ingredients needed in my inventory. Let me verify this first.
OK.

> inventory
Inventory: [stick] (1) 

> think: My assumption is false. I need to get the ingredients: 1 oak log first.
OK.

> get 1 oak log
Got 1 oak log

> craft 4 oak planks using 1 oak log
Crafted 4 minecraft:oak_planks

> inventory
Inventory: [stick] (1) [oak_planks] (4)

> think: I now have 2 oak planks in my inventory. Task Completed!
OK.
'''

atomic_exec_prompt = '''You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format "craft [target object] using [input ingredients]". You can either "fetch" an object (ingredients) from the inventory or the environment or "craft" (target) using any of the crafting commands. You can use ONLY these crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. "dark oak planks" in the command instead. For any other natural language or thoughts, use prefix 'think: '.

Here is a demo of how to fetch and craft objects.\n\n'''
atomic_exec_prompt +=  '\n\n'.join(atomic_examples[k] for k in atomic_examples.keys()) + '\n'
atomic_exec_prompt += 'Here is an example of a complex goal.\n\n' + react_trajectory + '\n'
atomic_exec_prompt += "Now here is a different goal. You can use these crafting commands to accomplish the goal. When you the desired item in your inventory, think: Task Completed! If you have tried your best but cannot proceed, think: task failed!\n" 

def plan_and_run(commands, task, idx, env, prompt, past_action_checkpoint=[], past_info_prop = '', depth = 1, num_runs = 0, verbose = False, comm_state = False, ttype=''):
    plan_list = []

    info_prop = past_info_prop 
    action_checkpoint = past_action_checkpoint
    running_completion = ''
    if isinstance(task, str):
        if verbose: print('Starting... ' + task, ' at depth ' + str(depth))
        custom_ob = commands + '\nGoal: ' + task + '\n'
        if len(info_prop):
            if verbose: print('Loading ...', info_prop)
            if LM in ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-0301"]:
                r, succ, term, completion, act_history, num = textcraft_run(prompt, to_print=False, ob=custom_ob + '\n' +  info_prop , env=env) #'\nResume from loaded checkpoint, last finished action:\n' +

            else:
                r, succ, term, completion, act_history, num = textcraft_run(prompt, to_print=False, ob=custom_ob + '\n' +  info_prop , env=env) #'\nResume from loaded checkpoint, last finished action:\n' +
        else:
            r, succ, term, completion, act_history, num = textcraft_run(prompt, to_print=False, ob=custom_ob, env=env)
        if verbose: 
            print_completion(completion)
            print('Task ({}) Success: '.format(task), succ)
        plan_list.append(task + ' at depth ' + str(depth) + ', success: ' + str(succ))
        num_runs += num
        if succ or depth >= max_depth:
            if succ: action_checkpoint.extend(act_history)
            running_completion += completion + '\n'
            if succ:
                info_prop = '> inventory\n'
                obs, _, _, _, _ = env.step('inventory')
                info_prop += obs + '\n'
            return r, succ, term, env, running_completion, action_checkpoint, depth, plan_list, num_runs, info_prop
        
        
        plan = plan_llm(plan_prompt + '\n' + custom_ob)
        if verbose: 
            print('-----')
            print_completion(plan)
            print('-----')
        plan_steps = plan_to_args(plan)
        if verbose: print(plan_steps)
        num_runs += 1
        if len(plan_steps['steps']) == 1: 
            plan_steps = plan_steps['steps'][0]
            if type(plan_steps) == str: plan_steps={'steps':[plan_steps]}
            if 'logic' not in plan_steps.keys():
                try:
                    logic = plan_steps['logic']
                except: logic = "AND"; plan_steps['logic'] = logic
        depth += 1
    else:
        plan_steps = task
        try: logic = plan_steps['logic']
        except: logic = "AND"; plan_steps['logic'] = logic

    if verbose: print('Identified subtasks... ' + str(plan_steps['steps']) + ' at depth {}, logic: '.format(depth) + str(plan_steps['logic']))
    plan_list.append(str(plan_steps['steps']) + ' at depth ' + str(depth) + ' and logic ' + str(plan_steps['logic']))
    for sub_task in plan_steps['steps']:
        if verbose: print('At subtask: ' + str(sub_task))
        r, succ, term, env, completion, act_history, _, decomp_plans, num, info_prop = plan_and_run(commands, sub_task, idx, env, prompt, past_action_checkpoint=action_checkpoint, past_info_prop=info_prop, depth=depth, verbose=verbose, comm_state=comm_state, ttype=ttype) #Need to propogate info_prop here.
        plan_list.extend(decomp_plans)
        num_runs += num
        if plan_steps['logic'].lower() == 'or':
            if succ:
                if not set(act_history).issubset(action_checkpoint): action_checkpoint.extend(act_history)
                running_completion += completion + '\n'
                info_prop = '> inventory\n'
                obs, _, _, _, _ = env.step('inventory')
                info_prop += obs + '\n'
                return r, succ, term, env, running_completion, action_checkpoint, depth, plan_list, num_runs, info_prop
        # If reached here you have succeeded.
        if succ:
            if not set(act_history).issubset(action_checkpoint): action_checkpoint.extend(act_history)
            running_completion += completion + '\n'
            info_prop = '> inventory\n'
            obs, _, _, _, _ = env.step('inventory')
            info_prop += obs + '\n'
    
        if plan_steps['logic'].lower() == 'and' and not succ:
            return r, succ, term, env, running_completion, action_checkpoint, depth, plan_list, num_runs, info_prop

    return r, succ, term, env, running_completion, action_checkpoint, depth, plan_list, num_runs, info_prop\


## Running main loop for evaluation
outputs = {}
pbar = tqdm(list(range(num_games)))
rs = []; cnts = []
rate = 0.0
pbar.set_postfix({'success rate': rate})
env = TextCraft(minecraft_dir="../EnvironmentWebs/environments/textcraft/")
for idx in pbar:
    obs, info = env.reset(seed=idx)
    commands, task = obs.split('Goal: ')
    trace = []
    r, succ, term, env, trace, actions, depth, plans, num_runs, _ = plan_and_run(commands, task, idx, env, atomic_exec_prompt, past_action_checkpoint=[], comm_state=False, verbose=verbose)
    rs.append(r)
    cnts.append(1)
    rate = sum(rs)/sum(cnts)
    outputs[f'env_{idx}'] = {'problem': task, 'commands': commands, 'trace': trace, 'plans': plans, 'reward': r, 'runs': num_runs}
    pbar.set_postfix({'rate': rate})
outputs['overall'] = {'rate': sum(rs) / sum(cnts),  'count': cnts}

dest_file = './results/textcraft/{}.json'.format(f'ADaPT_maxd{max_depth}_hybrid_exec_runs{max_runs}_test_num{num_games}')
os.makedirs('/'.join(dest_file.split('/')[:-1]), exist_ok=True)
t = open(dest_file, 'w+')
json.dump(outputs, t, indent=4)
t.close()
