import os
import argparse
import openai
import json
import sys
from tqdm import tqdm
import copy
import gym
import random
import re
import yaml
import alfworld
import alfworld.agents.environment
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)
# import utils
openai.api_key = open('KEY.txt').readlines()[0].rstrip()
os.environ["ALFWORLD_DATA"] = 'alfworld/data'


parser = argparse.ArgumentParser()
parser.add_argument("--LM", help="Name of OpenAI language model to be used. (by default planner)", type=str, default='text-davinci-003')
parser.add_argument("--exec-LM", help="Name of the Executor LLM to be used.", type=str, default='text-davinci-003')
parser.add_argument('--force-same-LM', help="force planner and executor LM to be the same.", type=bool, default=False)
parser.add_argument("--eval-type", help='Choose between in-domain or ood eval set (former is val set)', choices=['id', 'ood'], default='ood')
parser.add_argument("--fname", help='destination filename keyword for saving traces/logs', type=str, default = 'ADaPT_alfworld_run')
parser.add_argument('--num-task-samples', help='How much subsampling per task if at all?', default=5, type=int)
parser.add_argument('--eval-all', help='Should we evaluate on the full test set?', default=False, type=bool)
parser.add_argument('--max-depth', help='Max decomposition depth', default=3, type=int)
parser.add_argument('--max-runs', help='Max number of commands that can be executed in one run.', default=20, type=int)
parser.add_argument('--executor', help='Type of executor to be used', default='atomic', choices=['react', 'atomic', 'hybrid'])
parser.add_argument('--react-type', help='If using react exec, type of fs prompt to use.', default='std', choices=['std', 'cross', 'common'])
parser.add_argument('--store-results', help="Write the results in a separate file.", type=bool, default=True)
parser.add_argument('--no-store', help="Write the results in a separate file.", dest='store_results', action='store_false')
parser.add_argument('--info-prop-mode', help="What format of information should be propogated across executors?", default='last-step-last-act', choices=['last-step-last-act', 'all-step-last-act'])
parser.add_argument('--verbose', help='Verbosity: Print intermediate outputs.', default=False, type=bool)
parser.add_argument('--system-seed', help='Set common system independent seed', default=True, type=bool)
args = parser.parse_args()



LM = args.LM
if args.force_same_LM: 
    exec_LM = args.LM
else:
    exec_LM = args.exec_LM


max_runs = args.max_runs
environment_context = 'List of viable commands:\n\
- go to {recep}\n\
- open {recep}\n\
- close {recep}\n\
- take {obj} from {recep}\n\
- put {obj} in/on {recep}\n\
- use {lamp}\n\
- look\n\
- inventory\n\
- heat {obj} with {microwave}\n\
- cool {obj} with {fridge}\n\
- clean {obj} with {sink}\n\
- clean {obj} with {bathtub}\n\
- slice {obj} with {knife}\n\
where, "{recep}" denotes a receptacle, "{obj}" denotes any object from the environment, etc. Use the full name of the objects/receptacles including identifying number based on previous observations. Commands "heat", "cool", and "clean" are high-level shortcuts, so you should ONLY use "go to" commands before them. DO NOT use "open"/"put"/"take" commands if you plan to use "clean"/"heat"/"cool". Do not make assumptions if an object is hot/clean/cool without verifying.'


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def llm(prompt, stop=["\n"]):
    if 'davinci' in exec_LM:
      if isinstance(prompt, list): prompt = prompt[0]
      response = openai.Completion.create(
        model=exec_LM,
        prompt='Interact with a household to solve a task. ' + 'Commands "heat", "cool", and "clean" are high-level shortcuts, so you should only use "go to" commands before them. Do not use "open"/"put"/"take" commands if you plan to use "clean"/"heat"/"cool". Do not make assumptions if an object is hot/clean/cool without verifying.\n' + prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
      )
      return response["choices"][0]["text"]
    
    
       
    
    elif 'turbo-instruct' in exec_LM:
        if isinstance(prompt, list): prompt = prompt[0]
        response = openai.Completion.create(
            model=exec_LM,
            prompt='Interact with a household to solve a task.\n' + environment_context + '\n' + prompt,
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
      )
        return response["choices"][0]["text"]

    elif 'gpt-3.5' in LM or 'gpt-4' in exec_LM: 
          
    #   assert isinstance(prompt, list), print("Incorrect prompt format, expecting list of dict (messages)")
        messages = [
          {"role": "system", "content": 'You are a helpful robot navigating through a household. Interact with a household to solve a task by telling me the next action. Actions can be commands for the environment or thoughts/comments. All thoughts/comments or non-valid commands always start with "think: ". Actions will be passed to the environment which will return observations. Choose actions based on the observations.\n' + environment_context + '\n'},
	      {"role": "user", "content": prompt}
          ]  
        response = openai.ChatCompletion.create(
            model=exec_LM,
            messages=messages,
            
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )
        choices = response["choices"]
        completion_objs = [choice.message for choice in choices]
        completions = [completion.content for completion in completion_objs]
        return completions[0]
    

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
def plan_llm(prompt, stop=["\n\n"]):

    if 'davinci' in LM: 
      if isinstance(prompt, list): prompt = prompt[0]
      response = openai.Completion.create(
        model=LM,
        prompt='You are a helpful robot navigating through a household. ' + prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
      )
      return response["choices"][0]["text"]
    elif 'turbo-instruct' in LM:
        if isinstance(prompt, list): prompt = prompt[0]
        init_message = 'You are a helpful robot navigating through a household. The robot is capable of performing the following tasks:\n\
- Put an object on a receptacle\n\
- Take an object from a receptacle\n\
- Heat / Cool / Clean an object\n\
- Use a desklamp to look at an object.'
        response = openai.Completion.create(
            model=LM,
            prompt= init_message + '\n' + prompt,
            temperature=0,
            max_tokens=400,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )
        return response["choices"][0]["text"]
    elif 'gpt-3.5' in LM or 'gpt-4' in LM:
      messages = [
        {'role': 'system', 'content': "You are a helpful assistant generating plans to assist a robot navigate a household."},
        {'role': 'user', 'content':'The robot is capable of performing the following tasks:\n\
- Put an object on a receptacle\n\
- Take an object from a receptacle\n\
- Heat / Cool / Clean an object\n\
- Use a desklamp to look at an object.'
},
{"role": "user", "content": prompt}
	      ]  

      response = openai.ChatCompletion.create(
        model=LM,
        messages=messages,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
      )
      choices = response["choices"]
      completion_objs = [choice.message for choice in choices]
      completions = [completion.content for completion in completion_objs]
      return completions[0]
    
with open('base_config.yaml') as reader:
    config = yaml.safe_load(reader)

if args.eval_type == 'ood':    
    split = "eval_out_of_distribution"
elif args.eval_type == 'id':
    split = "eval_in_distribution"
else: assert False, 'Entered eval_type is incorrect!'

orig_env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
game_files = orig_env.game_files
if args.system_seed:
    game_files.sort()

env = orig_env.init_game(batch_size=1, game_file=game_files[2])

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]  
        ob = 'You arrive at the location. '  + ob
    return ob

folder = './prompts/'
prompt_file = 'alfworld_3prompts_endings.json'
plan_prompt_file = 'alfworld_plan_filled_prompts.json'
if 'davinci' in exec_LM:
    atomic_exec_file = 'alfworld_atomic_exec_prompts.json'
else: atomic_exec_file = 'alfworld_atomic_exec_prompts.json'

if args.store_results: 
    dest_file = './results/comparison/{}/{}_{}.json'.format(LM,args.fname,LM)
    t = open(dest_file, 'w+')
with open(folder + prompt_file, 'r') as f:
    d = json.load(f)
with open(folder + plan_prompt_file, 'r') as f:
    pdict = json.load(f)
with open(folder + atomic_exec_file, 'r') as f:
    exdict = json.load(f)


prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

ordered_game_files = {v:[] for v in prefixes.values()}
for game in game_files:
    for k,v in prefixes.items():
        if k in game: ordered_game_files[v].append(game)

random.seed(0)
sel_game_files = []
if not args.eval_all:
    for lst in ordered_game_files.values():
        sel_game_files.extend(random.sample(lst, k=args.num_task_samples))
else:
    for lst in ordered_game_files.values():
        sel_game_files.extend(lst)
print('Selected {} tasks from {}'.format(len(sel_game_files), split))

def convert_messages(prompt):
    messages = []
    entry = prompt.split('> ')
    messages.append({'type': 'env', 'content': entry[0]})
    for item in entry[1:]:
        item = item.rstrip('\n')
        cmmds = item.split('\n')
        if len(cmmds) > 1:
            act = cmmds[0]
            env = cmmds[1]
            if not len(env): env = 'OK.'
        else:
            act = cmmds[0]
            env = None
        messages.append({'type': 'act', 'content': act})
        if not env is None: messages.append({'type': 'env', 'content': env})
    return messages

def fetch_obj_recept(filename):
    entities = filename.split('-')
    if entities[2] != 'None':
        return [entities[1].lower(), entities[2].lower()], entities[3].lower()
    return [entities[1].lower(), ''], entities[3].lower()

def fetch_salient_info(completion, env, update_state=False):
    info = []
    completion = completion.rstrip('\n>')
    if 'Task completed!' not in completion: return ''
    lines = completion.split('\n')
    lines = lines[:-2]
    while len(lines) and 'think:' in lines[-2]:
        lines = lines[:-2]
    if not len(lines): return ''   
    act, obs = lines[-2], lines[-1]
    act = act.replace('Task completed!', '')
    info.extend([act, obs])
    obj = ''
    if update_state:
        for act in ['look', 'inventory', 'examine']:
            obs, _, _, _ = env.step([act])
            obs = obs[0]
            if act == 'inventory':
                if ":" in obs: 
                    obj = obs.split(': ')[-1].replace('a ', '').rstrip('.')
            if 'examine' in act:
                if obj!='': 
                    act = act.replace('<obj>', obj)
                    obs, _, _, _ = env.step([act])
                    obs = obs[0]
                else: return '\n'.join(info)
            info.extend(['> ' + act, obs])
    return '\n'.join(info)



atomic_exec_prompt = 'Here is a demo of actions you can perform.\n\n' + exdict['room'] + "\n\n"
for i in range(len(exdict.keys()) -1):
    atomic_exec_prompt += exdict['action_{}'.format(str(i))] + '\n\n'
atomic_exec_prompt += 'Here is a complex task for you to perform. If you think you have finished the task, think "Task completed!". If after trying your best, you conclude that the task is infeasible, think "Task failed!".'

plan_levels = [p for p in pdict.keys() if p != 'room']

if 'room' in pdict.keys():
    plan_prompt = "Here are some examples.\n" + pdict['room'] + '\n\n'
    for eg in plan_levels:
        plan_prompt += pdict[eg] + '\n\n'
    if 'davinci' in LM:
        plan_prompt += "Here is the goal.\n<room>\nGoal: <task>.\nCome up with an abstract plan to perform this task in a couple of steps. Constraints: The robot can hold/take/put only one object at a time to a location.\nEnsure each step can be understood independently and mentions the name of object.\nWhen stating the execution order, ensure that 'AND'/'OR' statements are properly nested using brackets '()'.\n"
    else:
        plan_prompt += "Here is the goal.\n<room>\nGoal: <task>.\nBased on the previously shown plans, come up with an abstract plan to perform this task in a couple of steps (NOT more than 3-4). Constraints:\n\
        - The robot can hold/take/put only one object at a time to a location.\n\
        - Ensure each step can be understood independently and mentions the name of object.\n\
        - DO NOT use 'OR' operations in the any step. Keep the step as abstract or generic as possible without mentioning location.\n\
        - DO NOT make assumptions about finding an object in a particular location or receptacle (if possible).\n\
        - When stating the execution order, ensure that 'AND'/'OR' statements are properly nested using brackets '()'.\n"
        

else:

    plan_prompt = "Here are some examples.\n" + '<room>' + '\n\n'
    for eg in plan_levels:
        plan_prompt += pdict[eg] + '\n\n'
    if 'davinci' in LM:
        plan_prompt += "Here is the goal.\n\nGoal: <task>.\nCome up with an abstract plan to perform this task in a couple of steps. Constraints: The robot can hold/take/put only one object at a time to a location.\nEnsure each step can be understood independently and mentions the name of object.\nWhen stating the execution order, ensure that 'AND'/'OR' statements are properly nested using brackets '()'.\n"
    else:
        plan_prompt += "Here is the goal.\n<room>\nGoal: <task>.\nCome up with an abstract plan to perform this task in a couple of steps. Constraints:\n\
        - The robot can hold/take/put only one object at a time to a location.\n\
        - Ensure each step can be understood independently and mentions the name of object.\n\
        - Do not use names of specific receptacles in steps nor assume you know if you will find an object in a receptacle.\n\
        - When stating the execution order, ensure that 'AND'/'OR' statements are properly nested using brackets '()'.\n"


def fetch_react_prompt(idx, prefixes, d, type):
    if type == 'std':
        v = prefixes[idx]
        prompt = 'Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0'] + '\nHere is the task. If you think you have finished the task, think "Task completed!". If after trying your best, you conclude that the task is infeasible, think "Task failed!\n'
    elif type == 'cross':
        alt_indices = [k for k in prefixes.keys() if k!=idx]
        v1 = prefixes[random.sample(alt_indices, k=1)[0]]
        v2 = prefixes[random.sample(alt_indices, k=1)[0]]
        v = prefixes[idx]
        prompt = 'Here are two examples.\n' + d[f'react_{v1}_1'] + d[f'react_{v2}_0'] + '\nHere is the task. If you think you have finished the task, think "Task completed!". If after trying your best, you conclude that the task is infeasible, think "Task failed!\n'
        
    elif type == 'common':
        v1 = 'heat'
        v2 = 'examine'
        v = prefixes[idx]
        prompt = 'Here are two example tasks.\n' + d[f'react_{v1}_1'] + d[f'react_{v2}_0'] + '\nHere is the task. If you think you have finished the task, think "Task completed!". If after trying your best, you conclude that the task is infeasible, think "Task failed!\n'
    return prompt

def fill_template(objs, recept, p_temp):
    plan = []
    for p in p_temp:
        p = p.replace('[obj]', objs[0])
        p = p.replace('[obj2]', objs[1])
        p = p.replace('[recept]', recept)
        plan.append(p)
    return plan

def print_completion(completion):
    out_lines = ['\t-----']
    lines = completion.split('\n')
    lines = ['\t' + line for line in lines]
    out_lines.extend(lines)
    out_lines.append('\t-----')
    out_text = "\n".join(out_lines)
    print(out_text)
    return

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

def fetch_args(args_lookup, logic_exp):
    out = copy.deepcopy(logic_exp)
    assert 'steps' in logic_exp.keys()
    for s, step in enumerate(logic_exp['steps']):
        if isinstance(step, int):
            out['steps'][s] = args_lookup[step]
        elif isinstance(step, dict):
            out['steps'][s] = fetch_args(args_lookup, step)
    return out

def alfworld_run(prompt, to_print=True, ob='', env=env, max_runs=max_runs, output_term=True):
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
        if isinstance(init_prompt, list):
            if len(prompt):
                action = llm(init_prompt + convert_messages(prompt), stop=['\n']).strip()
            else: 
                action = llm(init_prompt, stop=['\n']).strip()
        else:
            action = llm(init_prompt + prompt, stop=['\n']).strip()
        num_runs += 1
        action = action.lstrip('> ')
        if action.startswith('put'):
            action = action.replace(' in ', ' in/on ').replace(' on ', ' in/on ')
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]

        if action.startswith('think:'):
            observation = 'OK.'
            if 'task completed!' in action.lower(): done = True; success = True
            if 'task failed!' in action.lower(): done = True; success = False
        else: action_history.append(action)
        if observation == "Nothing happens." or observation == "OK.":
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
    

def plan_and_run(room, task, env, prompt, past_action_checkpoint=[], past_info_prop = '', depth = 1, num_runs = 0, verbose = False, comm_state = False, ttype=''):
    max_depth = args.max_depth
    plan_list = []
    env.reset()
    if len(past_action_checkpoint):
        if verbose: print('Loaded Checkpoint: ', past_action_checkpoint)
        # Load up env to previous state
        for act in past_action_checkpoint:
            observation, reward, done, info = env.step([act])
        

    info_prop = past_info_prop 
    action_checkpoint = past_action_checkpoint
    running_completion = ''
    if isinstance(task, str):
        if verbose: print('Starting... ' + task, ' at depth ' + str(depth))
        custom_ob = room + '\nYour task is to: ' + task
        if len(info_prop):
            if verbose: print('Loading ...', info_prop)
            if exec_LM in ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-0301"]:
                r, succ, term, completion, act_history, num = alfworld_run(prompt, to_print=False, ob=custom_ob + '\n' +  info_prop , env=env) 

            else:
                r, succ, term, completion, act_history, num = alfworld_run(prompt, to_print=False, ob=custom_ob + '\n' +  info_prop , env=env) 
        else:
            r, succ, term, completion, act_history, num = alfworld_run(prompt, to_print=False, ob=custom_ob, env=env)
        if verbose: 
            print_completion(completion)
            print('Task ({}) Success: '.format(task), succ)
        plan_list.append(task + ' at depth ' + str(depth) + ', success: ' + str(succ))
        num_runs += num
        if succ or depth >= max_depth:
            if succ: action_checkpoint.extend(act_history)
            running_completion += completion + '\n'
            if succ:
                if args.info_prop_mode == 'last-step-last-act':
                    info_prop = fetch_salient_info(running_completion, env, update_state=comm_state)
                elif args.info_prop_mode == 'all-step-last-act':
                    info_prop = past_info_prop + '\n' + fetch_salient_info(running_completion, env, update_state=comm_state)
                    info_prop = info_prop.lstrip('\n')
            return r, succ, term, env, running_completion, action_checkpoint, depth, plan_list, num_runs, info_prop
        
        
            
        plan = plan_llm(plan_prompt.replace('<room>', room).replace('<task>', task))
        if verbose: print_completion(plan)
        plan_steps = plan_to_args(plan)
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
        r, succ, term, env, completion, act_history, _, decomp_plans, num, info_prop = plan_and_run(room, sub_task, env, prompt, past_action_checkpoint=action_checkpoint, past_info_prop=info_prop, depth=depth, verbose=verbose, comm_state=comm_state, ttype=ttype) #Need to propogate info_prop here.
        plan_list.extend(decomp_plans)
        num_runs += num
        if plan_steps['logic'].lower() == 'or':
            if succ:
                if not set(act_history).issubset(action_checkpoint): action_checkpoint.extend(act_history)
                running_completion += completion + '\n'
                if args.info_prop_mode == 'last-step-last-act':
                    info_prop = fetch_salient_info(running_completion, env, update_state=comm_state)
                return r, succ, term, env, running_completion, action_checkpoint, depth, plan_list, num_runs, info_prop
        # If reached here you have succeeded.
        if succ:
            if not set(act_history).issubset(action_checkpoint): action_checkpoint.extend(act_history)
            running_completion += completion + '\n'
            if args.info_prop_mode == 'last-step-last-act':
                info_prop = fetch_salient_info(running_completion, env, update_state=comm_state)
    
        if plan_steps['logic'].lower() == 'and' and not succ:
            return r, succ, term, env, running_completion, action_checkpoint, depth, plan_list, num_runs, info_prop

    return r, succ, term, env, running_completion, action_checkpoint, depth, plan_list, num_runs, info_prop


outputs = {k:{} for k in prefixes.keys()}
run_count = [0] * 6
cnts = [0] * 6
rs = [0] * 6
rate = 0.0
pbar = tqdm(sel_game_files)
pbar.set_postfix({'success rate': rate})
for game in pbar:
    env = orig_env.init_game(batch_size=1, game_file=game)
    ob, info = env.reset()
    ob = '\n'.join(ob[0].split('\n\n')[1:])
    name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
    room, task = ob.split('\n')
    task = task.split(': ')[-1]
    trace = []
    for i, (k, v) in enumerate(prefixes.items()):
        if name.startswith(k):
            if args.executor == 'atomic':
                r, succ, term, env, trace, actions, depth, plans, num_runs, _ = plan_and_run(room, task, env, atomic_exec_prompt, past_action_checkpoint=[], comm_state=False, verbose=args.verbose, ttype=k)
            elif 'hybrid' == args.executor:
                prompt = '\n'.join(atomic_exec_prompt.split('\n')[:-1])
                prompt = prompt + fetch_react_prompt(k, prefixes, d, 'common')
                r, succ, term, env, trace, actions, depth, plans, num_runs, _ = plan_and_run(room, task, env, prompt, past_action_checkpoint=[], comm_state=False, verbose=args.verbose, ttype=k)
            elif 'react' == args.executor:
                r, succ, term, env, trace, actions, depth, plans, num_runs, _ = plan_and_run(room, task, env, fetch_react_prompt(k,prefixes,d,args.react_type), past_action_checkpoint=[], comm_state=False, verbose=args.verbose,ttype=k)
            rs[i] += r
            cnts[i] += 1
            run_count[i] += num_runs/args.num_task_samples
            rate = sum(rs)/sum(cnts)
            outputs[k][name] = {'problem': ob, 'trace': trace, 'plans': plans, 'reward': r, 'runs': num_runs}
            pbar.set_postfix({'rate': rate})
            break
    if args.verbose: print('\n\n')
print('rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts), 'runs: ', sum(run_count)/6)
outputs['overall'] = {'rate': sum(rs) / sum(cnts), 'runs': sum(run_count)/6, 'success': rs, 'count': cnts}
if args.store_results:
    json.dump(outputs, t, indent=4)
    t.close()

        