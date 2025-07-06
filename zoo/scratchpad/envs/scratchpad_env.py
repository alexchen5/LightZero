from enum import Enum
from gymnasium import spaces
import numpy as np
from typing import TypedDict, Literal, Iterable
from copy import deepcopy
import logging

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY

END_OF_TEXT = 42

class Action(Enum): 
    TO_INPUT = 0        # move cursor to input token array
    TO_SCRATCHPAD = 1   # move cursor to scratchpad token array
    TO_LEFT = 2         # move cursor left one index 
    TO_RIGHT = 3        # move cursor right one index 
    START_HIGHLIGHT = 4 # start highlighting at cursor index 
    STOP_HIGHLIGHT = 5  # stop highlighting at cursor index 
    CLONE = 6           # clone highlighted selection to cursor index
    DELETE = 7          # delete highlighted selection 
    LLM_INPUT = 8       # replace llm input with highlighted selection
    LLM_GENERATE = 9    # generate the next output token of the internal llm 
    LLM_DELETE = 10     # delete last token of llm output
    LLM_OUTPUT = 11     # write llm output to cursor index 
    OUTPUT = 12         # output the (single) token at cursor index 
    
class State(TypedDict):
    # observation = [
    # 0  [ ...input, ...scratchpad, ...llm_input, ...llm_output, ...output ]
    # 1  [ ...1, ...0 ] (input attention mask)
    # 2  [ ...0, ...1, ...0 ] (scratchpad attention mask)
    # 3     (llm_input attention mask)
    # 4     (llm_output attention mask)
    # 5     (output attention mask)
    # 6  [ ...0, 1, ...0 ] (cursor position)
    # 7  [ ...0, 1, ...0 ] (start highlight position)
    # 8  [ ...0, ...1, ...0 ] (highlighted)
    # ]
    input: np.ndarray
    scratchpad: np.ndarray
    llm_input: np.ndarray
    llm_output: np.ndarray
    output: np.ndarray
    cursor_pos: np.ndarray
    cursor_highlight: np.ndarray
    
class Observation(TypedDict):
    observation: np.ndarray
    action_mask: np.ndarray
    to_play: Literal[-1]

def state_to_observation(state: State) -> np.ndarray:
    raw_segments = [state['input'], state['scratchpad'], state['llm_input'], state['llm_output'], state['output']]
    segments = [np.concatenate([raw_segment, [END_OF_TEXT]]) for raw_segment in raw_segments]
    tokens = np.concatenate(segments)
    total_text_dim = len(tokens)

    # Row 0: Tokens
    row_0 = tokens
    
    # Compute start and end indices for each original (non-<eot>) segment
    masks_i = np.cumsum([0] + [len(segment) for segment in segments[:-1]])
    masks_s = [masks_i[i] + len(raw_segment) for i, raw_segment in enumerate(raw_segments)]
    
    # Rows 1–5: Attention masks
    masks = [np.zeros(total_text_dim, dtype=np.float32) for _ in range(5)]
    for i in range(5):
        masks[i][masks_i[i]:masks_s[i]] = 1
    row_1, row_2, row_3, row_4, row_5 = masks 
    
    # Row 6: Cursor position (one-hot)
    row_6 = np.zeros(total_text_dim, dtype=np.float32)
    cursor_seg, cursor_i = state['cursor_pos']
    cursor_i = masks_i[cursor_seg] + cursor_i
    row_6[cursor_i] = 1
    
    # Row 7: Cursor highlight start (one-hot)
    # Row 8: Highlighted span
    row_7 = np.zeros(total_text_dim, dtype=np.float32)
    row_8 = np.zeros(total_text_dim, dtype=np.float32)
    
    highlight_seg, highlight_i, highlight_s = state['cursor_highlight']
    highlight_i = masks_i[highlight_seg] + highlight_i
    highlight_s = masks_i[highlight_seg] + highlight_s
    row_7[highlight_i] = 1
    row_8[highlight_i:highlight_s] = 1
    
    dim2 = np.stack([row_0, row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8], dtype=np.float32)
    dim3 = dim2.reshape(1, 9, -1)
    return dim3 
    
def observation_to_state(obs: np.ndarray) -> State:
    obs = obs.reshape(9, -1)

    # Extract token row and attention masks
    row_0 = obs[0]
    masks = obs[1:6]  # rows 1–5
    row_6, row_7, row_8 = obs[6], obs[7], obs[8]
    
    raw_segments = []
    masks_i = []

    # Extract each segment using its attention mask
    for mask in masks:
        indices = np.where(mask == 1)[0]
        masks_i.append(indices[0])
        raw_segments.append(row_0[indices])
    input_, scratchpad, llm_input, llm_output, output = raw_segments
    
    cursor_i = np.where(row_6 == 1)[0][0]
    cursor_seg = np.searchsorted(masks_i, cursor_i, side="right") - 1
    cursor_i = cursor_i - masks_i[cursor_seg]
    
    cursor = (cursor_seg, cursor_i)
    
    highlight_i = np.where(row_7 == 1)[0][0]
    highlight_seg = np.searchsorted(masks_i, highlight_i, side="right") - 1
    highlight_i = highlight_i - masks_i[highlight_seg]
    
    if np.where(row_8 == 1)[0].size > 0:
        highlight_s = np.where(row_8 == 1)[0].max() + 1 # exclusive 
        highlight_s = highlight_s - masks_i[highlight_seg]
    else:
        highlight_s = highlight_i
    
    highlight = (highlight_seg, highlight_i, highlight_s)
    
    return {
        "input": input_,
        "scratchpad": scratchpad,
        "llm_input": llm_input,
        "llm_output": llm_output,
        "output": output,
        "cursor_pos": np.array(cursor, dtype=np.int32),
        "cursor_highlight": np.array(highlight, dtype=np.int32)
    }

@ENV_REGISTRY.register('scratchpad')
class ScratchpadEnv(BaseEnv):
    """
    Environment consists of input, output, scratchpad, cursor and llm 
    input/output. 
    
    For a model to understand all the text, some sort of pre-trained transformer 
    will be required for the observation->hidden-state. Then I'm hoping the rest 
    of the model can converge given this. 
    
    Checkpoints: 
    0: Tests to verify env works properly.
    I: MuZero to print foo bar foo bar repeatedly, from LLM which randomly gives 
    foo or bar. Can modify rewards to limit LLM calls. 
    II: Modify MuZero with a transformer observer. Test convergence with RL on a 
    small dataset of R1 reasoning traces. 
    
    """
    
    config = dict(
        env_id="scratchpad",
        render_mode=None,
    )
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def action_space(self):
        return self._action_space
        
    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space
        
    @property
    def legal_actions(self):
        legal_actions: list[int] = []
        
        if self._s['cursor_pos'][0] == 1:
            legal_actions.append(Action.TO_INPUT.value)
        if self._s['cursor_pos'][0] == 0:
            legal_actions.append(Action.TO_SCRATCHPAD.value)
        if self._s['cursor_pos'][1] > 0:
            legal_actions.append(Action.TO_LEFT.value)
        if self._s['cursor_pos'][0] == 0 and (self._s['cursor_pos'][1] < self.input_token_len):
            legal_actions.append(Action.TO_RIGHT.value)
        if self._s['cursor_pos'][0] == 1 and (self._s['cursor_pos'][1] < self.scratchpad_token_len):
            legal_actions.append(Action.TO_RIGHT.value)
        legal_actions.append(Action.START_HIGHLIGHT.value)
        if self._s['cursor_highlight'][0] == self._s['cursor_pos'][0]:
            legal_actions.append(Action.STOP_HIGHLIGHT.value)
        if self._s['cursor_pos'][0] == 1 and (self._s['cursor_highlight'][1] != self._s['cursor_highlight'][2]) and (self.scratchpad_token_len - self._s['cursor_pos'][1] >= self._s['cursor_highlight'][2] - self._s['cursor_highlight'][1]):
            legal_actions.append(Action.CLONE.value)
        if self._s['cursor_highlight'][0] == 1 and (self._s['cursor_highlight'][1] != self._s['cursor_highlight'][2]):
            legal_actions.append(Action.DELETE.value)
        if self._s['cursor_highlight'][1] != self._s['cursor_highlight'][2] and (self._s['cursor_highlight'][2] - self._s['cursor_highlight'][1] <= self.llm_input_token_len):
            legal_actions.append(Action.LLM_INPUT.value)
        if self._s['llm_input'][0] != END_OF_TEXT and (self._s['llm_output'][self.llm_output_token_len-1] == END_OF_TEXT): 
            legal_actions.append(Action.LLM_GENERATE.value)
        if self._s['llm_output'][0] != END_OF_TEXT: 
            legal_actions.append(Action.LLM_DELETE.value)
        if self._s['cursor_pos'][0] == 1:
            if self._s['llm_output'][self.llm_output_token_len-1] != END_OF_TEXT:
                output_len = self.llm_output_token_len
            else:
                output_len = np.where(self._s['llm_output'] == END_OF_TEXT)[0][0]
            if 0 < output_len and (output_len <= self.scratchpad_token_len - self._s['cursor_pos'][1]):
                legal_actions.append(Action.LLM_OUTPUT.value)
        if self._s['cursor_pos'][0] == 1 and (self._s['cursor_pos'][1] < self.scratchpad_token_len) and self._s['output'][self.output_token_len - 1] == END_OF_TEXT:
            legal_actions.append(Action.OUTPUT.value)
        
        return legal_actions
    
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        
        self.input_token_len = cfg.input_token_len
        self.scratchpad_token_len = cfg.scratchpad_token_len
        self.llm_input_token_len = cfg.llm_input_token_len
        self.llm_output_token_len = cfg.llm_output_token_len
        self.output_token_len = cfg.output_token_len
        self.max_episode_len = cfg.max_episode_len
        self.llm_model=cfg.llm_model # "test_01"
        self.evaluate_model=cfg.evaluate_model # "test_01"
        
        self._total_text_dim = self.input_token_len+1 + self.scratchpad_token_len+1 + self.llm_input_token_len+1 + self.llm_output_token_len+1 + self.output_token_len+1
        
        self._action_space = spaces.Discrete(len(Action))
        self._reward_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self._observation_space = spaces.Box(0, END_OF_TEXT, (self.input_token_len + self.output_token_len + self.scratchpad_token_len + 2 + 3 + self.llm_input_token_len + self.llm_output_token_len, ), dtype=np.float32)
        
    def reset(self) -> Observation:
        
        self._s: State = {
            'input': np.full(shape=(self.input_token_len), fill_value=END_OF_TEXT, dtype=np.float32),
            'scratchpad': np.full(shape=(self.scratchpad_token_len), fill_value=END_OF_TEXT, dtype=np.float32),
            'llm_input': np.full(shape=(self.llm_input_token_len), fill_value=END_OF_TEXT, dtype=np.float32),
            'llm_output': np.full(shape=(self.llm_output_token_len), fill_value=END_OF_TEXT, dtype=np.float32),
            'output': np.full(shape=(self.output_token_len), fill_value=END_OF_TEXT, dtype=np.float32),
            'cursor_pos': np.zeros((2), np.int32),
            'cursor_highlight': np.zeros((3), np.int32),
        }
        self.episode_length = 0
        self._state_value = self.get_state_value()
        
        if self.llm_model == "test_01":
            self.set_input([np.float32(np.random.randint(low=4, high=8))])
        
        action_mask = np.zeros(len(Action), 'int8')
        action_mask[self.legal_actions] = 1
        return { 'observation': state_to_observation(self._s), 'action_mask': action_mask, 'to_play': -1 }
    
    def set_input(self, tokens: Iterable[np.float32]):
        self._s['input'] = np.full(shape=(self.input_token_len), fill_value=END_OF_TEXT, dtype=np.float32)
        for i, t in enumerate(tokens):
            self._s['input'][i] = t
    def set_output(self, tokens: Iterable[np.float32]):
        self._s['output'] = np.full(shape=(self.output_token_len), fill_value=END_OF_TEXT, dtype=np.float32)
        for i, t in enumerate(tokens):
                self._s['output'][i] = t
        
    def step(self, action) -> BaseEnvTimestep:
        self.episode_length += 1
    
        # Check if the action is legal, otherwise choose a random legal action
        if action not in self.legal_actions:
            logging.warning(
                f"Illegal action: {action} ({Action(action).name}). Legal actions: {self.legal_actions}. "
                "Choosing a random action from legal actions."
            )
            action = np.random.choice(self.legal_actions)
        
        reward = 0 
        done = False
        
        if action == Action.TO_INPUT.value:
            self._s['cursor_pos'][0] = 0
            self._s['cursor_pos'][1] = 0
        if action == Action.TO_SCRATCHPAD.value: 
            self._s['cursor_pos'][0] = 1
            self._s['cursor_pos'][1] = 0
        if action == Action.TO_LEFT.value:
            self._s['cursor_pos'][1] -= 1
        if action == Action.TO_RIGHT.value:
            self._s['cursor_pos'][1] += 1
        if action == Action.START_HIGHLIGHT.value:
            self._s['cursor_highlight'][0] = self._s['cursor_pos'][0]
            self._s['cursor_highlight'][1] = self._s['cursor_pos'][1]
            self._s['cursor_highlight'][2] = self._s['cursor_pos'][1]
        if action == Action.STOP_HIGHLIGHT.value:
            if self._s['cursor_pos'][1] < self._s['cursor_highlight'][1]:
                self._s['cursor_highlight'][2] = self._s['cursor_highlight'][1]
                self._s['cursor_highlight'][1] = self._s['cursor_pos'][1]
            else:
                self._s['cursor_highlight'][2] = self._s['cursor_pos'][1]
        if action == Action.CLONE.value:
            if self._s['cursor_highlight'][0] == 0:
                src = self._s['input']
            else:
                src = self._s['scratchpad']
            src_len = self._s['cursor_highlight'][2] - self._s['cursor_highlight'][1]
            
            buffer = [src[self._s['cursor_highlight'][1] + i] for i in range(src_len)]
            for i in range(src_len):
                self._s['scratchpad'][self._s['cursor_pos'][1] + i] = buffer[i]
        if action == Action.DELETE.value:
            for i in range(self._s['cursor_highlight'][1], self._s['cursor_highlight'][2]):
                self._s['scratchpad'][i] = END_OF_TEXT
        if action == Action.LLM_INPUT.value:
            if self._s['cursor_highlight'][0] == 0:
                src = self._s['input']
            else:
                src = self._s['scratchpad']
            self._s['llm_input'] = np.full(shape=(self.llm_input_token_len), fill_value=END_OF_TEXT, dtype=np.float32)
            self._s['llm_output'] = np.full(shape=(self.llm_output_token_len), fill_value=END_OF_TEXT, dtype=np.float32)
            for i in range(self._s['cursor_highlight'][2] - self._s['cursor_highlight'][1]):
                self._s['llm_input'][i] = src[self._s['cursor_highlight'][1] + i]
        if action == Action.LLM_GENERATE.value:
            s = np.where(self._s['llm_output'] == END_OF_TEXT)[0][0]
            self._s['llm_output'][s] = self.generate_token()
        if action == Action.LLM_DELETE.value:
            if self._s['llm_output'][self.llm_output_token_len-1] != END_OF_TEXT:
                s = self.llm_output_token_len
            else:
                s = np.where(self._s['llm_output'] == END_OF_TEXT)[0][0]
            self._s['llm_output'][s-1] = END_OF_TEXT
        if action == Action.LLM_OUTPUT.value:
            for i in range(np.where(self._s['llm_output'] == END_OF_TEXT)[0][0]):
                self._s['scratchpad'][self._s['cursor_pos'][1] + i] = self._s['llm_output'][i]
        if action == Action.OUTPUT.value: 
            s = np.where(self._s['output'] == END_OF_TEXT)[0][0]
            self._s['output'][s] = self._s['scratchpad'][self._s['cursor_pos'][1]]
            done = (self._s['output'][s] == END_OF_TEXT) or (s == self.output_token_len - 1)
        
        done = done or self.episode_length >= self.max_episode_len
        
        prv_state_value = self._state_value
        self._state_value = self.get_state_value() - 0.005*self.episode_length
        
        reward = self._state_value - prv_state_value
        
        action_mask = np.zeros(len(Action), 'int8')
        action_mask[self.legal_actions] = 1
        
        obs = { 'observation': state_to_observation(self._s), 'action_mask': action_mask, 'to_play': -1 }
        
        info = {}
        if done:
            info['eval_episode_return'] = self._state_value
        
        return BaseEnvTimestep(obs, reward, done, info)
    
    def __repr__(self) -> str:
        return "LightZero ScratchPad Env"
    
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
        
    def close(self) -> None:
        pass
        
    def get_state_value(self) -> np.float32:
        # In MuZero, dynamics function g tries to predict true reward u given 
        # latent state s and action a. 
        # We design u to shape the model in giving us best output. The value 
        # of the world W is actually path agnostic - except for general penalty
        # for wasting time. 
        
        value = 0.0 
        if self.llm_model == "test_01":
            # Tier 4: Outputting sequence from scratchpad
            
            # Tier 3: Getting 0 and 1 into scratchpad 
            
            # Tier 2: Getting llm_generate to be legal
            
            # Tier 1: Getting valid highlighted range
            
            def evaluate_tier4():
                n_correct = 0
                n_incorrect = 0
                n_total_input = 0
                
                n = int(self._s['input'][0]) or 2
                answer = [i % 2 for i in range(n)] + [END_OF_TEXT for _ in range(self.llm_output_token_len - n)]
                for s, a in zip(self._s['output'], answer):
                    n_total_input += 1
                    if s == a:
                        n_correct += 1
                    else:
                        n_incorrect += 1
                    if s == END_OF_TEXT:
                        break
                        
                # Component rewards
                alpha = 0.5
                beta = 0.1
                gamma = 2.0
                delta = 0.5
                lambda_ = 0.1
                
                n_total_correct = n+1
                
                r_correct = (n_correct / n_total_correct) ** alpha
                r_incorrect = beta * ((1 - (n_correct / n_total_correct)) ** gamma) if n_incorrect > 0 else 0
                r_length = delta * np.exp(-lambda_ * abs(n_total_input - n_total_correct))
                
                progress = r_correct + r_incorrect + r_length
                if n_total_input == 1:
                    return 0
                return progress
            
            def evaluate_tier3():
                scratch = [False, False]
                llm = [False, False]
                for i in range(self.scratchpad_token_len):
                    if self._s['scratchpad'][i] == 0:
                        scratch[0] = True
                    if self._s['scratchpad'][i] == 1:
                        scratch[1] = True
                for i in range(self.llm_output_token_len):
                    if self._s['llm_output'][i] == 0:
                        llm[0] = True
                    if self._s['llm_output'][i] == 1:
                        llm[1] = True
                        
                progress = 0
                for i in range(2):
                    if scratch[i]:
                        progress += 0.5
                    elif llm[i]:
                        progress += 0.25
                return progress
            
            def evaluate_tier2():
                has_legal_generate = Action.LLM_GENERATE.value in self.legal_actions
                return 1 if has_legal_generate else 0
        
            def evaluate_tier1():
                has_valid_range = False
            
                if self._s['cursor_highlight'][0] == 0:
                    src = self._s['input']
                else:
                    src = self._s['scratchpad']
                src_len = self._s['cursor_highlight'][2] - self._s['cursor_highlight'][1]
                
                if src_len <= self.llm_input_token_len:
                    buffer = [src[self._s['cursor_highlight'][1] + i] for i in range(src_len)]
                    for c in buffer:
                        if c != END_OF_TEXT:
                            has_valid_range = True
                
                return 1 if has_valid_range else 0
            
            for i, e in enumerate([evaluate_tier4, evaluate_tier3, evaluate_tier2, evaluate_tier1]):
                progress = e()
                if progress > 0:
                    value = (3-i) + progress
                    break
                    
            return value
            
        if self.llm_model == "qwen":
            return 0
        
        logging.error("Could not find evaluation function: evaluate_output")
        quit()
        
    def generate_token(self):
        if self.llm_model == "test_01":
            return np.random.choice([0, 1])
        return END_OF_TEXT
