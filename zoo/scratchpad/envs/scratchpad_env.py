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
    input: np.ndarray
    output: np.ndarray
    scratchpad: np.ndarray
    cursor_pos: np.ndarray
    cursor_highlight: np.ndarray
    llm_input: np.ndarray
    llm_output: np.ndarray
    
class Observation(TypedDict):
    observation: np.ndarray
    action_mask: np.ndarray
    to_play: Literal[-1]

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
        self.output_token_len = cfg.output_token_len
        self.scratchpad_token_len = cfg.scratchpad_token_len
        self.llm_input_token_len = cfg.llm_input_token_len
        self.llm_output_token_len = cfg.llm_output_token_len
        self.llm_model=cfg.llm_model # "test_01"
        self.evaluate_model=cfg.evaluate_model # "test_01"
        
        self._action_space = spaces.Discrete(len(Action))
        self._reward_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self._observation_space = spaces.Box(0, END_OF_TEXT, (self.input_token_len + self.output_token_len + self.scratchpad_token_len + 2 + 3 + self.llm_input_token_len + self.llm_output_token_len, ), dtype=np.int32)
        
    def reset(self) -> Observation:
        
        self.episode_length = 0
        self._final_eval_reward = 0.0
        
        self._s: State = {
            'input': np.full(shape=(self.input_token_len), fill_value=END_OF_TEXT, dtype=np.int32),
            'output': np.full(shape=(self.output_token_len), fill_value=END_OF_TEXT, dtype=np.int32),
            'scratchpad': np.full(shape=(self.scratchpad_token_len), fill_value=END_OF_TEXT, dtype=np.int32),
            'cursor_pos': np.zeros((2), np.int32),
            'cursor_highlight': np.zeros((3), np.int32),
            'llm_input': np.full(shape=(self.llm_input_token_len), fill_value=END_OF_TEXT, dtype=np.int32),
            'llm_output': np.full(shape=(self.llm_output_token_len), fill_value=END_OF_TEXT, dtype=np.int32)
        }
        
        if self.llm_model == "test_01":
            self.set_input([np.int32(np.random.randint(low=2, high=self.output_token_len))])
        
        action_mask = np.zeros(len(Action), 'int8')
        action_mask[self.legal_actions] = 1
        return { 'observation': self.state_to_flat(self._s), 'action_mask': action_mask, 'to_play': -1 }
    
    def state_to_flat(self, state: State):
        return np.concatenate([v.flatten() for v in state.values()])
        
    def flat_to_state(self, flat: np.ndarray) -> State:
        """
        Reconstructs the original structured state dictionary from the flat array,
        using the `template` dictionary to infer shapes and keys.
        """
        new_state: State = {}
        offset = 0
        for key, value in self._s.items():
            size = np.prod(value.shape)
            reshaped = flat[offset:offset + size].reshape(value.shape)
            new_state[key] = reshaped.astype(value.dtype)
            offset += size
        return new_state
    
    def set_input(self, tokens: Iterable[np.int32]):
        self._s['input'] = np.full(shape=(self.input_token_len), fill_value=END_OF_TEXT, dtype=np.int32)
        for i, t in enumerate(tokens):
            self._s['input'][i] = t
    def set_output(self, tokens: Iterable[np.int32]):
        self._s['output'] = np.full(shape=(self.output_token_len), fill_value=END_OF_TEXT, dtype=np.int32)
        for i, t in enumerate(tokens):
                self._s['output'][i] = t
        
    def step(self, action) -> BaseEnvTimestep:
    
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
            
            buffer = [src[self._s['cursor_pos'][1] + i] for i in range(src_len)]
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
                
            self._s['llm_input'] = np.full(shape=(self.llm_input_token_len), fill_value=END_OF_TEXT, dtype=np.int32)
            self._s['llm_output'] = np.full(shape=(self.llm_output_token_len), fill_value=END_OF_TEXT, dtype=np.int32)
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
            reward = self.evaluate_output()
        
        self._final_eval_reward = reward
        
        action_mask = np.zeros(len(Action), 'int8')
        action_mask[self.legal_actions] = 1
        
        obs = { 'observation': self.state_to_flat(self._s), 'action_mask': action_mask, 'to_play': -1 }
        
        info = {}
        if done:
            info['eval_episode_return'] = self._final_eval_reward
        
        return BaseEnvTimestep(obs, reward, done, info)
    
    def __repr__(self) -> str:
        return "LightZero ScratchPad Env"
    
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
        
    def close(self) -> None:
        pass
        
    def evaluate_output(self) -> float:
        if self.llm_model == "test_01":
            reward = 0
            n = self._s['input'][0]
            if not n:
                return 0
            answer = [i % 2 for i in range(n)]
            
            for s, a in zip(self._s['output'], answer):
                if s == a:
                    reward += 1
                else:
                    break

            return reward / n
        if self.llm_model == "qwen":
            return 0
        
        logging.error("Could not find evaluation function: evaluate_output")
        quit()
        
    def generate_token(self):
        if self.llm_model == "test_01":
            return np.random.choice([0, 1])
        return END_OF_TEXT
