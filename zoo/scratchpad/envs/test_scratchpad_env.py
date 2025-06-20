import pytest
from easydict import EasyDict
import numpy as np

from .scratchpad_env import ScratchpadEnv, END_OF_TEXT, Action, observation_to_state as s

input_token_len=10
scratchpad_token_len=10
llm_input_token_len=4
llm_output_token_len=24
output_token_len=20

total_text_dim = input_token_len+1+scratchpad_token_len+1+llm_input_token_len+1+llm_output_token_len+1+output_token_len+1

# pytest zoo/scratchpad/envs/test_scratchpad_env.py

@pytest.mark.unittest
class TestScratchpad:
    def setup_method(self, method) -> None:
        cfg = EasyDict({
            'env_id': "scratchpad",
            'input_token_len': input_token_len,
            'scratchpad_token_len': scratchpad_token_len,
            'llm_input_token_len': llm_input_token_len,
            'llm_output_token_len': llm_output_token_len,
            'output_token_len': output_token_len,
            'llm_model': "test_01",
            'evaluate_model': "test_01",
        })
        self.env = ScratchpadEnv(cfg)
    
    def test_init(self):
        assert isinstance(self.env, ScratchpadEnv), "Environment is not an instance of ScratchpadEnv"
        
    def test_reset(self):
        obs = self.env.reset()
        assert obs['observation'].shape == (9, total_text_dim)
        assert obs['observation'][0][input_token_len] == END_OF_TEXT
        assert obs['observation'][1][0] == 1
        assert obs['observation'][1][input_token_len] == 0
        
    def test_cursor_to_input(self):
        self.env.reset()
        self.env.step(Action.TO_RIGHT.value)
        obs, reward, done, info = self.env.step(Action.TO_SCRATCHPAD.value)
        obs, reward, done, info = self.env.step(Action.TO_INPUT.value)
        assert s(obs['observation'])['cursor_pos'][0] == 0
        assert s(obs['observation'])['cursor_pos'][1] == 0
        
    def test_cursor_to_input_illegal(self):
        self.env.reset()
        assert Action.TO_INPUT.value not in self.env.legal_actions 
        
    def test_cursor_to_scratchpad(self):
        self.env.reset()
        self.env.step(Action.TO_RIGHT.value)
        obs, reward, done, info = self.env.step(Action.TO_SCRATCHPAD.value)
        assert s(obs['observation'])['cursor_pos'][0] == 1
        assert s(obs['observation'])['cursor_pos'][1] == 0
        
    def test_cursor_to_scratchpad_illegal(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(Action.TO_SCRATCHPAD.value)
        assert Action.TO_SCRATCHPAD.value not in self.env.legal_actions 
    
    def test_cursor_left(self):
        self.env.reset()
        self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.TO_RIGHT.value)
        obs1, reward, done, info = self.env.step(Action.TO_LEFT.value)
        obs2, reward, done, info = self.env.step(Action.TO_LEFT.value)
        assert s(obs1['observation'])['cursor_pos'][1] == 1
        assert s(obs2['observation'])['cursor_pos'][1] == 0
        
    def test_cursor_left_illegal(self):
        self.env.reset()
        assert Action.TO_LEFT.value not in self.env.legal_actions 
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        assert Action.TO_LEFT.value not in self.env.legal_actions 
        
    def test_cursor_right(self):
        self.env.reset()
        obs1, reward, done, info = self.env.step(Action.TO_RIGHT.value)
        obs2, reward, done, info = self.env.step(Action.TO_RIGHT.value)
        assert s(obs1['observation'])['cursor_pos'][1] == 1
        assert s(obs2['observation'])['cursor_pos'][1] == 2
        
    def test_cursor_right_illegal(self):
        self.env.reset()
        for i in range(self.env.input_token_len-1):
            self.env.step(Action.TO_RIGHT.value)
        assert Action.TO_RIGHT.value in self.env.legal_actions 
        self.env.step(Action.TO_RIGHT.value)
        assert Action.TO_RIGHT.value not in self.env.legal_actions 
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        for i in range(self.env.scratchpad_token_len-1):
            self.env.step(Action.TO_RIGHT.value)
        assert Action.TO_RIGHT.value in self.env.legal_actions 
        self.env.step(Action.TO_RIGHT.value)
        assert Action.TO_RIGHT.value not in self.env.legal_actions 
    
    def test_highlight(self):
        self.env.reset()
        for i in range(10):
            self.env.step(Action.TO_RIGHT.value)
        obs1, reward, done, info = self.env.step(Action.START_HIGHLIGHT.value)
        
        for i in range(3):
            self.env.step(Action.TO_LEFT.value)
        obs2, reward, done, info = self.env.step(Action.STOP_HIGHLIGHT.value)
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        obs3, reward, done, info = self.env.step(Action.START_HIGHLIGHT.value)
        
        assert s(obs1['observation'])['cursor_highlight'][0] == 0
        assert s(obs1['observation'])['cursor_highlight'][1] == 10
        assert s(obs1['observation'])['cursor_highlight'][2] == 10
        
        assert s(obs2['observation'])['cursor_highlight'][0] == 0
        assert s(obs2['observation'])['cursor_highlight'][1] == 7
        assert s(obs2['observation'])['cursor_highlight'][2] == 10
        
        assert s(obs3['observation'])['cursor_highlight'][0] == 1
        assert s(obs3['observation'])['cursor_highlight'][1] == 0
        assert s(obs3['observation'])['cursor_highlight'][2] == 0
        
    def test_highlight_illegal(self):
        self.env.reset()
        
        self.env.step(Action.START_HIGHLIGHT.value)
        self.env.step(Action.TO_SCRATCHPAD.value)
        assert Action.STOP_HIGHLIGHT.value not in self.env.legal_actions 
    
    def test_clone(self):
        self.env.reset()
        
        self.env.set_input([np.int32(v) for v in range(self.env.input_token_len)])
        
        for i in range(self.env.input_token_len):
            self.env.step(Action.TO_RIGHT.value)
            self.env.step(Action.STOP_HIGHLIGHT.value)
        self.env.step(Action.TO_SCRATCHPAD.value)
        obs1, reward, done, info = self.env.step(Action.CLONE.value)
        
        for i in range(self.env.input_token_len):
            assert s(obs1['observation'])['scratchpad'][i] == i 
        
    def test_clone_illegal(self):
        self.env.reset()
        
        assert Action.CLONE.value not in self.env.legal_actions 
        self.env.step(Action.TO_SCRATCHPAD.value)
        assert Action.CLONE.value not in self.env.legal_actions 
        
        self.env.step(Action.START_HIGHLIGHT.value)
        self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.STOP_HIGHLIGHT.value)
        assert Action.CLONE.value in self.env.legal_actions 
        
        self.env.step(Action.TO_INPUT.value)
        assert Action.CLONE.value not in self.env.legal_actions 
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        assert Action.CLONE.value in self.env.legal_actions 
        
        for i in range(self.env.scratchpad_token_len-1):
            self.env.step(Action.TO_RIGHT.value)
        assert Action.CLONE.value in self.env.legal_actions 
        self.env.step(Action.TO_RIGHT.value)
        assert Action.CLONE.value not in self.env.legal_actions 
    
    def test_delete(self):
        self.env.reset()
        
        for i in range(self.env.scratchpad_token_len):
            self.env._s['scratchpad'][i] = i+1 
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.START_HIGHLIGHT.value)
        for i in range(self.env.scratchpad_token_len - 2):
            self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.STOP_HIGHLIGHT.value)
        
        obs1, reward, done, info = self.env.step(Action.DELETE.value)
        for i in range(1, self.env.scratchpad_token_len - 2):
            assert s(obs1['observation'])['scratchpad'][i] == END_OF_TEXT
            
        assert s(obs1['observation'])['scratchpad'][0] == 1
        assert s(obs1['observation'])['scratchpad'][self.env.scratchpad_token_len - 1] == self.env.scratchpad_token_len
        
    def test_delete_illegal(self):
        self.env.reset()
        
        assert Action.DELETE.value not in self.env.legal_actions 
        
        self.env.step(Action.START_HIGHLIGHT.value)
        self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.STOP_HIGHLIGHT.value)
        assert Action.DELETE.value not in self.env.legal_actions 
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        self.env.step(Action.START_HIGHLIGHT.value)
        assert Action.DELETE.value not in self.env.legal_actions 
        
        self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.STOP_HIGHLIGHT.value)
        assert Action.DELETE.value in self.env.legal_actions 
        
        self.env.step(Action.TO_INPUT.value)
        assert Action.DELETE.value in self.env.legal_actions 
    
    def test_llm_input(self):
        self.env.reset()
        
        self.env.set_input([np.int32(v+1) for v in range(self.env.input_token_len)])
        self.env.step(Action.START_HIGHLIGHT.value)
        for i in range(self.env.llm_input_token_len):
            self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.STOP_HIGHLIGHT.value)
        
        obs1, reward, done, info = self.env.step(Action.LLM_INPUT.value)
        self.env.step(Action.LLM_GENERATE.value)
        
        self.env.step(Action.TO_LEFT.value)
        self.env.step(Action.STOP_HIGHLIGHT.value)
        obs2, reward, done, info = self.env.step(Action.LLM_INPUT.value)
        
        for i in range(self.env.llm_input_token_len-1):
            assert s(obs1['observation'])['llm_input'][i] == i+1
            assert s(obs2['observation'])['llm_input'][i] == i+1
        assert s(obs1['observation'])['llm_input'][self.env.llm_input_token_len-1] == self.env.llm_input_token_len-1+1
        assert s(obs2['observation'])['llm_input'][self.env.llm_input_token_len-1] == END_OF_TEXT
        
        assert s(obs2['observation'])['llm_output'][0] == END_OF_TEXT
        
    def test_llm_input_illegal(self):
        self.env.reset()
        
        assert Action.LLM_INPUT.value not in self.env.legal_actions 
        
        self.env.step(Action.START_HIGHLIGHT.value)
        for i in range(self.env.llm_input_token_len):
            self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.STOP_HIGHLIGHT.value)
        assert Action.LLM_INPUT.value in self.env.legal_actions 
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        self.env.step(Action.START_HIGHLIGHT.value)
        for i in range(self.env.llm_input_token_len+1):
            self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.STOP_HIGHLIGHT.value)
        assert Action.LLM_INPUT.value not in self.env.legal_actions 
        
    def test_llm_generate(self):
        self.env.reset()
        
        self.env.set_input([np.int32(1)])
        self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.STOP_HIGHLIGHT.value)
        self.env.step(Action.LLM_INPUT.value)
        
        obs1, reward, done, info = self.env.step(Action.LLM_GENERATE.value)
        assert s(obs1['observation'])['llm_output'][0] == 0 or s(obs1['observation'])['llm_output'][0] == 1
        assert s(obs1['observation'])['llm_output'][1] == END_OF_TEXT
        
    def test_llm_generate_illegal(self):
        self.env.reset()
        assert Action.LLM_GENERATE.value not in self.env.legal_actions 
        
        self.env.set_input([np.int32(1)])
        self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.STOP_HIGHLIGHT.value)
        self.env.step(Action.LLM_INPUT.value)
        
        for i in range(self.env.llm_output_token_len):
            assert Action.LLM_GENERATE.value in self.env.legal_actions 
            self.env.step(Action.LLM_GENERATE.value)
        assert Action.LLM_GENERATE.value not in self.env.legal_actions 
        
    def test_llm_delete(self):
        self.env.reset()
        
        self.env.set_input([np.int32(1)])
        self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.STOP_HIGHLIGHT.value)
        self.env.step(Action.LLM_INPUT.value)
        
        self.env.step(Action.LLM_GENERATE.value)
        obs1, reward, done, info = self.env.step(Action.LLM_DELETE.value)
        
        assert s(obs1['observation'])['llm_output'][0] == END_OF_TEXT
        
    def test_llm_delete_illegal(self):
        self.env.reset()
        assert Action.LLM_DELETE.value not in self.env.legal_actions 
        
        self.env.set_input([np.int32(1)])
        self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.STOP_HIGHLIGHT.value)
        self.env.step(Action.LLM_INPUT.value)
        
        self.env.step(Action.LLM_GENERATE.value)
        self.env.step(Action.LLM_DELETE.value)
        
        assert Action.LLM_DELETE.value not in self.env.legal_actions 
        
    def test_llm_output(self):
        self.env.reset()
        
        self.env.set_input([np.int32(1)])
        self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.STOP_HIGHLIGHT.value)
        self.env.step(Action.LLM_INPUT.value)
        
        self.env.step(Action.LLM_GENERATE.value)
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        self.env.step(Action.TO_RIGHT.value)
        obs1, reward, done, info = self.env.step(Action.LLM_OUTPUT.value)
        
        assert s(obs1['observation'])['scratchpad'][0] == END_OF_TEXT
        assert s(obs1['observation'])['scratchpad'][1] == s(obs1['observation'])['llm_output'][0]
        
    def test_llm_output_illegal(self):
        self.env.reset()
        
        assert Action.LLM_OUTPUT.value not in self.env.legal_actions 
        
        self.env.set_input([np.int32(1)])
        self.env.step(Action.TO_RIGHT.value)
        self.env.step(Action.STOP_HIGHLIGHT.value)
        self.env.step(Action.LLM_INPUT.value)
        
        self.env.step(Action.LLM_GENERATE.value)
        
        assert Action.LLM_OUTPUT.value not in self.env.legal_actions 
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        assert Action.LLM_OUTPUT.value in self.env.legal_actions 
        self.env.step(Action.TO_RIGHT.value)
        for i in range(self.env.scratchpad_token_len-1):
            assert Action.LLM_OUTPUT.value in self.env.legal_actions 
            self.env.step(Action.LLM_GENERATE.value)
        
        assert Action.LLM_OUTPUT.value not in self.env.legal_actions 
        
    def test_output(self):
        self.env.reset()
        
        self.env._s['scratchpad'][0] = 1
        self.env._s['scratchpad'][self.env.scratchpad_token_len-1] = 2
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        obs1, reward, done1, info = self.env.step(Action.OUTPUT.value)
        for i in range(self.env.scratchpad_token_len-1):
            self.env.step(Action.TO_RIGHT.value)
            
        obs2, reward, done2, info = self.env.step(Action.OUTPUT.value)
        
        self.env.step(Action.TO_LEFT.value)
        obs3, reward, done3, info = self.env.step(Action.OUTPUT.value)
        
        assert s(obs1['observation'])['output'][0] == 1
        assert s(obs1['observation'])['output'][1] == END_OF_TEXT
        
        assert s(obs2['observation'])['output'][0] == 1
        assert s(obs2['observation'])['output'][1] == 2
        assert s(obs2['observation'])['output'][2] == END_OF_TEXT
        
        assert s(obs3['observation'])['output'][0] == 1
        assert s(obs3['observation'])['output'][1] == 2
        assert s(obs3['observation'])['output'][2] == END_OF_TEXT
        
        assert done2 == False
        assert done3 == True
        
    def test_output_illegal(self):
        self.env.reset()
        
        assert Action.OUTPUT.value not in self.env.legal_actions 
        self.env.step(Action.TO_RIGHT.value)
        assert Action.OUTPUT.value not in self.env.legal_actions 
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        assert Action.OUTPUT.value in self.env.legal_actions 
        
        for i in range(self.env.scratchpad_token_len):
            assert Action.OUTPUT.value in self.env.legal_actions 
            self.env.step(Action.TO_RIGHT.value)
        
        assert Action.OUTPUT.value not in self.env.legal_actions 
        assert Action.TO_RIGHT.value not in self.env.legal_actions 
    
    def test_output_full(self):
        self.env.reset()
        self.env._s['scratchpad'][0] = 1
        self.env.step(Action.TO_SCRATCHPAD.value)
        
        for i in range(self.env.output_token_len-1):
            assert Action.OUTPUT.value in self.env.legal_actions 
            obs1, reward, done1, info = self.env.step(Action.OUTPUT.value)
        
        obs2, reward, done2, info = self.env.step(Action.OUTPUT.value)
        
        assert s(obs1['observation'])['output'][self.env.output_token_len-2] == 1
        assert s(obs1['observation'])['output'][self.env.output_token_len-1] == END_OF_TEXT
        
        assert Action.OUTPUT.value not in self.env.legal_actions 
        assert s(obs2['observation'])['output'][self.env.output_token_len-1] == 1
        assert done2 == True
    
    def test_01_reward_0(self):
        self.env.reset()
        self.env.set_input([np.int32(0)])
        self.env.set_output([])
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        obs1, reward, done, info = self.env.step(Action.OUTPUT.value)
        assert reward == 0
        
    def test_01_reward_1(self):
        self.env.reset()
        self.env.set_input([np.int32(1)])
        self.env.set_output([np.int32(0)])
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        obs1, reward, done, info = self.env.step(Action.OUTPUT.value)
        assert reward == 1
        
    def test_01_reward_n(self):
        self.env.reset()
        self.env.set_input([np.int32(5)])
        self.env.set_output([np.int32(0), np.int32(1), np.int32(0), np.int32(1), np.int32(0)])
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        obs1, reward, done, info = self.env.step(Action.OUTPUT.value)
        assert reward == 5 / 5
    
    def test_01_reward_over(self):
        self.env.reset()
        self.env.set_input([np.int32(1)])
        self.env.set_output([np.int32(0), np.int32(1)])
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        obs1, reward, done, info = self.env.step(Action.OUTPUT.value)
        assert reward == 1
        
    def test_01_reward_truncate(self):
        self.env.reset()
        self.env.set_input([np.int32(5)])
        self.env.set_output([np.int32(0), np.int32(1), np.int32(1), np.int32(1), np.int32(0)])
        
        self.env.step(Action.TO_SCRATCHPAD.value)
        obs1, reward, done, info = self.env.step(Action.OUTPUT.value)
        assert reward == 2 / 5
        
        