from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 25
update_per_collect = 50
batch_size = 256
max_env_step = int(2e5)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

scratchpad_muzero_config = dict(
    exp_name=f'data_muzero/scratchpad_muzero',
    env=dict(
        input_token_len=10,
        output_token_len=20,
        scratchpad_token_len=10,
        llm_input_token_len=4,
        llm_output_token_len=24,
        llm_model="test_01",
        evaluate_model="test_01",
        
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(10+20+10+4+24,),
            action_space_size=13,
            image_channel=1,
            # We use the small size model for tictactoe.
            num_res_blocks=1,
            num_channels=1,
            reward_head_hidden_channels=[8],
            value_head_hidden_channels=[8],
            policy_head_hidden_channels=[8],
            support_scale=1,
            reward_support_size=1,
            value_support_size=1,
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        cuda=True,
        action_type='varied_action_space',
        game_segment_length=9,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=9,
        num_unroll_steps=3,
        # NOTE：In board_games, we set discount_factor=1.
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e4),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
scratchpad_muzero_config = EasyDict(scratchpad_muzero_config)

scratchpad_muzero_create_config = dict(
    env=dict(
        type='scratchpad',
        import_names=['zoo.scratchpad.envs.scratchpad_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
scratchpad_muzero_create_config = EasyDict(scratchpad_muzero_create_config)

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([scratchpad_muzero_config, scratchpad_muzero_create_config], seed=0, model_path=scratchpad_muzero_config.policy.model_path, max_env_step=max_env_step)
