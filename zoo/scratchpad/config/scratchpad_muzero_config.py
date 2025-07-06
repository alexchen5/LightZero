from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
input_token_len=10
scratchpad_token_len=10
llm_input_token_len=4
llm_output_token_len=24
output_token_len=20
max_episode_len=40

collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 200
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0.

model_path = None
model_path = "/scratch/pawsey1151/alexchen5/LightZero/data_muzero/scratchpad_muzero_250704_204611/ckpt/iteration_100000.pth.tar"

# =========== for debug ===========
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 2
# update_per_collect = 2
# batch_size = 2

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
total_text_dim = input_token_len+1+scratchpad_token_len+1+llm_input_token_len+1+llm_output_token_len+1+output_token_len+1

scratchpad_muzero_config = dict(
    exp_name=f'data_muzero/scratchpad_muzero',
    env=dict(
        input_token_len=input_token_len,
        scratchpad_token_len=scratchpad_token_len,
        llm_input_token_len=llm_input_token_len,
        llm_output_token_len=llm_output_token_len,
        output_token_len=output_token_len,
        max_episode_len=max_episode_len,
        llm_model="test_01",
        evaluate_model="test_01",
        
        observation_shape=(1, 9, total_text_dim),
        obs_type='dict_encoded_board',
        raw_reward_type='raw',  # 'merged_tiles_plus_log_max_tile_num'
        reward_normalize=False,
        reward_norm_scale=100,
        
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(1, 9, total_text_dim),
            action_space_size=13,
            image_channel=1,
            # # We use the small size model for tictactoe.
            # num_res_blocks=1,
            # num_channels=1,
            # reward_head_hidden_channels=[8],
            # value_head_hidden_channels=[8],
            # policy_head_hidden_channels=[8],
            # support_scale=10,
            # reward_support_size=21,
            # value_support_size=21,
            self_supervised_learning_loss=True,
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=model_path,
        cuda=True,
        # env_type="not_board_games",
        action_type='varied_action_space',
        game_segment_length=max_episode_len,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        td_steps=10,
        discount_factor=0.999,
        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=int(1e5),
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=3e-3,
        # (float) Weight decay for training policy network.
        weight_decay=1e-4,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,  # default is 0
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e4),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
scratchpad_muzero_config = EasyDict(scratchpad_muzero_config)
main_config = scratchpad_muzero_config

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
create_config = scratchpad_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([scratchpad_muzero_config, scratchpad_muzero_create_config], seed=0, model_path=scratchpad_muzero_config.policy.model_path, max_env_step=max_env_step)
