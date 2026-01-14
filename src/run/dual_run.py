import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

# 引入必要的组件
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from smac.env import StarCraft2Env

def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                     indent=4,
                                     width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs",args.env_args['map_name'])
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    # 【改动】：调用新的 run_dual_sequential 函数
    run_dual_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)

# 评测函数，进行一定轮次的测试（保持不变）
def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

# 【核心改动】：双 Learner 顺序训练的主循环
def run_dual_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }

    # 添加 VAE 所需的 scheme
    if getattr(args, "use_critical_agent_obs", False):
        scheme["critical_id"] = {"vshape": (1,), "dtype": th.long}
        scheme["critical_id_active"] = {"vshape": (1,), "dtype": th.long} 

    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    # 经验回放池
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # =================== 【重点修改：初始化两个 Learner】 ===================
    # 1. 初始化 RL Learner (用于训练 Agent, Mixer 等)
    # 使用 args.learner 指定的类
    logger.console_logger.info(f"Initializing RL Learner: {args.learner}")
    rl_learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    if args.use_cuda:
        rl_learner.cuda()

    # 2. 初始化 WM Learner (用于训练 World Model / VAE)
    # 我们假设配置文件中有一个新的参数 args.wm_learner，如果没有，则默认使用 'dvd_wm_learner' 或其他
    wm_learner_name = getattr(args, "wm_learner", "dvd_wm_learner") 
    logger.console_logger.info(f"Initializing WM Learner: {wm_learner_name}")
    
    # 注意：这里我们传入同一个 mac，因为 mac 可能持有 agent，而 agent 可能是 WM 的一部分或需要共享 embedding
    # 如果 wm_learner 不需要 mac，它在内部可能不会使用
    wm_learner = le_REGISTRY[wm_learner_name](mac, buffer.scheme, logger, args)
    if args.use_cuda:
        wm_learner.cuda()

    # 如果 RL Learner 需要访问 WM Learner 的内部模型（例如 encoder），可以在这里进行链接
    # 假设 RL learner 有一个 set_wm_model 方法
    if hasattr(rl_learner, "set_wm_model"):
        logger.console_logger.info("Linking WM model to RL Learner...")
        rl_learner.set_wm_model(wm_learner)
    # ======================================================================

    # "单一模型引用" 链接逻辑 (适配 VAE)
    # 注意：现在我们链接的是 wm_learner 中的 vae_model
    if getattr(args, "use_online_vae_training", False) and getattr(args, "use_critical_agent_obs", False):
        if hasattr(wm_learner, "vae_model") and hasattr(runner, "set_vae_model"):
            logger.console_logger.info("Linking WM Learner's VAE model to Runner...")
            runner.set_vae_model(wm_learner.vae_model) 
        else:
            logger.console_logger.error("Failed to link VAE: WM Learner or Runner mismatch.")

    # 加载 Checkpoint
    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0
        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            timestep_to_load = max(timesteps)
        else:
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
        logger.console_logger.info("Loading model from {}".format(model_path))
        
        # 【修改】：分别为两个 Learner 加载模型
        # 注意：这里假设保存的时候是分别保存的，或者是在同一个目录下
        # 通常建议 learner.load_models 会加载自己负责的部分
        rl_learner.load_models(model_path)
        wm_learner.load_models(model_path)
        
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # =================== 【准备 Batch Size 参数】 ===================
    # 获取 WM 的 batch size，如果配置没写，默认和 RL 一样，但通常我们希望它更大
    wm_batch_size = getattr(args, "wm_batch_size", args.batch_size) 
    rl_batch_size = args.batch_size
    
    # 确保 WM batch size 至少和 RL batch size 一样大
    if wm_batch_size < rl_batch_size:
        logger.console_logger.warning(f"wm_batch_size ({wm_batch_size}) < rl_batch_size ({rl_batch_size}). Forcing wm_batch_size to equal rl_batch_size.")
        wm_batch_size = rl_batch_size
    # =============================================================

    while runner.t_env <= args.t_max:

        # 1. 运行环境，收集数据
        with th.no_grad():
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

        # 2. 检查 Buffer 是否足够进行 WM 的训练（因为 WM 需要更大的 Batch）
        if buffer.can_sample(wm_batch_size):
            
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            # =================== 【核心改动：分层采样与更新】 ===================
            # 步骤 A: 从 Buffer 中采样大 Batch (用于 WM)
            wm_sample = buffer.sample(wm_batch_size)

            # 数据对齐/截断 (Truncate to max_t_filled)
            max_ep_t = wm_sample.max_t_filled()
            wm_sample = wm_sample[:, :max_ep_t]

            if wm_sample.device != args.device:
                wm_sample.to(args.device)

            # 步骤 B: 构造 RL 的小 Batch (通过切片操作)
            # 我们直接取 wm_sample 的前 rl_batch_size 个样本
            # 这样做既保证了数据同源，又避免了重复采样和传输的开销
            rl_sample = wm_sample[:rl_batch_size]

            # 步骤 C: 更新 WM 权重
            # 注意：在 wm_learner.train 中，应当计算 loss 并执行反向传播
            wm_learner.train(wm_sample, runner.t_env, episode)

            # 步骤 D: 更新 RL 权重
            # 注意：如果 RL 需要用到 WM 的输出（如 hidden state 或 embedding），
            # 必须确保 wm_learner 更新后的参数被用于本次 rl_learner 的计算（如果两者共享 encoder）
            # 或者 rl_learner 内部通过引用调用 wm_learner 的 forward
            rl_learner.train(rl_sample, runner.t_env, episode)

            # 释放显存
            del wm_sample
            # rl_sample 只是一个 view，删除 wm_sample 应该也会处理引用计数，但显式删除好习惯
            del rl_sample
            # =================================================================

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            
            test_batch = None
            for i in range(n_test_runs):
                batch = runner.run(test_mode=True)
                if i == 0:
                    test_batch = batch

            # 【修改】：调用 WM Learner 计算 WM 相关的评测指标
            if test_batch is not None and hasattr(wm_learner, "evaluate_world_model"):
                wm_learner.evaluate_world_model(test_batch, runner.t_env)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # 【修改】：保存两个 Learner 的模型
            rl_learner.save_models(save_path)
            # 为 WM 模型增加一个子文件夹或前缀，防止文件名冲突（如果 save_models 内部没有处理）
            # 这里假设它们内部处理好了文件名，或者我们简单地让它们都保存到同一个路径
            wm_learner.save_models(save_path) 

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config