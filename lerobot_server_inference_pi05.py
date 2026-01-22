"""
lerobot_server_inference_pi05.py
LeRobot 官方风格实机推理服务
特点：使用官方 Processor 流水线，自动处理 Resize、Prompt 构造和归一化
"""
import zmq
import torch
import pickle
import numpy as np
import cv2
import json
import os
from peft import PeftModel
from huggingface_hub import hf_hub_download

# --- LeRobot 核心组件 ---
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
import pprint

# ================= 配置区域 =================
ADAPTER_REPO_ID = "moriis/pi05_piper_third_v3"
BASE_MODEL_ID = "lerobot/pi05_base"

PORT = 5555
DEVICE = "cuda"
VISUALIZE = False

# 硬件名称 -> 模型输入名称 的映射
CAMERA_MAPPING = {
    "pikaGripperDepthCamera":   "observation.images.pikaGripperDepthCamera",
    "pikaGripperFisheyeCamera": "observation.images.pikaGripperFisheyeCamera",
    "pikaThirdPersonCamera":    "observation.images.pikaThirdPersonCamera",
}
STATS_PATH = "../lerobot_dataset_third_v2/meta/stats.json"
# ===========================================

def load_dataset_stats(stats_path, device="cpu"):
    """加载统计文件并转换为 Tensor (工厂函数需要)"""
    print(f"Loading stats from: {stats_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    with open(stats_path, 'r') as f:
        stats_dict = json.load(f)
    
    def convert(item):
        if isinstance(item, dict): return {k: convert(v) for k, v in item.items()}
        if isinstance(item, list): return torch.tensor(item, dtype=torch.float32, device=device)
        return item
    return convert(stats_dict)

def get_clean_config(repo_id):
    """手动清洗配置，修复 LeRobot 加载报错"""
    print(f"Downloading config from {repo_id}...")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    # 1. 移除不兼容字段
    for k in ["type", "transformers_version", "_commit_hash", "peft", "use_peft"]:
        if k in cfg_dict: del cfg_dict[k]

    # 2. 修复 Enum
    if "normalization_mapping" in cfg_dict:
        norm_map = {}
        for k, v in cfg_dict["normalization_mapping"].items():
            norm_map[k] = NormalizationMode[v] if isinstance(v, str) else v
        cfg_dict["normalization_mapping"] = norm_map

    # 3. 修复 Feature 类型
    def fix_features(feats):
        new_feats = {}
        for k, v in feats.items():
            if isinstance(v, dict):
                if "type" in v and isinstance(v["type"], str): v["type"] = FeatureType[v["type"]]
                new_feats[k] = PolicyFeature(**v)
            else:
                new_feats[k] = v
        return new_feats
    
    cfg_dict["input_features"] = fix_features(cfg_dict.get("input_features", {}))
    cfg_dict["output_features"] = fix_features(cfg_dict.get("output_features", {}))
    
    return PI05Config(**cfg_dict)

def main():
    # ---------------- 初始化模型与流水线 ----------------
    try:
        # 1. 准备配置和统计数据
        user_config = get_clean_config(ADAPTER_REPO_ID)
        pprint.pprint(user_config)
        dataset_stats = load_dataset_stats(STATS_PATH, device=DEVICE)

        # 2. 加载 Policy (Base + Adapter)
        print(f"Loading Policy (Base: {BASE_MODEL_ID})...")
        policy = PI05Policy.from_pretrained(BASE_MODEL_ID, config=user_config)
        print(f"Loading Adapter ({ADAPTER_REPO_ID})...")
        policy = PeftModel.from_pretrained(policy, ADAPTER_REPO_ID)
        policy.to(DEVICE)
        policy.eval()

        # 3. 构建官方 Processor 流水线
        # 这会创建类似 lerobot_eval.py 中的 env_preprocessor/preprocessor
        print("Building Official Pre/Post Processors...")
        preprocessor, postprocessor = make_pi05_pre_post_processors(
            config=user_config,
            dataset_stats=dataset_stats
        )
        # 注意：make_pi05_pre_post_processors 创建的 pipeline 默认在 CPU
        # 我们需要手动将内部步骤的 device 设置好，或者在运行时由 DeviceProcessorStep 处理
        # 官方代码中包含 DeviceProcessorStep(device=config.device)，所以它会自动把数据挪到 GPU
        
        print("✅ System Ready!")

    except Exception as e:
        print(f"❌ Init failed: {e}")
        import traceback; traceback.print_exc()
        return

    # ---------------- ZMQ 服务循环 ----------------
    if VISUALIZE:
        init_rerun(session_name="Pi05_Real_Inference", ip="127.0.0.1", port=9876)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    print(f"🎧 Listening on {PORT}...")

    while True:
        try:
            msg = socket.recv()
            payload = pickle.loads(msg)
            
            # === 1. 构建原始观测字典 (Raw Observation) ===
            # 这里只需要把数据转成 Tensor 格式，无需 Resize，无需 Normalize，无需 Batch Dim
            raw_observation = {}
            payload_images = payload.get('images', {})
            task_text = payload.get("text", "Grab the carrot and put it into the box.")
            
            # --- 图像处理 ---
            # 目标：[C, H, W], float32, 0-1
            imgs_ok = True
            for hw_key, model_key in CAMERA_MAPPING.items():
                if hw_key in payload_images and payload_images[hw_key] is not None:
                    nparr = np.frombuffer(payload_images[hw_key], np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # 转 Tensor: [H, W, C] -> [C, H, W]
                        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                        raw_observation[model_key] = img_tensor
                    else:
                        imgs_ok = False
                else:
                    imgs_ok = False
            
            if not imgs_ok:
                print("⚠️ Missing images, skipping...")
                socket.send(pickle.dumps(None))
                continue

            # --- 状态处理 ---
            # 目标：[D], float32
            joints = payload.get('joint_state', [])
            if len(joints) != 7:
                socket.send(pickle.dumps(None)); continue
            
            raw_observation["observation.state"] = torch.tensor(joints, dtype=torch.float32)

            # --- 任务处理 ---
            # 官方 Processor 需要在 complementary_data 中找到任务，或者我们直接通过 hack 方式传入
            # Pi05PrepareStateTokenizerProcessorStep 默认从 transition 字典里找 task
            # 我们构造一个包含 'task' 的字典，这符合 LeRobot 数据集读取时的格式
            input_batch = raw_observation
            # 注意：Pi05 的 processor 比较特殊，它通过 task_key="task" 来读文本
            # 我们直接把 task 放入 input_batch，因为 ProcessorStep 会遍历整个 dict
            # 但更标准的做法是遵循 processor_pi05.py 的 transition 结构
            # 简单起见，直接赋值：
            input_batch["task"] = [task_text] # 注意这里用列表，因为 AddBatchDimension 会处理 Tensor，但字符串通常是列表处理

            # === 2. 执行预处理流水线 (Official Pipeline) ===
            # 这一步会自动：
            # 1. AddBatchDimension: [C,H,W] -> [1,C,H,W]
            # 2. Normalize State: 使用 stats.json
            # 3. Prepare Prompt: 拼接 "Task: ..., State: ..." 并 Tokenize
            # 4. Move to Device: 转到 GPU
            
            # 修正：AddBatchDimensionProcessorStep 可能会因为 "task" 是 list 而报错或者忽略
            # 如果 preprocessor 第一步是 AddBatchDimension，它期望输入是无 batch 的 Tensor
            # 我们手动处理一下 task 的 batch 问题
            
            with torch.no_grad():
                # 调用 preprocessor
                # 警告：make_pi05_pre_post_processors 返回的 processor 期望字典结构包含 'observation.state' 等
                batch = preprocessor(input_batch)
                # print("Processed Batch Keys:", batch.keys())
                
                
                # === 3. 执行策略 (Policy Inference) ===
                # policy.select_action 会调用 predict_action_chunk
                # 内部会自动调用 resize_with_pad_torch (产生 -3.0 黑边)
                action = policy.select_action(batch)
                
                # === 4. 执行后处理 (Unnormalize) ===
                # 反归一化并移除 Batch 维度
                # action: [1, Action_Dim] -> [Action_Dim]
                raw_action = postprocessor(action)
                
            # === 5. 返回结果 ===
            action_np = raw_action.squeeze(0).cpu().numpy()
            print(f"Action: {action_np}")
            socket.send(pickle.dumps(action_np))

            # 可视化 (可选)
            if VISUALIZE:
                # 这里的 batch['observation.images...'] 已经是 resize 过的吗？
                # 不，Preprocessor 不处理图像 resize，Resize 是在 Policy 内部发生的。
                # 所以这里可视化的是原始分辨率图像。
                vis_obs = {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                log_rerun_data(observation=vis_obs, action=torch.from_numpy(action_np), compress_images=False)

        except Exception as e:
            print(f"Loop Error: {e}")
            import traceback; traceback.print_exc()
            socket.send(pickle.dumps(None))

if __name__ == "__main__":
    main()