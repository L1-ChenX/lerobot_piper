"""
lerobot_server_v3.py
修复版: 手动构造配置 -> 加载 Base -> 加载 Adapter
"""
import zmq
import torch
import pickle
import numpy as np
import cv2
import json
from peft import PeftModel
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# 引入 LeRobot 的核心类
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ================= 配置区域 =================
ADAPTER_REPO_ID = "moriis/pi05_piper_third"
BASE_MODEL_ID = "lerobot/pi05_base"
PORT = 5555
DEVICE = "cuda"
VISUALIZE = False

# [关键] 这里的 Key 必须和 config.json 里的完全一致
# 因为我们使用的是没有 Remap 的配置
CAMERA_MAPPING = {
    "pikaGripperDepthCamera":   "observation.images.pikaGripperDepthCamera",
    "pikaGripperFisheyeCamera": "observation.images.pikaGripperFisheyeCamera",
    "pikaThirdPersonCamera":    "observation.images.pikaThirdPersonCamera",
}

MODEL_STATE_DIM = 32
MAX_TOKEN_LEN = 200
TOKENIZER_ID = "google/paligemma-3b-pt-224"
# ===========================================

def get_pi05_prompt(task_text, state_tensor):
    cleaned_text = task_text.strip().replace("_", " ").replace("\n", " ")
    state_np = state_tensor.cpu().numpy()
    state_np = np.clip(state_np, -1.0, 1.0)
    bins = np.linspace(-1, 1, 256 + 1)[:-1]
    discretized_states = np.digitize(state_np, bins) - 1
    state_str = " ".join(map(str, discretized_states))
    return f"Task: {cleaned_text}, State: {state_str};\nAction: "

def get_clean_config(repo_id):
    """
    手动下载并构建 PI05Config 对象，绕过所有自动加载的坑
    """
    print(f"Downloading config from {repo_id}...")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    # 1. 清理不支持的字段
    keys_to_remove = ["type", "transformers_version", "_commit_hash", "peft", "use_peft"]
    for k in keys_to_remove:
        if k in cfg_dict:
            del cfg_dict[k]

    # 2. 转换 Enum 类型 (NormalizationMode)
    if "normalization_mapping" in cfg_dict:
        norm_map = {}
        for k, v in cfg_dict["normalization_mapping"].items():
            # 将字符串 "IDENTITY" 转为 NormalizationMode.IDENTITY
            norm_map[k] = NormalizationMode[v] if isinstance(v, str) else v
        cfg_dict["normalization_mapping"] = norm_map

    # 3. 转换 PolicyFeature 对象
    def dict_to_feature(features_dict):
        new_features = {}
        for name, data in features_dict.items():
            if isinstance(data, dict):
                # 将 "VISUAL" 字符串转为 FeatureType.VISUAL
                if "type" in data and isinstance(data["type"], str):
                    data["type"] = FeatureType[data["type"]]
                new_features[name] = PolicyFeature(**data)
            else:
                new_features[name] = data
        return new_features

    if "input_features" in cfg_dict:
        cfg_dict["input_features"] = dict_to_feature(cfg_dict["input_features"])
    
    if "output_features" in cfg_dict:
        cfg_dict["output_features"] = dict_to_feature(cfg_dict["output_features"])

    # 4. 实例化
    return PI05Config(**cfg_dict)

def main():
    print(f"Server A: Loading Tokenizer from {TOKENIZER_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, padding_side="right")

    try:
        # [Step 1] 获取干净的配置对象
        user_config = get_clean_config(ADAPTER_REPO_ID)
        
        # [Step 2] 加载 Base 模型，但强制注入我们的 Config
        # 这一步会下载并加载 lerobot/pi05_base 的权重，但使用 pika... 的配置
        print(f"Server A: Loading BASE weights from {BASE_MODEL_ID} with CUSTOM CONFIG...")
        policy = PI05Policy.from_pretrained(BASE_MODEL_ID, config=user_config)
        
        # [Step 3] 加载 Adapter
        print(f"Server A: Loading Adapter from {ADAPTER_REPO_ID}...")
        policy = PeftModel.from_pretrained(policy, ADAPTER_REPO_ID)
        
        policy.to(DEVICE)
        policy.eval()
        print("✅ Server A: Policy loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading policy: {e}")
        import traceback
        traceback.print_exc()
        return

    if VISUALIZE:
        init_rerun(session_name="Server_Inference", ip="127.0.0.1", port=9876)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    print(f"🎧 Server A: Listening on port {PORT}...")
    
    while True:
        try:
            msg = socket.recv()
            payload = pickle.loads(msg)
            
            task_text = payload.get("text", "Grab the carrot")
            observation = {}
            processed_any_image = False

            # --- 图像处理 ---
            for hw_key, model_key in CAMERA_MAPPING.items():
                if hw_key in payload.get('images', {}):
                    img_bytes = payload['images'][hw_key]
                    if img_bytes is not None:
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_rgb = cv2.resize(frame_rgb, (224, 224))
                            observation[model_key] = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                            processed_any_image = True
            
            # 填充缺失图像 (必须填充，否则模型会报错)
            required_keys = list(CAMERA_MAPPING.values())
            if processed_any_image:
                for req_key in required_keys:
                    if req_key not in observation:
                        observation[req_key] = torch.zeros((3, 224, 224), dtype=torch.float32)

            # --- 状态处理 ---
            joints = payload.get('joint_state', [])
            gripper = payload.get('gripper_state', [0.0])
            base_state = list(joints) + list(gripper)
            
            if len(base_state) < MODEL_STATE_DIM:
                base_state += [0.0] * (MODEL_STATE_DIM - len(base_state))
            
            state_tensor = torch.tensor(base_state, dtype=torch.float32)
            observation["observation.state"] = state_tensor

            # --- 推理 ---
            prompt = get_pi05_prompt(task_text, state_tensor)
            tokenized = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=MAX_TOKEN_LEN, truncation=True)

            batch = {k: v.unsqueeze(0).to(DEVICE) for k, v in observation.items() if isinstance(v, torch.Tensor)}
            batch["observation.language.tokens"] = tokenized.input_ids.to(DEVICE)
            batch["observation.language.attention_mask"] = tokenized.attention_mask.to(DEVICE).bool()

            with torch.no_grad():
                action = policy.select_action(batch)

            if action.ndim > 1: action = action[0]
            socket.send(pickle.dumps(action.cpu().numpy().tolist()[:13])) # 只返回前13维

            if VISUALIZE:
                vis_obs = {k: v.cpu() for k, v in observation.items()}
                log_rerun_data(observation=vis_obs, action=action.cpu(), compress_images=False)

        except Exception as e:
            print(f"Error in loop: {e}")
            socket.send(pickle.dumps(None))

if __name__ == "__main__":
    main()