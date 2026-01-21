"""
lerobot_server.py (Server A - GPU)
适配: 3x RGB Cameras + 2x Depth Maps
"""
import zmq
import torch
import pickle
import numpy as np
import cv2
from peft import PeftModel
from transformers import AutoTokenizer
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ================= 配置区域 =================
ADAPTER_REPO_ID = "moriis/pi05_piper_third" 
BASE_MODEL_ID = "lerobot/pi05_base"

PORT = 5555
DEVICE = "cuda"
VISUALIZE = False

# 键名映射：[Server B Key] -> [Model Dataset Key]
# 必须完全匹配您 info.json 中的 keys
CAMERA_MAPPING = {
    # 1. 主视角 (Main)
    "pikaGripperDepthCamera":   "observation.images.base_0_rgb",
    
    # 2. 鱼眼视角 (Fisheye) -> 映射为 Left Wrist
    "pikaGripperFisheyeCamera": "observation.images.left_wrist_0_rgb",
    
    # 3. 第三人称视角 (Third Person) -> 映射为 Right Wrist [新增]
    "pikaThirdPersonCamera":    "observation.images.right_wrist_0_rgb",
}

# PI0.5 配置
MODEL_STATE_DIM = 32   # 从 configuration_pi05.py 可知通常 pad 到 32
MAX_TOKEN_LEN = 200    # 从 configuration_pi05.py 可知
TOKENIZER_ID = "google/paligemma-3b-pt-224" # 对应的 Tokenizer

# 深度图映射
# DEPTH_MAPPING = {
#     "pikaGripperDepthCamera_Depth": "observation.depths.pikaGripperDepthCamera",
#     "pikaThirdPersonCamera_Depth":  "observation.depths.pikaThirdPersonCamera",
# }

MODEL_STATE_DIM = 13
# ===========================================

def get_pi05_prompt(task_text, state_tensor):
    """
    复现 processor_pi05.py 中的 Prompt 构造逻辑:
    格式: "Task: {task}, State: {quantized_state};\nAction: "
    """
    # 1. 简单的文本清洗
    cleaned_text = task_text.strip().replace("_", " ").replace("\n", " ")
    
    # 2. 状态离散化 (Discretize State)
    # 源码逻辑：将 [-1, 1] 的状态映射到 0-255 的整数 bin
    # 注意：这里假设输入的 state_tensor 已经是归一化比较好的数据。
    # 如果你的机械臂发来的是原始弧度，这里直接用可能效果不好，但在 Inference Server 里
    # 我们暂时只能这么做，或者你需要手动除以最大值进行归一化。
    state_np = state_tensor.cpu().numpy()
    
    # 将数值 clip 到 -1, 1 之间防止越界
    state_np = np.clip(state_np, -1.0, 1.0)
    
    # 离散化 (0 ~ 255)
    bins = np.linspace(-1, 1, 256 + 1)[:-1]
    discretized_states = np.digitize(state_np, bins) - 1
    
    # 3. 拼接 State 字符串
    state_str = " ".join(map(str, discretized_states))
    
    # 4. 最终 Prompt
    full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
    return full_prompt

def main():
    print(f"Server A: Loading Tokenizer from {TOKENIZER_ID}...")
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, padding_side="right")

    print(f"Server A: Loading BASE policy from {BASE_MODEL_ID}...")
    try:
        # 1. 先加载基座模型 (Base Model)
        policy = PI05Policy.from_pretrained(BASE_MODEL_ID)
        
        # 2. 再加载 LoRA 适配器 (Adapter)
        print(f"Server A: Loading LoRA adapter from {ADAPTER_REPO_ID}...")
        policy = PeftModel.from_pretrained(policy, ADAPTER_REPO_ID)
        
        # 3. 此时模型结构发生变化，需要确保合并或正确设置模式
        # 对于推理，通常建议 merge (可选，但为了性能推荐)
        # policy = policy.merge_and_unload() 
        
        policy.to(DEVICE)
        policy.eval()
        print("Server A: Policy (Base + LoRA) loaded successfully.")
        
    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    if VISUALIZE:
        init_rerun(session_name="Server_Inference", ip="127.0.0.1", port=9876)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    print(f"Server A: Listening on port {PORT}...")
    
    warned_keys = set()
    while True:
        try:
            msg = socket.recv()
            payload = pickle.loads(msg)
            
            task_text = payload.get("text", "Grab the carrot and put it into the box.")
            
            observation = {}

            required_model_keys = [
                "observation.images.base_0_rgb",
                "observation.images.left_wrist_0_rgb",
                "observation.images.right_wrist_0_rgb"
            ]
            
            # 用于判断是否有任何图像被处理
            processed_any_image = False

            # --- 1. 处理 RGB 图像 ---
            for hw_key, model_key in CAMERA_MAPPING.items():
                if hw_key in payload.get('images', {}):
                    img_bytes = payload['images'][hw_key]
                    if img_bytes is not None:
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # Resize 这里的 224, 224 取决于 config.json 中的 "image_resolution"
                            # 你的 config 说是 224x224，为了保险最好 Resize 一下
                            frame_rgb = cv2.resize(frame_rgb, (224, 224))
                            
                            observation[model_key] = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                            processed_any_image = True
                            
            if processed_any_image:
                for req_key in required_model_keys:
                    if req_key not in observation:
                        # 生成 3x224x224 的全黑/全灰图像
                        # 注意：必须和上面的 Tensor 设备/类型一致，但在放入 batch 前先在 CPU 生成即可
                        dummy_img = torch.zeros((3, 224, 224), dtype=torch.float32)
                        observation[req_key] = dummy_img
                        # [修改] 只在第一次遇到这个 key 时打印 Warning
                        if req_key not in warned_keys:
                            print(f"Warning: Filling missing key {req_key} with dummy black image. (Suppressed for future steps)")
                            warned_keys.add(req_key)

            # --- 2. 处理深度图 (可选) ---
            # 深度图通常是 16位 (CV_16U)，不能用普通的 imdecode
            # for hw_key, model_key in DEPTH_MAPPING.items():
            #     if hw_key in payload.get('depths', {}):
            #         depth_bytes = payload['depths'][hw_key]
            #         if depth_bytes is not None:
            #             nparr = np.frombuffer(depth_bytes, np.uint8)
            #             # 注意：深度图解码需要标志位 -1 (IMREAD_UNCHANGED)
            #             depth_frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            #             if depth_frame is not None:
            #                 # 增加 Channel 维度 (H,W) -> (1,H,W)
            #                 depth_tensor = torch.from_numpy(depth_frame.astype(np.float32)).unsqueeze(0)
            #                 observation[model_key] = depth_tensor

            # --- 3. 处理状态 ---
            joints = payload.get('joint_state', [])
            gripper = payload.get('gripper_state', [0.0])
            base_state = list(joints) + list(gripper)
            
            # Pad 到 32 维 (PI05Config.max_state_dim)
            if len(base_state) < MODEL_STATE_DIM:
                base_state += [0.0] * (MODEL_STATE_DIM - len(base_state))
            
            # 转 Tensor
            state_tensor = torch.tensor(base_state, dtype=torch.float32)
            observation["observation.state"] = state_tensor

            prompt = get_pi05_prompt(task_text, state_tensor)
            
            # Tokenize
            tokenized = tokenizer(
                prompt, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=MAX_TOKEN_LEN, 
                truncation=True
            )

            # --- 4. 增加 Batch 维度 ---
            batch = {}
            for k, v in observation.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0).to(DEVICE)

            batch["observation.language.tokens"] = tokenized.input_ids.to(DEVICE)
            # 强制转换为 bool 类型
            batch["observation.language.attention_mask"] = tokenized.attention_mask.to(DEVICE).bool()

            # --- 5. 推理 ---
            with torch.no_grad():
                action = policy.select_action(batch)

            # --- 6. 返回动作 ---
            if action.ndim > 1: action = action[0]
            action_list = action.cpu().numpy().tolist()
            socket.send(pickle.dumps(action_list[:13])) # 只发前13维

            # --- 7. 可视化 ---
            if VISUALIZE:
                # 仅为了可视化，将数据转回 CPU
                vis_obs = {k: v.cpu() for k, v in observation.items()}
                # Rerun 可能不支持直接显示深度图 tensor，这里可能会报错，建议只看 RGB
                # 过滤掉 depth keys 避免报错 (如果 rerun 报错的话)
                vis_obs_rgb = {k:v for k,v in vis_obs.items() if "depth" not in k.lower()}
                
                log_rerun_data(
                    observation=vis_obs_rgb, 
                    action=action.cpu(),
                    compress_images=False
                )

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            socket.send(pickle.dumps(None))

if __name__ == "__main__":
    main()