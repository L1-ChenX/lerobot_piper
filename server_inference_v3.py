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
import json
import os

# 引入 LeRobot 的核心类
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# 引入反归一化所需的类
from lerobot.processor import UnnormalizerProcessorStep, PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME

# ================= 配置区域 =================
ADAPTER_REPO_ID = "moriis/pi05_piper_third_v2"
BASE_MODEL_ID = "lerobot/pi05_base"
PORT = 5555
DEVICE = "cuda"
VISUALIZE = False

# [关键] 这里的 Key 必须和 config.json 里的完全一致
CAMERA_MAPPING = {
    "pikaGripperDepthCamera":   "observation.images.pikaGripperDepthCamera",
    "pikaGripperFisheyeCamera": "observation.images.pikaGripperFisheyeCamera",
    "pikaThirdPersonCamera":    "observation.images.pikaThirdPersonCamera",
}

MODEL_STATE_DIM = 32
MAX_TOKEN_LEN = 200
TOKENIZER_ID = "google/paligemma-3b-pt-224"

STATS_PATH = "../lerobot_dataset_third_v2/meta/stats.json"
# ===========================================

class Pi05PromptBuilder:
    def __init__(self, joint_min, joint_max):
        self.joint_min = np.array(joint_min)
        self.joint_max = np.array(joint_max)

    def normalize_state(self, state):
        # 转换输入为 numpy 数组
        state = np.array(state)
        
        # 计算分母，防止除以零
        denominator = self.joint_max - self.joint_min
        denominator[denominator == 0] = 1.0
        
        # Min-Max 归一化: 映射到 [-1, 1]
        norm_state = 2 * (state - self.joint_min) / denominator - 1.0
        
        # 截断超出范围的值 (这对推理很重要，防止异常值导致 Token 溢出)
        return np.clip(norm_state, -1.0, 1.0)

    def discretize_state(self, norm_state):
        # 线性分桶 [-1, 1] -> 256份
        bins = np.linspace(-1, 1, 256 + 1)[:-1]
        tokens = np.digitize(norm_state, bins) - 1
        return np.clip(tokens, 0, 255)

    def build_prompt(self, task_text, joint_state):
        # 1. 文本清洗
        clean_text = task_text.strip().replace("_", " ").replace("\n", " ")
        
        # 2. 状态处理 (截取前7维)
        current_joints = joint_state[:7] 
        norm_state = self.normalize_state(current_joints)
        tokens = self.discretize_state(norm_state)
        state_str = " ".join(map(str, tokens))
        
        # 3. 拼接 (格式严格对齐训练代码)
        full_prompt = f"Task: {clean_text}, State: {state_str};\nAction: "
        return full_prompt

def load_dataset_stats(stats_path, device="cpu"):
    """
    加载 stats.json 并将所有 list 数据转换为 torch.Tensor
    """
    print(f"正在加载统计文件: {stats_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"找不到统计文件: {stats_path}")

    with open(stats_path, 'r') as f:
        stats_dict = json.load(f)

    # 递归将所有 list 转为 Tensor
    def convert_to_tensor(item):
        if isinstance(item, dict):
            return {k: convert_to_tensor(v) for k, v in item.items()}
        elif isinstance(item, list):
            return torch.tensor(item, dtype=torch.float32, device=device)
        return item

    dataset_stats = convert_to_tensor(stats_dict)
    
    # 获取 state 的 min/max 用于 PromptBuilder (保留原有功能)
    state_min = dataset_stats["observation.state"]["min"].cpu().numpy().tolist()
    state_max = dataset_stats["observation.state"]["max"].cpu().numpy().tolist()
    
    print("✅ 成功加载 Dataset Stats (已转换为 Tensor)")
    return dataset_stats, state_min, state_max

def get_task_prompt(task_text):
    # Pi0 通常只需要纯文本任务描述，状态会通过 observation.state 自动注入
    # 注意：根据训练时的格式，有时需要特定的前缀，比如 "Task: "
    # 如果你的数据集是标准格式，通常只需要清洗一下文本
    return f"Task: {task_text.strip()}"

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

def resize_with_pad(image, target_size=224):
    """
    模拟 modeling_pi05.py 中的 resize_with_pad_torch 逻辑
    :param image: 输入图像 (H, W, C), BGR 或 RGB
    :param target_size: 目标尺寸 (int)
    :return: 归一化并填充后的 Tensor (C, H, W)
    """
    h, w = image.shape[:2]
    
    # 1. 计算缩放比例 (保持长宽比)
    # 代码逻辑是: ratio = max(cur_width / width, cur_height / height)
    # 这意味着它会基于最长边进行缩放，确保图像完全放入框内
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 2. 缩放
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 3. 创建画布并填充
    # 注意：训练代码中 padding value 对于 float32 是 -1.0
    # 我们这里先生成 [0, 255] 的 uint8，后面转 float 再归一化
    # 或者直接生成灰色背景 (127) 对应归一化后的 0，或者黑色 (0) 对应 -1?
    # modeling_pi05.py 中: value = -1.0 (float32, 此时图像范围是[-1, 1])
    # 这意味着填充区域是 "最黑" 的颜色。
    
    # 创建一个全黑画布 (0)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # 计算居中位置
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    
    # 填入图像
    canvas[top:top+new_h, left:left+new_w] = resized
    
    # 4. 转 RGB (如果输入是 BGR)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
    # 5. 归一化到 [-1, 1]
    # 先转 float [0, 1]
    img_tensor = torch.from_numpy(canvas).float() / 255.0
    # 再转 [-1, 1] (填充的 0 变成了 -1，与训练一致)
    img_tensor = img_tensor * 2.0 - 1.0
    
    # 6. 维度变换 (H, W, C) -> (C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1)
    
    return img_tensor

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

    
    # 为了防止路径错误，可以使用绝对路径构建 (可选)
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # STATS_PATH = os.path.join(base_dir, "../lerobot_dataset_third_v2/meta/stats.json")

    try:
        # [Step 1] 加载统计数据 (注意：传入 device 以便后续 GPU 计算)
        dataset_stats, real_min, real_max = load_dataset_stats(STATS_PATH, device=DEVICE)
        
        # [Step 2] 初始化 Prompt Builder
        prompt_builder = Pi05PromptBuilder(joint_min=real_min, joint_max=real_max)
        
        # [Step 3] 初始化 Post-processor (反归一化)
        print("Server A: 构建 Action 反归一化处理器...")
        unnormalizer = UnnormalizerProcessorStep(
            features=user_config.output_features,
            norm_map=user_config.normalization_mapping,
            stats=dataset_stats
        )
        postprocessor = PolicyProcessorPipeline(
            steps=[unnormalizer],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            # 删除 to_transition 和 to_output 参数
        )
        print("✅ Post-processor Ready!")
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
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
            
            task_text = payload.get("text", "Grab the carrot and put it into the box.")
            observation = {}
            payload_images = payload.get('images', {})
            
            # 1. 尝试解码所有存在的图像
            for hw_key, model_key in CAMERA_MAPPING.items():
                if hw_key in payload_images:
                    img_bytes = payload_images[hw_key]
                    if img_bytes is not None:
                        try:
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                # 1. 转 RGB
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                # 2. 使用新函数处理 (缩放+填充+归一化到-1~1)
                                # 注意: resize_with_pad 内部已经完成了 permute 和 归一化
                                observation[model_key] = resize_with_pad(frame_rgb, target_size=224)
                        except Exception:
                            pass

            missing_keys = []
            for req_key in CAMERA_MAPPING.values():
                if req_key not in observation:
                    missing_keys.append(req_key)
            
            if len(missing_keys) > 0:
                # 如果缺少任何一张图，拒绝推理
                print(f"🛑 STRICT MODE: 丢弃帧! 缺少图像: {missing_keys}")
                # 发送 None 给客户端，客户端会打印 Warning 并保持当前姿态或重试
                socket.send(pickle.dumps(None))
                continue # 直接进入下一次循环，不跑 inference

            # --- 状态处理 ---
            # 1. 获取关节数据
            joints = payload.get('joint_state', [])
            if len(joints) != 7:
                print(f"⚠️ 关节数据维度错误: 期望 7, 实际 {len(joints)}。跳过此帧。")
                socket.send(pickle.dumps(None))
                continue  # 直接进入下一次循环

            # 3. 高效转换 Tensor
            state_tensor = torch.tensor(joints, dtype=torch.float32)

            # 4. 存入观测字典
            observation["observation.state"] = state_tensor

            # --- 推理 ---
            prompt_text = prompt_builder.build_prompt(task_text, joints)
            
            # 调试打印，确认 State 是否变成了数字序列
            print(f"Generated Prompt: {prompt_text}")
            
            tokenized = tokenizer(
                prompt_text, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=MAX_TOKEN_LEN, 
                truncation=True
            )

            batch = {k: v.unsqueeze(0).to(DEVICE) for k, v in observation.items() if isinstance(v, torch.Tensor)}
            batch["observation.language.tokens"] = tokenized.input_ids.to(DEVICE)
            batch["observation.language.attention_mask"] = tokenized.attention_mask.to(DEVICE).bool()

            # --- 推理 ---
            with torch.no_grad():
                # 1. 获取模型输出 (Normalized)
                raw_action_norm = policy.select_action(batch)
                
                # 2. 反归一化
                # 确保维度是 [Batch, Dim]
                if raw_action_norm.ndim == 1:
                    raw_action_norm = raw_action_norm.unsqueeze(0)
                
                # 使用 Post-processor
                action_dict = {"action": raw_action_norm}
                unnormalized_dict = postprocessor(action_dict)
                physical_action = unnormalized_dict["action"]
            
            # 3. 转 Numpy
            action_np = physical_action.squeeze(0).cpu().numpy()

            # 调试打印 (对比一下就知道是否修复了)
            # 正常物理值: 夹爪应该在 0.0 ~ 0.1 之间，关节应该在 -3.14 ~ 3.14 之间
            print(f"DEBUG -> Norm: {raw_action_norm[0, :3].cpu().numpy()} | Phys: {action_np[:3]}")

            socket.send(pickle.dumps(action_np))

            if VISUALIZE:
                # 可视化依然使用完整的 chunk
                vis_action = torch.from_numpy(action_np)
                if vis_action.ndim == 3: vis_action = vis_action[0]
                vis_obs = {k: v.cpu() for k, v in observation.items()}
                log_rerun_data(observation=vis_obs, action=vis_action, compress_images=False)

        except Exception as e:
            print(f"Error in loop: {e}")
            import traceback
            traceback.print_exc()
            socket.send(pickle.dumps(None))

if __name__ == "__main__":
    main()