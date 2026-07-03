# import os
# import cv2
# import numpy as np
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor

# # 路径替换成环境里的实际路径
# checkpoint_path = "/root/sam2/checkpoints/sam2_hiera_large.pt"
# model_cfg_path = "sam2/sam2_hiera_l.yaml"
# image_path = "/root/autodl-tmp/Ego-IRGBench_dataset/RGB/ZY20210800001_H1_C1_N19_S100_s02_T1_00044.jpg"  # 你本地随便拿一张图片测试

# assert os.path.exists(checkpoint_path), "Checkpoint not found"
# assert os.path.exists(model_cfg_path), "Config not found"
# assert os.path.exists(image_path), "Test image not found"

# # 加载 SAM2
# print("Loading SAM2...")
# sam2_model = build_sam2(model_cfg_path, checkpoint_path)
# predictor = SAM2ImagePredictor(sam2_model)
# print("SAM2 loaded.")

# # 准备图像
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor.set_image(image_rgb)

# # 随机测试两个 bbox（左上角 100x100，图片中部 150x150）
# h, w, _ = image.shape
# bboxes = [
#     (10, 10, 110, 110),
#     (w//2-75, h//2-75, w//2+75, h//2+75)
# ]

# # 推理
# masks = []
# for box in bboxes:
#     x1,y1,x2,y2 = box
#     input_box = np.array([x1,y1,x2,y2])
#     mask_output, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
#     masks.append(mask_output[0].astype(np.uint8))

# print(f"Generated {len(masks)} masks for {len(bboxes)} bboxes.")
# for i, m in enumerate(masks):
#     print(f"Mask {i} shape: {m.shape}, unique values: {np.unique(m)}")

# # 可选：保存可视化图像
# for i, m in enumerate(masks):
#     vis = (m * 255).astype(np.uint8)
#     cv2.imwrite(f"mask_{i}.png", vis)
# print("Masks saved.")


# import os
# import logging
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def get_sam2_predictor(
#     checkpoint_path: str = "/root/sam2/checkpoints/sam2_hiera_large.pt",
#     model_cfg_name: str = "sam2_hiera_l.yaml"
# ):
#     """单例模式加载 SAM2 predictor（测试版）"""
#     global _sam2_predictor
#     if '_sam2_predictor' not in globals():
#         _sam2_predictor = None

#     if _sam2_predictor is None:
#         logger.info("Initializing SAM2 predictor for the first time...")

#         # ✅ 只检查 checkpoint
#         if not os.path.exists(checkpoint_path):
#             logger.error(f"SAM2 checkpoint not found: {checkpoint_path}")
#             return None

#         # ❌ 不要检查 config，因为这只是名字，Hydra 会自己去找
#         # if not os.path.exists(model_cfg_name):
#         #     logger.error(f"SAM2 config file not found: {model_cfg_name}")
#         #     return None

#         try:
#             logger.info(f"Loading SAM2 model with config: {model_cfg_name}")
#             sam2_model = build_sam2(model_cfg_name, checkpoint_path)
#             _sam2_predictor = SAM2ImagePredictor(sam2_model)
#             logger.info("SAM2 model loaded successfully.")
#         except Exception as e:
#             logger.error(f"Failed to load SAM2 model: {e}")
#             _sam2_predictor = None

#     return _sam2_predictor


# if __name__ == "__main__":
#     predictor = get_sam2_predictor()
#     if predictor is None:
#         print("❌ Failed to initialize SAM2 predictor")
#     else:
#         print("✅ SAM2 predictor created successfully:", predictor)


import os
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 路径替换成环境里的实际路径
checkpoint_path = "/root/sam2/checkpoints/sam2_hiera_large.pt"
model_cfg_name = "sam2_hiera_l.yaml"  # ✅ 用配置名而不是路径
image_path = "/root/autodl-tmp/Ego-IRGBench_dataset/RGB/ZY20210800001_H1_C1_N19_S100_s02_T1_00044.jpg"

# 只检查真正的文件路径：checkpoint + 测试图像
assert os.path.exists(checkpoint_path), "Checkpoint not found"
assert os.path.exists(image_path), "Test image not found"

# 加载 SAM2
print("Loading SAM2...")
sam2_model = build_sam2(model_cfg_name, checkpoint_path)  # ✅ 注意这里第二个参数还是 ckpt 绝对路径
predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 loaded.")

# 准备图像
image = cv2.imread(image_path)
assert image is not None, f"Failed to load image {image_path}"
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)
h, w, _ = image.shape

# 随机两个 bbox：左上角 100x100，中间 150x150
bboxes = [
    (10, 10, 110, 110),
    (w // 2 - 75, h // 2 - 75, w // 2 + 75, h // 2 + 75)
]

# 推理
masks = []
for box in bboxes:
    x1, y1, x2, y2 = box
    input_box = np.array([x1, y1, x2, y2])
    mask_output, _, _ = predictor.predict(
        box=input_box[None, :],
        multimask_output=False
    )
    mask = mask_output[0].astype(np.uint8)
    masks.append(mask)

print(f"Generated {len(masks)} masks for {len(bboxes)} bboxes.")
for i, m in enumerate(masks):
    print(f"Mask {i}: shape={m.shape}, unique={np.unique(m)}")

# 保存可视化结果
for i, m in enumerate(masks):
    vis = (m * 255).astype(np.uint8)
    out_path = f"mask_{i}.png"
    cv2.imwrite(out_path, vis)
    print(f"Saved {out_path}")