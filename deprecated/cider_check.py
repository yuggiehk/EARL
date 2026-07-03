# import json
# import argparse
# import re
# import os

# import nltk
# from nltk.translate.meteor_score import meteor_score
# from nltk.tokenize import word_tokenize
# from pycocoevalcap.cider.cider import Cider


# def ensure_nltk():
#     for pkg in ["punkt", "wordnet"]:
#         try:
#             if pkg == "punkt":
#                 nltk.data.find(f"tokenizers/{pkg}")
#             else:
#                 nltk.data.find(f"corpora/{pkg}")
#         except LookupError:
#             nltk.download(pkg)


# def parse_xml_tags(text: str) -> dict:
#     tags = {}
#     tags["analyzing"] = (
#         m.group(1).strip()
#         if (m := re.search(r"<analyzing>(.*?)</analyzing>", text, re.DOTALL))
#         else ""
#     )
#     return tags


# def compute_meteor(candidate: str, reference: str) -> float:
#     if not candidate.strip() or not reference.strip():
#         return float("nan")
#     c_tokens = word_tokenize(candidate.lower())
#     r_tokens = word_tokenize(reference.lower())
#     return float(meteor_score([r_tokens], c_tokens))


# def main():
#     parser = argparse.ArgumentParser(description="逐条输出analyzing的CIDEr和METEOR")
#     parser.add_argument("results_path", type=str, help="推理结果JSON路径")
#     parser.add_argument("--out", type=str, default=None, help="输出文件路径（默认：<results>.analyzing_scores.tsv）")
#     args = parser.parse_args()

#     ensure_nltk()

#     with open(args.results_path, "r", encoding="utf-8") as f:
#         results = json.load(f)

#     gts_dict, res_dict = {}, {}
#     valid_ids = []
#     meteor_map = {}

#     for i, item in enumerate(results):
#         pred_text = item.get("prediction", "") or ""
#         gt_text = item.get("ground_truth", "") or ""
#         pred_an = parse_xml_tags(pred_text).get("analyzing", "")
#         gt_an = parse_xml_tags(gt_text).get("analyzing", "")

#         sid = str(i)
#         if pred_an and gt_an:
#             valid_ids.append(sid)
#             res_dict[sid] = [pred_an]
#             gts_dict[sid] = [gt_an]
#             meteor_map[sid] = compute_meteor(pred_an, gt_an)
#         else:
#             meteor_map[sid] = float("nan")

#     cider_map = {}
#     if valid_ids:
#         cider = Cider()
#         _, per_sample_scores = cider.compute_score(gts_dict, res_dict)
#         for sid, s in zip(valid_ids, per_sample_scores):
#             try:
#                 cider_map[sid] = float(s)
#             except Exception:
#                 cider_map[sid] = float("nan")

#     out_path = args.out or (os.path.splitext(args.results_path)[0] + ".analyzing_scores.tsv")
#     os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

#     with open(out_path, "w", encoding="utf-8") as wf:
#         for i in range(len(results)):
#             sid = str(i)
#             c = cider_map.get(sid, float("nan"))
#             m = meteor_map.get(sid, float("nan"))
#             if c == c and m == m:
#                 wf.write(f"{sid}\t{c:.6f}\t{m:.6f}\n")
#             else:
#                 wf.write(f"{sid}\tNA\tNA\n")


# if __name__ == "__main__":
#     main()


import json
import argparse
import re
import os

import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from pycocoevalcap.cider.cider import Cider  # 回滚到 pycocoevalcap


def ensure_nltk():
    for pkg in ["punkt", "wordnet"]:
        try:
            if pkg == "punkt":
                nltk.data.find(f"tokenizers/{pkg}")
            else:
                nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)


def parse_xml_tags(text: str) -> dict:
    tags = {}
    tags["analyzing"] = (
        m.group(1).strip()
        if (m := re.search(r"<analyzing>(.*?)</analyzing>", text, re.DOTALL))
        else ""
    )
    return tags


def normalize_text(text: str) -> str:
    """可选: 规范化文本（小写 + 移除标点），减少差异影响"""
    text = text.lower().rstrip('.').rstrip()  # 移除末尾句点和空格
    return text


def compute_meteor(candidate: str, reference: str) -> float:
    if not candidate.strip() or not reference.strip():
        return float("nan")
    c_tokens = word_tokenize(candidate.lower())
    r_tokens = word_tokenize(reference.lower())
    return float(meteor_score([r_tokens], c_tokens))


def main():
    parser = argparse.ArgumentParser(description="逐条输出analyzing的CIDEr和METEOR")
    parser.add_argument("results_path", type=str, help="推理结果JSON路径")
    parser.add_argument("--out", type=str, default=None, help="输出文件路径（默认：<results>.analyzing_scores.tsv）")
    args = parser.parse_args()

    ensure_nltk()

    with open(args.results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    gts_dict, res_dict = {}, {}
    valid_ids = []
    meteor_map = {}

    for i, item in enumerate(results):
        pred_text = item.get("prediction", "") or ""
        gt_text = item.get("ground_truth", "") or ""
        pred_an = parse_xml_tags(pred_text).get("analyzing", "")
        gt_an = parse_xml_tags(gt_text).get("analyzing", "")

        # 可选规范化（取消注释以启用）
        pred_an = normalize_text(pred_an)
        gt_an = normalize_text(gt_an)

        sid = str(i)
        if pred_an and gt_an:
            valid_ids.append(sid)
            res_dict[sid] = [pred_an]
            gts_dict[sid] = [gt_an]
            meteor_map[sid] = compute_meteor(pred_an, gt_an)
        else:
            meteor_map[sid] = float("nan")

    cider_map = {}
    if valid_ids:
        cider = Cider()
        try:
            _, per_sample_scores = cider.compute_score(gts_dict, res_dict)
            for sid, s in zip(valid_ids, per_sample_scores):
                cider_map[sid] = float(s) if s == s else float("nan")  # 处理 NaN
        except Exception as e:
            print(f"Error computing CIDEr: {e}")
            per_sample_scores = [float("nan")] * len(valid_ids)
            for sid in valid_ids:
                cider_map[sid] = float("nan")

    out_path = args.out or (os.path.splitext(args.results_path)[0] + ".analyzing_scores.tsv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as wf:
        for i in range(len(results)):
            sid = str(i)
            c = cider_map.get(sid, float("nan"))
            m = meteor_map.get(sid, float("nan"))
            if c == c and m == m:  # 检查非 NaN
                wf.write(f"{sid}\t{c:.6f}\t{m:.6f}\n")
            else:
                wf.write(f"{sid}\tNA\tNA\n")


if __name__ == "__main__":
    main()