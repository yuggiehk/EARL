import argparse

def parse_float(x: str):
    x = x.strip()
    if x.lower() in {"na", "nan", ""}:
        return None
    try:
        return float(x)
    except ValueError:
        return None

def main():
    ap = argparse.ArgumentParser(description="计算两列分数的平均值（忽略NA）")
    ap.add_argument("path", help="输入文件路径（例如：/root/VLM-R1/check_1.csv）")
    args = ap.parse_args()

    sum2 = count2 = 0
    sum3 = count3 = 0

    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split()  # 按任意空白分割（制表符/空格）
            if len(cols) < 3:
                continue
            v2 = parse_float(cols[1])
            v3 = parse_float(cols[2])
            if v2 is not None:
                sum2 += v2
                count2 += 1
            if v3 is not None:
                sum3 += v3
                count3 += 1

    mean2 = (sum2 / count2) if count2 else float("nan")
    mean3 = (sum3 / count3) if count3 else float("nan")

    print(f"col2_mean\t{mean2:.6f}\t(n={count2})")
    print(f"col3_mean\t{mean3:.6f}\t(n={count3})")

if __name__ == "__main__":
    main()