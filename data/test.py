import numpy as np
from types import SimpleNamespace

# 按照你项目实际导入的方式
from dataset import UnifiedTrafficDataset

def count_valid_windows(ds):
    """
    分层统计合法窗口数：
    1) 只要求 y 合法
    2) y + closeness
    3) y + closeness + period
    4) y + closeness + period + trend
    """
    is_obs = ds.is_observed
    t_out = ds.t_out
    len_c = ds.len_c
    spd = ds.spd
    use_period = ds.use_period
    use_trend = ds.use_trend

    total_candidates = 0
    cnt_y = 0
    cnt_yc = 0
    cnt_ycp = 0
    cnt_ycpt = 0

    # 你当前严格物理隔离法里，dataset 内部通常已经切过 split
    # 所以这里只在 ds 当前 split 内部统计
    T = len(is_obs)

    max_offset = len_c
    if use_period:
        max_offset = max(max_offset, spd)
    if use_trend:
        max_offset = max(max_offset, 7 * spd)

    start = max_offset
    end = T - t_out + 1

    for i in range(start, end):
        total_candidates += 1

        ok_y = np.all(is_obs[i:i+t_out] == 1)
        if not ok_y:
            continue
        cnt_y += 1

        ok_c = True
        if len_c > 0:
            ok_c = np.all(is_obs[i-len_c:i] == 1)
        if not ok_c:
            continue
        cnt_yc += 1

        ok_p = True
        if use_period:
            p_start = i - spd
            if not np.all(is_obs[p_start : p_start + t_out] == 1):
                continue
        if not ok_p:
            continue
        cnt_ycp += 1

        ok_t = True
        if use_trend:
            t_start = i - 7 * spd
            if not np.all(is_obs[t_start : t_start + t_out] == 1):
                continue
        if not ok_t:
            continue
        cnt_ycpt += 1

    return {
        "total_candidates": total_candidates,
        "valid_y": cnt_y,
        "valid_y_c": cnt_yc,
        "valid_y_c_p": cnt_ycp,
        "valid_y_c_p_t": cnt_ycpt,
    }


def longest_zero_run(arr):
    """统计 is_observed 中最长连续缺失段长度"""
    max_run = 0
    cur = 0
    for x in arr:
        if x == 0:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def print_split_stats(split_name, ds):
    is_obs = ds.is_observed.astype(int)
    obs_ratio = is_obs.mean()
    miss_ratio = 1.0 - obs_ratio
    zero_run = longest_zero_run(is_obs)

    print(f"\n========== {split_name.upper()} ==========")
    print(f"is_observed 长度: {len(is_obs)}")
    print(f"观测比例 obs_ratio: {obs_ratio:.6f}")
    print(f"缺失比例 miss_ratio: {miss_ratio:.6f}")
    print(f"最长连续缺失长度: {zero_run}")

    stats = count_valid_windows(ds)
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    # 请按你项目实际的参数调整 city 名称
    train_ds = UnifiedTrafficDataset(
        city="TaxiBJ",  # 替换为你实际的数据集城市名
        split="train",  # 用于加载 train 数据集
        len_c=6,
        use_period=True,
        use_trend=True,
        t_out=3,
        view_type="grid"  # 根据你的需求，选择 grid 或 graph
    )

    val_ds = UnifiedTrafficDataset(
        city="TaxiBJ",  # 替换为你实际的数据集城市名
        split="val",  # 用于加载 val 数据集
        len_c=6,
        use_period=True,
        use_trend=True,
        t_out=3,
        view_type="grid"  # 根据你的需求，选择 grid 或 graph
    )

    test_ds = UnifiedTrafficDataset(
        city="TaxiBJ",  # 替换为你实际的数据集城市名
        split="test",  # 用于加载 test 数据集
        len_c=6,
        use_period=True,
        use_trend=True,
        t_out=3,
        view_type="grid"  # 根据你的需求，选择 grid 或 graph
    )

    print_split_stats("train", train_ds)
    print_split_stats("val", val_ds)
    print_split_stats("test", test_ds)