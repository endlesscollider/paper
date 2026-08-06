#!/usr/bin/env python3
"""
频域抖动分析脚本
对机器人操作数据做 FFT，计算高频能量占比（抖动分数）
"""
import os
import sys
import glob
import numpy as np
import h5py
from pathlib import Path


def compute_jitter_score(signal, high_freq_ratio=0.5):
    """
    计算一维或多维时序信号的抖动分数
    signal: (T, D) 或 (T,)
    high_freq_ratio: 频率轴后 X% 视为高频
    返回: 每维的抖动分数 (D,)
    """
    if signal.ndim == 1:
        signal = signal[:, None]
    T, D = signal.shape
    if T < 8:
        return np.zeros(D)
    
    # rFFT 沿时间轴
    spectrum = np.abs(np.fft.rfft(signal, axis=0))  # (T//2+1, D)
    
    # 排除直流(k=0)
    energy = spectrum[1:]  # (T//2, D)
    total_energy = energy.sum(axis=0)  # (D,)
    
    # 高频能量
    n_freqs = energy.shape[0]
    cutoff = int(n_freqs * (1 - high_freq_ratio))
    high_freq_energy = energy[cutoff:].sum(axis=0)  # (D,)
    
    score = high_freq_energy / (total_energy + 1e-8)
    return score


def analyze_episode(filepath, demo_key="data/demo_0"):
    """分析单个 episode 的各通道抖动"""
    results = {}
    
    with h5py.File(filepath, 'r') as f:
        # 关节位置（最重要的抖动指标）
        channels = {
            "left_arm_joint": f"{demo_key}/obs/left_arm_joint_position",
            "right_arm_joint": f"{demo_key}/obs/right_arm_joint_position",
            "left_eef_pos": f"{demo_key}/obs/left_eef_position_xyz",
            "right_eef_pos": f"{demo_key}/obs/right_eef_position_xyz",
            "left_gripper": f"{demo_key}/obs/left_gripper_joint_position",
            "right_gripper": f"{demo_key}/obs/right_gripper_joint_position",
        }
        
        for name, path in channels.items():
            if path in f:
                data = f[path][:]
                score = compute_jitter_score(data)
                results[name] = {
                    "mean_jitter": float(score.mean()),
                    "max_jitter": float(score.max()),
                    "per_dim": score.tolist(),
                    "length": data.shape[0],
                }
    
    return results


def analyze_folder(folder_path):
    """分析一个文件夹中所有 episode"""
    files = sorted(glob.glob(os.path.join(folder_path, "*.hdf5")))
    if not files:
        print(f"  ⚠️  {folder_path}: 没有找到 hdf5 文件")
        return None
    
    all_results = []
    for filepath in files:
        try:
            result = analyze_episode(filepath)
            result["_file"] = os.path.basename(filepath)
            all_results.append(result)
        except Exception as e:
            print(f"  ⚠️  {os.path.basename(filepath)}: {e}")
    
    return all_results


def print_summary(folder_name, results):
    """打印一个文件夹的汇总统计"""
    if not results:
        return
    
    print(f"\n{'='*70}")
    print(f"📁 {folder_name}")
    print(f"   Episodes: {len(results)}")
    print(f"{'='*70}")
    
    # 汇总每个通道
    channels = ["left_arm_joint", "right_arm_joint", "left_eef_pos", 
                "right_eef_pos", "left_gripper", "right_gripper"]
    
    print(f"\n{'通道':<20} {'平均抖动':>10} {'最大抖动':>10} {'最差Episode':>20}")
    print(f"{'-'*60}")
    
    for ch in channels:
        jitters = []
        worst_ep = ""
        worst_val = 0
        for r in results:
            if ch in r:
                val = r[ch]["mean_jitter"]
                jitters.append(val)
                if val > worst_val:
                    worst_val = val
                    worst_ep = r["_file"]
        
        if jitters:
            avg = np.mean(jitters)
            mx = np.max(jitters)
            # 用颜色标记：>0.4 红色警告，>0.3 黄色注意
            flag = "🔴" if avg > 0.4 else ("🟡" if avg > 0.3 else "🟢")
            print(f"{flag} {ch:<18} {avg:>10.4f} {mx:>10.4f} {worst_ep:>20}")
    
    # 找出抖动最严重的 episode
    print(f"\n  📊 抖动最严重的 Top-5 Episodes (按 arm joint 平均):")
    episode_scores = []
    for r in results:
        arm_scores = []
        for ch in ["left_arm_joint", "right_arm_joint"]:
            if ch in r:
                arm_scores.append(r[ch]["mean_jitter"])
        if arm_scores:
            episode_scores.append((r["_file"], np.mean(arm_scores)))
    
    episode_scores.sort(key=lambda x: x[1], reverse=True)
    for name, score in episode_scores[:5]:
        flag = "🔴" if score > 0.4 else ("🟡" if score > 0.3 else "🟢")
        print(f"     {flag} {name}: {score:.4f}")


def main():
    base_dir = "/media/wahaha/EXTERNAL_USB/Ego2_ok"
    
    # 自动扫描所有子文件夹
    folders = sorted([
        os.path.join(base_dir, d) 
        for d in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, d))
    ])
    
    print("🔬 频域抖动分析")
    print("=" * 70)
    print("指标说明：")
    print("  抖动分数 = 高频能量 / 总能量 (排除直流)")
    print("  范围 0~1，越大越抖。阈值：🟢<0.3  🟡0.3~0.4  🔴>0.4")
    print("  高频定义：频率轴后 50% 的分量")
    
    for folder in folders:
        folder_name = os.path.basename(folder)
        if not os.path.exists(folder):
            print(f"\n⚠️  路径不存在: {folder}")
            continue
        results = analyze_folder(folder)
        if results:
            print_summary(folder_name, results)
    
    print(f"\n{'='*70}")
    print("✅ 分析完成")


if __name__ == "__main__":
    main()
