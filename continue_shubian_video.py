#!/usr/bin/env python3
"""
继续完成戍边哨卡视频生成（断点续传）
使用已有的章节分析结果，继续生成图片和合成视频
"""

import sys
import os

# 确保使用正确的路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

# 设置命令行参数 - 使用 novel_chapter 能力，并启用断点续传
sys.argv = [
    'main.py',
    '--capability', 'novel_chapter',
    '--chapter-file', 'stories/戍边哨卡.txt',
    '--language', 'Chinese',
    '--topic', '戍边哨卡',
    '--orientation', 'landscape',
    '--keep-temp'  # 保留临时文件，实现断点续传
]

print("=" * 60)
print("🔄 MoneyPrinterV2 断点续传 - 戍边哨卡")
print("=" * 60)
print(f"📖 小说: 戍边哨卡")
print(f"📄 文件: stories/戍边哨卡.txt")
print(f"🎤 TTS: MiniMax")
print(f"📐 方向: 横屏 (16:9)")
print(f"🌐 语言: 中文")
print(f"🎯 能力: novel_chapter")
print(f"♻️  模式: 断点续传 (--keep-temp)")
print("=" * 60)
print()

# 检查已有进度
import json
analysis_file = os.path.join(ROOT_DIR, ".mp", "last_chapter_analysis.json")
if os.path.exists(analysis_file):
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    print(f"✅ 发现已有章节分析")
    print(f"   - 总分段数: {analysis.get('total_segments', 0)}")
    print(f"   - 风格指南: {analysis.get('style_bible', '')[:80]}...")
    print()

# 检查已生成的图片
from glob import glob
existing_images = glob(os.path.join(ROOT_DIR, ".mp", "戍边哨卡_seg_*.jpg"))
print(f"📊 已生成图片: {len(existing_images)}/16")
if existing_images:
    for img in sorted(existing_images)[:3]:
        print(f"   ✓ {os.path.basename(img)}")
    if len(existing_images) > 3:
        print(f"   ... 还有 {len(existing_images) - 3} 张")
print()

print("🚀 继续生成流程...")
print("=" * 60)
print()

# 导入并运行main
from main import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        print("💡 提示: 再次运行此脚本可继续从当前进度开始")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 提示: 再次运行此脚本可继续从当前进度开始")
        sys.exit(1)
