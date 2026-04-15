#!/usr/bin/env python3
"""
戍边哨卡小说全流程测试
"""

import sys
import os

# 确保使用正确的路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

# 设置命令行参数 - 使用 novel_chapter 能力
sys.argv = [
    'main.py',
    '--capability', 'novel_chapter',
    '--chapter-file', 'stories/戍边哨卡.txt',
    '--language', 'Chinese',
    '--topic', '戍边哨卡',
    '--orientation', 'landscape'
]

print("=" * 60)
print("🎬 MoneyPrinterV2 小说章节视频生成测试")
print("=" * 60)
print(f"📖 小说: 戍边哨卡")
print(f"📄 文件: stories/戍边哨卡.txt")
print(f"🎤 TTS: MiniMax")
print(f"📐 方向: 横屏 (16:9)")
print(f"🌐 语言: 中文")
print(f"🎯 能力: novel_chapter")
print("=" * 60)
print()

# 导入并运行main
from main import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
