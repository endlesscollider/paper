#!/bin/bash
# 每日快报定时执行脚本
# 由 cron 在每天 00:00 调用

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
LOG_FILE="$SCRIPT_DIR/daily_report.log"

echo "========================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始生成每日快报" >> "$LOG_FILE"

# 加载环境变量
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# 执行脚本
cd "$PROJECT_DIR"
"$VENV_PYTHON" "$SCRIPT_DIR/daily_report.py" >> "$LOG_FILE" 2>&1

# 自动 git commit + push
cd "$PROJECT_DIR"
git add 每日快报/
if ! git diff --cached --quiet; then
    DATE=$(ls 每日快报/*.md | grep -v index | sort | tail -1 | sed 's/.*\///' | sed 's/\.md//')
    git commit -m "📰 每日快报: ${DATE}" >> "$LOG_FILE" 2>&1
    git push >> "$LOG_FILE" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 已提交并推送" >> "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 无变更，跳过提交" >> "$LOG_FILE"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 完成" >> "$LOG_FILE"
