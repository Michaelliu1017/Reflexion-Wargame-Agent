#!/bin/bash

cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "creating virtual env and installing openai..."
  python3 -m venv .venv
  .venv/bin/pip install openai
fi
if [ -z "$OPENAI_API_KEY" ]; then
  echo "请先设置: export OPENAI_API_KEY=sk-你的key"
  exit 1
fi
exec .venv/bin/python chatgpt_driver.py --auto "$@"
