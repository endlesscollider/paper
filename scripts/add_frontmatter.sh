#!/bin/bash
# 给前置知识文件加 frontmatter

declare -A TITLES
declare -A TAGS
declare -A ORDERS

TITLES[000a]="策略梯度与 PPO"
TITLES[000b]="扩散模型 DDPM"
TITLES[000c]="Diffusion Policy"
TITLES[000d]="行为克隆与 RL 微调范式"
TITLES[000e]="对数似然与变分下界"
TITLES[000f]="为什么扩散策略难以 RL 微调"
TITLES[000g]="Flow Matching 与连续归一化流"
TITLES[000h]="Consistency Model 与一步生成"

TAGS[000a]="[强化学习]"
TAGS[000b]="[扩散模型, 深度学习]"
TAGS[000c]="[扩散模型, 机器人, 模仿学习]"
TAGS[000d]="[模仿学习, 强化学习]"
TAGS[000e]="[深度学习]"
TAGS[000f]="[扩散模型, 强化学习]"
TAGS[000g]="[扩散模型, 深度学习]"
TAGS[000h]="[扩散模型, 深度学习]"

ORDERS[000a]=1
ORDERS[000b]=2
ORDERS[000c]=3
ORDERS[000d]=4
ORDERS[000e]=5
ORDERS[000f]=6
ORDERS[000g]=7
ORDERS[000h]=8

DIR="/home/wahaha/paper/前置知识"

for key in 000a 000b 000c 000d 000e 000f 000g 000h; do
  FILE=$(ls "$DIR"/${key}_*.md 2>/dev/null)
  if [ -z "$FILE" ]; then continue; fi
  
  # Check if already has frontmatter
  if head -1 "$FILE" | grep -q "^---"; then
    continue
  fi
  
  TITLE="${TITLES[$key]}"
  TAG="${TAGS[$key]}"
  ORDER="${ORDERS[$key]}"
  
  FRONTMATTER="---\ntitle: ${TITLE}\norder: ${ORDER}\ntags: ${TAG}\ncategory: 前置知识\n---\n\n"
  
  # Prepend frontmatter
  TEMP=$(mktemp)
  printf "%b" "$FRONTMATTER" > "$TEMP"
  cat "$FILE" >> "$TEMP"
  mv "$TEMP" "$FILE"
  
  echo "Added frontmatter to: $FILE"
done
