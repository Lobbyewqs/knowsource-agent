"""
工具，设计文件清洗
"""

import os
import jieba
import unicodedata
import re

# 文本清洗
def clean_text(text):
    # 1. 统一字符标准
    text = unicodedata.normalize('NFKC', text)
    # 2. 清洗零宽空格 BOM不可见字符
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    last_line = ""  # 修正变量名，原本是 last_lines，但未使用
    for line in text.splitlines():
        # 多个空格和制表符压缩成单个空格
        line = re.sub(r"[ \t]+", " ", line).strip()
        # 去掉重复行
        if line and line == last_line:
            continue

        lines.append(line)
        last_line = line if line else last_line

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# 分词函数（修改了 jieba 调用方式）
def tokenize(text):
    # 所有的非中文、英文、数字、下划线、短横线都替换成空格
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9_-]", " ", text.lower())

    words = []
    # 兼容写法：如果 jieba 有 lcut 则用，否则用 cut 转为 list
    if hasattr(jieba, 'lcut'):
        seg_list = jieba.lcut(text)
    else:
        seg_list = list(jieba.cut(text))

    for word in seg_list:
        word = word.strip()
        if len(word) > 1 or word.isdigit():
            words.append(word)
    return words
