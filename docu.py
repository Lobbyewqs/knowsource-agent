"""列出本地可用知识库。"""
from config import BASE_DIR


def main():
    root = BASE_DIR / "knowledge_bases"
    for path in sorted(root.iterdir()):
        if path.is_dir():
            print(path.name)


if __name__ == "__main__":
    main()
