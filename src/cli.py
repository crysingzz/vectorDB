import os
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.embedder import Embedder
from src.vector_store import VectorStore

_BATCH_SIZE = 50


def _load_config(config_path: str = "config.yaml") -> dict:
    load_dotenv()
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"[Ошибка] Файл конфигурации не найден: {config_path}", file=sys.stderr)
        sys.exit(1)
    content = config_file.read_text(encoding="utf-8")
    content = re.sub(r"\$\{([^}]+)\}", lambda m: os.getenv(m.group(1), ""), content)
    return yaml.safe_load(content)


def _create_store(config: dict) -> VectorStore:
    return VectorStore(
        db_path=config["database"]["path"],
        collection_name=config["database"]["collection_name"],
    )


def _create_embedder(config: dict) -> Embedder:
    credentials = config["gigachat"]["credentials"]
    if not credentials:
        print(
            "[Ошибка] Переменная окружения GIGACHAT_CREDENTIALS не задана.\n"
            "  Установите её командой: export GIGACHAT_CREDENTIALS='ваш_ключ'",
            file=sys.stderr,
        )
        sys.exit(1)
    return Embedder(
        credentials=credentials,
        model=config["gigachat"]["model"],
        verify_ssl_certs=config["gigachat"]["verify_ssl_certs"],
    )


def _read_names(input_file: str) -> list[str]:
    path = Path(input_file)
    if not path.exists():
        print(f"[Ошибка] Файл не найден: {input_file}", file=sys.stderr)
        sys.exit(1)
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        print("[Предупреждение] Файл пустой или не содержит названий.")
        sys.exit(0)
    return names


def cmd_index(input_file: str, config_path: str = "config.yaml"):
    config = _load_config(config_path)
    names = _read_names(input_file)
    embedder = _create_embedder(config)
    store = _create_store(config)

    print(f"Загружено {len(names)} названий из «{input_file}». Начинаю индексацию...")

    total_added = 0
    for i in range(0, len(names), _BATCH_SIZE):
        batch = names[i : i + _BATCH_SIZE]
        end = min(i + len(batch), len(names))
        print(f"  Векторизация {i + 1}–{end} из {len(names)}...")
        embeddings = embedder.encode(batch)
        added = store.add(batch, embeddings)
        total_added += added

    skipped = len(names) - total_added
    print(f"\nГотово. Добавлено: {total_added}  |  Пропущено дублей: {skipped}")
    print(f"Всего в индексе: {store.count()} документов.")


def cmd_search(query: str, config_path: str = "config.yaml"):
    config = _load_config(config_path)
    embedder = _create_embedder(config)
    store = _create_store(config)

    top_k = config["search"]["top_k"]
    min_score = config["search"]["min_score"]

    print(f'Запрос: "{query}"\n')

    embedding = embedder.encode_one(query)
    results = store.search(embedding, top_k=top_k, min_score=min_score)

    if not results:
        if store.count() == 0:
            print("Индекс пуст. Сначала выполните: python main.py index --input <файл>")
        else:
            print(f"Ничего не найдено (порог сходства: {min_score}).")
        return

    print("Результаты:")
    for i, r in enumerate(results, 1):
        name = r["name"]
        score = r["score"]
        print(f"  {i}. {name:<50} (сходство: {score:.2f})")


def cmd_add(name: str, config_path: str = "config.yaml"):
    config = _load_config(config_path)
    embedder = _create_embedder(config)
    store = _create_store(config)

    print(f'Добавляю: "{name}"')
    embedding = embedder.encode_one(name)
    added = store.add([name], [embedding])

    if added:
        print(f"Документ добавлен. Всего в индексе: {store.count()}.")
    else:
        print("Документ уже существует в индексе.")


def cmd_delete(name: str, config_path: str = "config.yaml"):
    config = _load_config(config_path)
    store = _create_store(config)

    if store.delete(name):
        print(f'Удалено: "{name}"')
        print(f"Всего в индексе: {store.count()} документов.")
    else:
        print(f'Документ не найден в индексе: "{name}"')


def cmd_reindex(input_file: str, config_path: str = "config.yaml"):
    config = _load_config(config_path)
    names = _read_names(input_file)
    embedder = _create_embedder(config)
    store = _create_store(config)

    print(f"Пересоздание индекса. Загружено {len(names)} названий из «{input_file}».")
    store.reset()

    for i in range(0, len(names), _BATCH_SIZE):
        batch = names[i : i + _BATCH_SIZE]
        end = min(i + len(batch), len(names))
        print(f"  Векторизация {i + 1}–{end} из {len(names)}...")
        embeddings = embedder.encode(batch)
        store.add(batch, embeddings)

    print(f"\nГотово. Всего в индексе: {store.count()} документов.")


def cmd_info(config_path: str = "config.yaml"):
    config = _load_config(config_path)
    store = _create_store(config)
    info = store.info()

    db_path = Path(info["db_path"]).resolve()
    print(f"База данных:    {db_path}")
    print(f"Коллекция:      {info['collection']}")
    print(f"Модель:         {config['gigachat']['model']}")
    print(f"Документов:     {info['count']}")
    print(f"Top-K:          {config['search']['top_k']}")
    print(f"Мин. сходство:  {config['search']['min_score']}")
