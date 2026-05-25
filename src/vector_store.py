import hashlib
import chromadb


class VectorStore:
    """Обёртка над ChromaDB с хранением на диске и косинусным сходством."""

    def __init__(self, db_path: str, collection_name: str):
        self._db_path = db_path
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _name_to_id(self, name: str) -> str:
        return hashlib.md5(name.encode("utf-8")).hexdigest()

    def add(self, names: list[str], embeddings: list[list[float]]) -> int:
        """Добавляет документы, пропуская дубликаты. Возвращает количество добавленных."""
        if not names:
            return 0

        candidate_ids = [self._name_to_id(n) for n in names]
        existing_result = self._collection.get(ids=candidate_ids)
        existing_ids = set(existing_result["ids"])

        new_items = [
            (name, emb, doc_id)
            for name, emb, doc_id in zip(names, embeddings, candidate_ids)
            if doc_id not in existing_ids
        ]
        if not new_items:
            return 0

        new_names, new_embeddings, new_ids = zip(*new_items)
        self._collection.add(
            documents=list(new_names),
            embeddings=list(new_embeddings),
            ids=list(new_ids),
        )
        return len(new_names)

    def search(
        self, query_embedding: list[float], top_k: int = 3, min_score: float = 0.5
    ) -> list[dict]:
        """Поиск по косинусному сходству. Возвращает список {"name": str, "score": float}."""
        total = self._collection.count()
        if total == 0:
            return []

        n_results = min(top_k, total)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "distances"],
        )

        matches = []
        for name, distance in zip(results["documents"][0], results["distances"][0]):
            # ChromaDB cosine distance = 1 - cosine_similarity, so similarity = 1 - distance
            similarity = 1.0 - distance
            if similarity >= min_score:
                matches.append({"name": name, "score": round(similarity, 4)})

        return matches

    def delete(self, name: str) -> bool:
        """Удаляет документ по названию. Возвращает True если документ существовал."""
        doc_id = self._name_to_id(name)
        result = self._collection.get(ids=[doc_id])
        if not result["ids"]:
            return False
        self._collection.delete(ids=[doc_id])
        return True

    def count(self) -> int:
        return self._collection.count()

    def info(self) -> dict:
        return {
            "count": self._collection.count(),
            "collection": self._collection.name,
            "db_path": self._db_path,
        }

    def reset(self):
        """Удаляет коллекцию и создаёт заново."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
