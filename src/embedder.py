import time
import sys
from langchain_gigachat import GigaChatEmbeddings


class Embedder:
    """Обёртка над GigaChatEmbeddings с повторными попытками при ошибках API."""

    _RETRYABLE_KEYWORDS = ("429", "rate limit", "timeout", "timed out", "connection", "503", "502")
    _AUTH_KEYWORDS = ("401", "403", "unauthorized", "authentication", "forbidden", "credentials")

    def __init__(self, credentials: str, model: str = "EmbeddingsGigaR", verify_ssl_certs: bool = False):
        self._model = model
        self._client = GigaChatEmbeddings(
            credentials=credentials,
            model=model,
            verify_ssl_certs=verify_ssl_certs,
        )
        self._max_retries = 3
        self._retry_delay = 2

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Векторизует список текстов. Возвращает список векторов размерностью 2560."""
        return self._with_retry(lambda: self._client.embed_documents(texts), context=f"{len(texts)} текстов")

    def encode_one(self, text: str) -> list[float]:
        """Векторизует один текст."""
        return self._with_retry(lambda: self._client.embed_query(text), context=f'"{text[:60]}"')

    def _with_retry(self, fn, context: str = ""):
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return fn()
            except Exception as e:
                err_str = str(e).lower()
                if any(kw in err_str for kw in self._AUTH_KEYWORDS):
                    print(
                        f"[Ошибка авторизации] Проверьте переменную GIGACHAT_CREDENTIALS.\nДетали: {e}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                is_retryable = any(kw in err_str for kw in self._RETRYABLE_KEYWORDS)
                last_exc = e
                if is_retryable and attempt < self._max_retries - 1:
                    wait = self._retry_delay * (attempt + 1)
                    print(
                        f"  [Предупреждение] Ошибка запроса ({context}), "
                        f"попытка {attempt + 1}/{self._max_retries}: {e}. Повтор через {wait}с...",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                else:
                    raise
        raise last_exc  # type: ignore[misc]
