import typer

from src.cli import cmd_add, cmd_delete, cmd_index, cmd_info, cmd_reindex, cmd_search

app = typer.Typer(
    name="vectordb",
    help="Система семантического поиска документов (GigaChat + ChromaDB)",
    add_completion=False,
)


@app.command("index", help="Индексация документов из текстового файла (одно название на строку)")
def index(
    input_file: str = typer.Option(..., "--input", "-i", help="Путь к файлу со списком документов"),
):
    cmd_index(input_file)


@app.command("search", help="Семантический поиск по запросу")
def search(
    query: str = typer.Option(..., "--query", "-q", help="Поисковый запрос"),
):
    cmd_search(query)


@app.command("add", help="Добавить один документ в индекс")
def add(
    name: str = typer.Option(..., "--name", "-n", help="Название документа"),
):
    cmd_add(name)


@app.command("delete", help="Удалить документ из индекса по названию")
def delete(
    name: str = typer.Option(..., "--name", "-n", help="Название документа"),
):
    cmd_delete(name)


@app.command("reindex", help="Полное пересоздание индекса из файла")
def reindex(
    input_file: str = typer.Option(..., "--input", "-i", help="Путь к файлу со списком документов"),
):
    cmd_reindex(input_file)


@app.command("info", help="Показать статистику индекса")
def info():
    cmd_info()


if __name__ == "__main__":
    app()
