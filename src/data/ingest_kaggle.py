"""Kaggle Dataset Ingestion: Downloads, filters @AppleSupport pairs via DuckDB, and indexes into ChromaDB."""

import os
import sys
import zipfile
import subprocess
from pathlib import Path
from typing import Optional
import duckdb
import typer
from rich.console import Console
from rich.panel import Panel

from src.config import DATA_DIR, TARGET_BRAND
from src.drafting.vector_store import HistoricalVectorStore

app = typer.Typer(help="Kaggle Dataset Ingestion & In-Memory DuckDB Streamer")
console = Console()

KAGGLE_DATASET_ID = "thoughtvector/customer-support-on-twitter"
CSV_FILENAME = "twcs.csv"
ZIP_FILENAME = "customer-support-on-twitter.zip"
EXTRACTED_PAIRS_PATH = DATA_DIR / "apple_support_kaggle_pairs.jsonl"


def download_dataset():
    """Attempts to download the Kaggle dataset using Kaggle CLI."""
    console.print(f"[bold cyan]Attempting to download {KAGGLE_DATASET_ID} via Kaggle API...[/bold cyan]")
    try:
        cmd = ["kaggle", "datasets", "download", KAGGLE_DATASET_ID, "-p", str(DATA_DIR)]
        subprocess.run(cmd, check=True)
        console.print("[bold green]Download complete![/bold green]")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        console.print(Panel(
            "[bold yellow]Kaggle CLI not configured or failed.[/bold yellow]\n\n"
            "To download manually:\n"
            "1. Run: [bold green]kaggle datasets download thoughtvector/customer-support-on-twitter[/bold green]\n"
            "2. Unzip or move [bold green]twcs.csv[/bold green] into the [bold cyan]data/[/bold cyan] folder.\n"
            "3. Run: [bold green]python -m src.data.ingest_kaggle extract[/bold green]",
            title="[bold red]Kaggle Setup Instruction[/bold red]"
        ))
        sys.exit(1)


@app.command()
def extract(
    csv_path: Optional[str] = typer.Option(None, "--csv", "-c", help="Path to twcs.csv"),
    max_pairs: int = typer.Option(5000, "--max-pairs", "-m", help="Maximum pairs to extract"),
    index_chroma: bool = typer.Option(True, "--index/--no-index", help="Index into ChromaDB vector store"),
):
    """Streams twcs.csv using DuckDB, filters @AppleSupport pairs, and optionally indexes into ChromaDB."""
    target_csv = Path(csv_path) if csv_path else DATA_DIR / CSV_FILENAME
    target_zip = DATA_DIR / ZIP_FILENAME

    # Check for zip file if CSV doesn't exist directly
    if not target_csv.exists() and target_zip.exists():
        console.print(f"[cyan]Unzipping {target_zip}...[/cyan]")
        with zipfile.ZipFile(target_zip, 'r') as zip_ref:
            zip_ref.extract(CSV_FILENAME, str(DATA_DIR))

    if not target_csv.exists():
        console.print(f"[bold red]File not found: {target_csv}[/bold red]")
        console.print("Please place [bold green]twcs.csv[/bold green] into the [bold cyan]data/[/bold cyan] directory.")
        return

    console.print(f"[bold cyan]Querying {target_csv} using DuckDB (zero RAM overhead)...[/bold cyan]")

    query = f"""
    COPY (
        SELECT 
            inbound.tweet_id AS tweet_id,
            inbound.author_id AS customer_author_id,
            inbound.text AS customer_text,
            outbound.text AS agent_reply,
            'OS_SOFTWARE_TROUBLESHOOTING' AS intent
        FROM '{target_csv}' AS inbound
        JOIN '{target_csv}' AS outbound 
          ON inbound.tweet_id = outbound.in_reply_to_tweet_id
        WHERE outbound.author_id = 'AppleSupport'
          AND inbound.inbound = true
          AND length(inbound.text) > 15
          AND length(outbound.text) > 15
        LIMIT {max_pairs}
    ) TO '{EXTRACTED_PAIRS_PATH}' (FORMAT JSON);
    """

    con = duckdb.connect()
    con.execute(query)
    con.close()

    console.print(f"[bold green]Successfully extracted {max_pairs} @AppleSupport resolution pairs to {EXTRACTED_PAIRS_PATH}![/bold green]")

    if index_chroma:
        console.print("[bold cyan]Indexing extracted Kaggle pairs into ChromaDB...[/bold cyan]")
        import json
        with open(EXTRACTED_PAIRS_PATH, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        vs = HistoricalVectorStore()
        # Batch index in chunks of 500
        chunk_size = 500
        for i in range(0, min(len(records), 2000), chunk_size):
            chunk = records[i:i + chunk_size]
            vs.index_records(chunk)
            console.print(f"Indexed records {i} to {i + len(chunk)}...")

        console.print(f"[bold green]Successfully indexed real Kaggle pairs into ChromaDB![/bold green]")


@app.command()
def download():
    """Downloads the Kaggle dataset directly."""
    download_dataset()


if __name__ == "__main__":
    app()
