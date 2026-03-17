"""Click-based CLI entry point."""

import os

import click
from dotenv import load_dotenv

load_dotenv()


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


@click.group()
def cli():
    """Search dashcam footage using natural language queries."""


# -----------------------------------------------------------------------
# index
# -----------------------------------------------------------------------

@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--chunk-duration", default=30, show_default=True,
              help="Chunk duration in seconds.")
@click.option("--overlap", default=5, show_default=True,
              help="Overlap between chunks in seconds.")
def index(directory, chunk_duration, overlap):
    """Index mp4 files in DIRECTORY for searching."""
    from .chunker import chunk_video, scan_directory
    from .embedder import embed_video_chunk
    from .store import SentryStore

    if os.path.isfile(directory):
        videos = [os.path.abspath(directory)]
    else:
        videos = scan_directory(directory)

    if not videos:
        click.echo("No mp4 files found.")
        return

    store = SentryStore()
    total_files = len(videos)
    new_files = 0
    new_chunks = 0

    for file_idx, video_path in enumerate(videos, 1):
        abs_path = os.path.abspath(video_path)
        basename = os.path.basename(video_path)

        if store.is_indexed(abs_path):
            click.echo(f"Skipping ({file_idx}/{total_files}): {basename} (already indexed)")
            continue

        chunks = chunk_video(abs_path, chunk_duration=chunk_duration, overlap=overlap)
        num_chunks = len(chunks)
        embedded = []

        for chunk_idx, chunk in enumerate(chunks, 1):
            click.echo(
                f"Indexing file {file_idx}/{total_files}: {basename} "
                f"[chunk {chunk_idx}/{num_chunks}]"
            )
            embedding = embed_video_chunk(chunk["chunk_path"])
            embedded.append({**chunk, "embedding": embedding})

        store.add_chunks(embedded)
        new_files += 1
        new_chunks += len(embedded)

    stats = store.get_stats()
    click.echo(
        f"\nIndexed {new_chunks} new chunks from {new_files} files. "
        f"Total: {stats['total_chunks']} chunks from "
        f"{stats['unique_source_files']} files."
    )


# -----------------------------------------------------------------------
# search
# -----------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("-n", "--results", "n_results", default=5, show_default=True,
              help="Number of results to return.")
@click.option("-o", "--output-dir", default=".", show_default=True,
              help="Directory to save trimmed clips.")
@click.option("--trim/--no-trim", default=True, show_default=True,
              help="Auto-trim the top result.")
def search(query, n_results, output_dir, trim):
    """Search indexed footage with a natural language QUERY."""
    from .search import search_footage
    from .store import SentryStore

    store = SentryStore()
    results = search_footage(query, store, n_results=n_results)

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        basename = os.path.basename(r["source_file"])
        start_str = _fmt_time(r["start_time"])
        end_str = _fmt_time(r["end_time"])
        click.echo(
            f"  #{i} [{r['similarity_score']:.2f}] {basename} "
            f"@ {start_str}-{end_str}"
        )

    if trim:
        from .trimmer import trim_top_result
        clip_path = trim_top_result(results, output_dir)
        click.echo(f"\nSaved clip: {clip_path}")


# -----------------------------------------------------------------------
# stats
# -----------------------------------------------------------------------

@cli.command()
def stats():
    """Print index statistics."""
    from .store import SentryStore

    store = SentryStore()
    s = store.get_stats()

    if s["total_chunks"] == 0:
        click.echo("Index is empty. Run `sentrysearch index <directory>` first.")
        return

    click.echo(f"Total chunks:  {s['total_chunks']}")
    click.echo(f"Source files:  {s['unique_source_files']}")
    click.echo("\nIndexed files:")
    for f in s["source_files"]:
        click.echo(f"  {f}")
