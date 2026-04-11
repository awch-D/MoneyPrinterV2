import os
import random
import zipfile
import requests

from status import *
from config import *

DEFAULT_SONG_ARCHIVE_URLS = []

_LOCAL_AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")


def _local_songs_dirs() -> list[str]:
    """Project dirs that may hold background music (case-sensitive FS may have `songs` vs `Songs`)."""
    out: list[str] = []
    seen: set[str] = set()
    for name in ("Songs", "songs"):
        path = os.path.join(ROOT_DIR, name)
        if not os.path.isdir(path):
            continue
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        out.append(path)
    return out


def _dir_has_audio_files(path: str) -> bool:
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isfile(p) and name.lower().endswith(_LOCAL_AUDIO_EXTS):
            return True
    return False


def rem_temp_files() -> None:
    """
    Removes temporary files in the `.mp` directory.

    Returns:
        None
    """
    # Path to the `.mp` directory
    mp_dir = os.path.join(ROOT_DIR, ".mp")

    files = os.listdir(mp_dir)

    for file in files:
        if not file.endswith(".json"):
            os.remove(os.path.join(mp_dir, file))


def fetch_songs() -> None:
    """
    Downloads songs into songs/ directory to use with geneated videos.

    Returns:
        None
    """
    try:
        info(f" => Fetching songs...")

        # If user already placed audio under Songs/ or songs/, skip network — do not look "stuck" on this step.
        for d in _local_songs_dirs():
            if _dir_has_audio_files(d):
                if get_verbose():
                    info(f" => Using local audio in {os.path.basename(d)}/ ({d}), skipping download.")
                return

        files_dir = os.path.join(ROOT_DIR, "Songs")
        if not os.path.exists(files_dir):
            os.mkdir(files_dir)
            if get_verbose():
                info(f" => Created directory: {files_dir}")

        configured_url = get_zip_url().strip()
        download_urls = [configured_url] if configured_url else []
        download_urls.extend(DEFAULT_SONG_ARCHIVE_URLS)

        if len(download_urls) == 0:
            if get_verbose():
                warning("No songs archive URL configured. Continuing without background music.")
            return

        archive_path = os.path.join(files_dir, "songs.zip")
        downloaded = False

        for download_url in download_urls:
            try:
                response = requests.get(download_url, timeout=60)
                response.raise_for_status()

                with open(archive_path, "wb") as file:
                    file.write(response.content)

                SAFE_EXTENSIONS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")
                with zipfile.ZipFile(archive_path, "r") as zf:
                    for member in zf.namelist():
                        basename = os.path.basename(member)
                        if not basename or not basename.lower().endswith(SAFE_EXTENSIONS):
                            warning(f"Skipping non-audio file in archive: {member}")
                            continue
                        if ".." in member or member.startswith("/"):
                            warning(f"Skipping suspicious path in archive: {member}")
                            continue
                        zf.extract(member, files_dir)

                downloaded = True
                break
            except Exception as err:
                warning(f"Failed to fetch songs from {download_url}: {err}")

        if not downloaded:
            raise RuntimeError(
                "Could not download a valid songs archive from any configured URL"
            )

        # Remove the zip file
        if os.path.exists(archive_path):
            os.remove(archive_path)

        success(" => Downloaded songs into Songs/.")

    except Exception as e:
        error(f"Error occurred while fetching songs: {str(e)}")


def choose_random_song() -> str:
    """
    Chooses a random song from the songs/ directory.

    Returns:
        str: The path to the chosen song.
    """
    try:
        songs: list[tuple[str, str]] = []
        for songs_dir in _local_songs_dirs():
            for name in os.listdir(songs_dir):
                path = os.path.join(songs_dir, name)
                if os.path.isfile(path) and name.lower().endswith(_LOCAL_AUDIO_EXTS):
                    songs.append((songs_dir, name))
        if len(songs) == 0:
            warning("No audio files found in Songs/ or songs/. Background music will be skipped.")
            return ""
        songs_dir, song = random.choice(songs)
        success(f" => Chose song: {song}")
        return os.path.join(songs_dir, song)
    except Exception as e:
        error(f"Error occurred while choosing random song: {str(e)}")
        return ""
