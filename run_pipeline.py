import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    load_dotenv()
    project_root = Path(__file__).parent

    run([sys.executable, str(project_root / "fetch_jira_data.py")])
    run([sys.executable, str(project_root / "analyze_similarity.py")])
    run([sys.executable, str(project_root / "summarize_clusters.py")])


if __name__ == "__main__":
    main()



