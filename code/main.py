import sys
import subprocess
import os
import argparse

from dotenv import load_dotenv

# ==============================================================================
# AGENT AUTONOMY STAGE 1: Self-Healing Dependencies
# ==============================================================================
def install_dependencies():
    print("\n[Agent Autonomy] Missing dependencies detected. Installing automatically...")
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
        print("[Agent Autonomy] Dependencies installed successfully. Restarting agent...\n")
        # Restart the script
        os.execv(sys.executable, ['python'] + sys.argv)
    except Exception as e:
        print(f"\n[Fatal Error] Failed to auto-install dependencies: {e}")
        print(f"Please run 'pip install -r {req_path}' manually.")
        sys.exit(1)

try:
    import pandas as pd
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    # We import agent here so its external dependencies (like faiss) trigger the ImportError block
    from agent import process_ticket
except ImportError:
    install_dependencies()

console = Console()
HR_GREEN = "#00EA64"


def preflight_checks(input_path):
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    errors = []
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        errors.append("OPENAI_API_KEY is not set in .env or the current environment.")

    index_path = os.path.join(os.path.dirname(__file__), "faiss_index.bin")
    meta_path = os.path.join(os.path.dirname(__file__), "faiss_meta.json")
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        errors.append("FAISS artifacts are missing. Run 'python ingest.py' from the code/ directory first.")

    if not os.path.exists(input_path):
        errors.append(f"Input file not found at {input_path}")

    if errors:
        console.print("\n[bold red]Startup check failed:[/bold red]")
        for error in errors:
            console.print(f"[bold red]- {error}[/bold red]")
        return False

    console.print("[bold green]Startup check passed:[/bold green] dependencies, .env, and FAISS artifacts are ready.")
    return True

# ==============================================================================
# TERMINAL USER INTERFACE
# ==============================================================================
def display_banner():
    console.print(Panel.fit(
        f"[{HR_GREEN} bold]HackerRank Orchestrate - AI Triage Agent[/{HR_GREEN} bold]\n[white]Terminal User Interface (TUI)[/white]",
        border_style=HR_GREEN,
        padding=(1, 5)
    ))

def interactive_mode():
    os.system('cls' if os.name == 'nt' else 'clear')
    display_banner()
    console.print(f"\n[{HR_GREEN} bold]=== Interactive Chatbot Mode ===[/{HR_GREEN} bold]")
    console.print("[dim]Type your subject and issue to simulate a live support ticket.[/dim]\n")
    
    dev_mode = Confirm.ask(f"[{HR_GREEN}]Enable Dev Mode (shows internal routing/justification)?[/{HR_GREEN}]", default=True)
    
    while True:
        issue = Prompt.ask("\n[bold white]Describe your issue (or type 'exit' to quit)[/bold white]")
        if issue.lower() in ['exit', 'quit']:
            break

        subject = Prompt.ask("[bold white]Subject[/bold white]", default="General inquiry")
            
        company = Prompt.ask("[bold white]Which product? (HackerRank/Claude/Visa/None)[/bold white]", default="None")
        
        console.print()
        with console.status(f"[{HR_GREEN}]Agent is analyzing corpus and routing ticket...[/{HR_GREEN}]", spinner="dots"):
            result = process_ticket(issue, subject, company)
            
        # Display Response
        console.print(Panel(
            result["response"],
            title=f"[{HR_GREEN}]Agent Response[/{HR_GREEN}]",
            border_style=HR_GREEN,
            padding=(1, 2)
        ))
        
        if dev_mode:
            table = Table(title="[dim]Internal Agent State[/dim]", show_header=True, header_style=f"bold {HR_GREEN}")
            table.add_column("Key", style="dim", width=20)
            table.add_column("Value")
            
            table.add_row("Status", f"[bold]{result['status']}[/bold]")
            table.add_row("Request Type", result["request_type"])
            table.add_row("Product Area", result["product_area"])
            table.add_row("Justification", result["justification"])
            
            console.print(table)

def batch_mode(input_path, output_path):
    console.print(f"\n[{HR_GREEN} bold]=== Dataset Batch Processing ===[/{HR_GREEN} bold]")
    
    if not os.path.exists(input_path):
        console.print(f"[bold red]Error:[/bold red] Input file not found at {input_path}")
        return

    console.print(f"Reading tickets from [cyan]{input_path}[/cyan]...")
    df = pd.read_csv(input_path)
    df.columns = [str(c).lower().strip() for c in df.columns]

    statuses, product_areas, responses, justifications, request_types = [], [], [], [], []

    with Progress(
        SpinnerColumn(style=HR_GREEN),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style=HR_GREEN, finished_style=HR_GREEN),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[{HR_GREEN}]Processing {len(df)} tickets...", total=len(df))

        for idx, row in df.iterrows():
            issue = str(row.get("issue", ""))
            subject = str(row.get("subject", ""))
            company = str(row.get("company", "none"))

            result = process_ticket(issue, subject, company)

            statuses.append(result["status"])
            product_areas.append(result["product_area"])
            responses.append(result["response"])
            justifications.append(result["justification"])
            request_types.append(result["request_type"])

            progress.update(task, advance=1)

    df["status"] = statuses
    df["product_area"] = product_areas
    df["response"] = responses
    df["justification"] = justifications
    df["request_type"] = request_types

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    console.print(f"\n[{HR_GREEN} bold]Success![/{HR_GREEN} bold] Predictions saved to [cyan]{output_path}[/cyan]")

def main():
    parser = argparse.ArgumentParser(description="HackerRank Support Agent")
    parser.add_argument("--input", type=str, default="support_tickets/support_tickets.csv", help="Path to input CSV")
    parser.add_argument("--output", type=str, default="support_tickets/output.csv", help="Path to output CSV")
    parser.add_argument("--ui", action="store_true", help="Launch the interactive Terminal User Interface (TUI)")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not preflight_checks(input_path):
        sys.exit(1)

    if not args.ui:
        display_banner()
        batch_mode(input_path, output_path)
        return

    os.system('cls' if os.name == 'nt' else 'clear')
    display_banner()
    
    console.print("\n[bold white]Select an option:[/bold white]")
    console.print(f"[{HR_GREEN}]1.[/{HR_GREEN}] Interactive Chatbot Mode (with Dev Mode)")
    console.print(f"[{HR_GREEN}]2.[/{HR_GREEN}] Batch Process Dataset (evaluates {os.path.basename(input_path)})")
    console.print(f"[{HR_GREEN}]3.[/{HR_GREEN}] Exit")
    
    choice = Prompt.ask("\n[bold white]Choice[/bold white]", choices=["1", "2", "3"], default="1")
    
    if choice == "1":
        interactive_mode()
    elif choice == "2":
        batch_mode(input_path, output_path)
    else:
        console.print(f"[{HR_GREEN}]Goodbye![/{HR_GREEN}]")

if __name__ == "__main__":
    main()
