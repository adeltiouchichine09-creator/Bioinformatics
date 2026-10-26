"""
Codon Frequency Analyzer
------------------------
Reads a FASTA file (one viral or bacterial genome), slides a 3-nucleotide
window over the sequence, counts every codon (triplet) seen, shows the top 10
most frequent codons, translates each one DNA -> RNA -> amino acid, and then
asks an AI which foods are rich in those amino acids.

Usage:
    python codon_app.py                  # GUI
    python codon_app.py genome.fasta     # GUI, file preloaded
    python codon_app.py genome.fasta --cli [--step 3]   # console only

Optional AI step: set ANTHROPIC_API_KEY (pip install anthropic).
Without a key, the app falls back to a built-in nutrition knowledge base.
"""

import os
import sys
import argparse
from collections import Counter

# ---------------------------------------------------------------------------
# 1. Genetic code (RNA codons -> amino acid)
# ---------------------------------------------------------------------------
BASES = "UCAG"
AA_STRING = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = {}
_i = 0
for b1 in BASES:
    for b2 in BASES:
        for b3 in BASES:
            CODON_TABLE[b1 + b2 + b3] = AA_STRING[_i]
            _i += 1

AA_NAMES = {
    "A": "Alanine", "R": "Arginine", "N": "Asparagine", "D": "Aspartic acid",
    "C": "Cysteine", "Q": "Glutamine", "E": "Glutamic acid", "G": "Glycine",
    "H": "Histidine", "I": "Isoleucine", "L": "Leucine", "K": "Lysine",
    "M": "Methionine", "F": "Phenylalanine", "P": "Proline", "S": "Serine",
    "T": "Threonine", "W": "Tryptophan", "Y": "Tyrosine", "V": "Valine",
    "*": "Stop codon",
}

# Fallback knowledge base used when no AI key is available
AA_FOODS = {
    "Alanine": "meat, poultry, fish, eggs, dairy, soy",
    "Arginine": "turkey, chicken, pumpkin seeds, peanuts, soybeans, lentils",
    "Asparagine": "asparagus, dairy, eggs, potatoes, legumes, nuts",
    "Aspartic acid": "oysters, meat, sprouting seeds, avocado, sugarcane",
    "Cysteine": "eggs, poultry, garlic, onions, broccoli, oats",
    "Glutamine": "beef, chicken, fish, eggs, tofu, cabbage, dairy",
    "Glutamic acid": "cheese (parmesan), tomatoes, soy sauce, mushrooms, meat",
    "Glycine": "gelatin, bone broth, pork skin, chicken, fish",
    "Histidine": "meat, fish, poultry, whole grains, buckwheat",
    "Isoleucine": "chicken, eggs, fish, lentils, cheese, soy",
    "Leucine": "whey/dairy, beef, chicken, fish, soybeans, peanuts",
    "Lysine": "meat, fish, eggs, cheese, quinoa, beans, lentils",
    "Methionine": "eggs, fish, sesame seeds, Brazil nuts, meat",
    "Phenylalanine": "meat, fish, eggs, dairy, soy, nuts",
    "Proline": "gelatin, bone broth, dairy, cabbage, asparagus",
    "Serine": "soybeans, eggs, peanuts, lentils, meat, fish",
    "Threonine": "cottage cheese, poultry, fish, lentils, sesame seeds",
    "Tryptophan": "turkey, eggs, cheese, oats, bananas, pumpkin seeds",
    "Tyrosine": "cheese, chicken, turkey, almonds, avocado, bananas",
    "Valine": "cheese, meat, mushrooms, soy, peanuts, whole grains",
    "Stop codon": "n/a - stop codons do not encode an amino acid",
}

ESSENTIAL = {"Histidine", "Isoleucine", "Leucine", "Lysine", "Methionine",
             "Phenylalanine", "Threonine", "Tryptophan", "Valine"}


# ---------------------------------------------------------------------------
# 2. Core logic
# ---------------------------------------------------------------------------
def read_fasta(path):
    """Return (header, sequence) of the first record in a FASTA file."""
    header, chunks = "", []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:          # only one genome expected; stop at 2nd record
                    break
                header = line[1:]
            else:
                chunks.append(line.upper())
    seq = "".join(chunks)
    if not seq:
        raise ValueError("No sequence found in file.")
    return header or os.path.basename(path), seq


def count_codons(seq, step=1):
    """Slide a window of length 3 over the sequence and count each triplet.
    step=1 -> overlapping window (every position); step=3 -> reading frame 1.
    Windows containing anything other than A/C/G/T (e.g. N) are skipped."""
    counts = Counter()
    valid = set("ACGT")
    for i in range(0, len(seq) - 2, step):
        codon = seq[i:i + 3]
        if set(codon) <= valid:
            counts[codon] += 1
    return counts


def dna_to_rna(codon):
    return codon.replace("T", "U")


def translate(rna_codon):
    letter = CODON_TABLE[rna_codon]
    return letter, AA_NAMES[letter]


def top_codons(counts, n=10):
    total = sum(counts.values())
    rows = []
    for rank, (codon, c) in enumerate(counts.most_common(n), 1):
        rna = dna_to_rna(codon)
        letter, name = translate(rna)
        rows.append({
            "rank": rank, "dna": codon, "rna": rna, "aa_letter": letter,
            "aa": name, "count": c, "freq": 100.0 * c / total if total else 0,
        })
    return rows, total


# ---------------------------------------------------------------------------
# 3. AI step: which foods are rich in these amino acids?
# ---------------------------------------------------------------------------
def ask_ai_for_foods(amino_acids):
    amino_acids = [a for a in dict.fromkeys(amino_acids) if a != "Stop codon"]
    question = ("For each of these amino acids, list the foods that are the best "
                "dietary sources of it (short list per amino acid, mark if it is "
                "essential): " + ", ".join(amino_acids))
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            msg = client.messages.create(
                model="claude-sonnet-5-5", max_tokens=1200,
                messages=[{"role": "user", "content": question}],
            )
            answer = "".join(b.text for b in msg.content if hasattr(b, "text"))
            return "Q: " + question + "\n\nAI answer:\n" + answer
        except Exception as exc:  # network / package / key problems
            fallback_note = f"(AI request failed: {exc}; using built-in data)\n\n"
    else:
        fallback_note = "(No ANTHROPIC_API_KEY set; using built-in data)\n\n"

    lines = [fallback_note + "Q: " + question + "\n"]
    for aa in amino_acids:
        tag = " [essential]" if aa in ESSENTIAL else ""
        lines.append(f"- {aa}{tag}: {AA_FOODS.get(aa, 'unknown')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Console interface
# ---------------------------------------------------------------------------
def run_cli(path, step):
    header, seq = read_fasta(path)
    counts = count_codons(seq, step)
    rows, total = top_codons(counts)
    print(f"Genome : {header}")
    print(f"Length : {len(seq):,} nt | windows counted: {total:,} | step: {step}\n")
    print(f"{'#':>2}  {'DNA':<4} {'RNA':<4} {'AA':<18} {'Count':>9} {'Freq %':>7}")
    print("-" * 50)
    for r in rows:
        print(f"{r['rank']:>2}  {r['dna']:<4} {r['rna']:<4} "
              f"{r['aa'] + ' (' + r['aa_letter'] + ')':<18} "
              f"{r['count']:>9,} {r['freq']:>7.3f}")
    print("\n" + ask_ai_for_foods([r["aa"] for r in rows]))


# ---------------------------------------------------------------------------
# 5. Graphical interface (tkinter + matplotlib)
# ---------------------------------------------------------------------------
def run_gui(initial_path=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("Codon Frequency Analyzer")
    root.geometry("1050x760")
    state = {"rows": []}

    # --- top bar
    bar = ttk.Frame(root, padding=8)
    bar.pack(fill="x")
    ttk.Button(bar, text="Open FASTA...", command=lambda: open_file()).pack(side="left")
    ttk.Label(bar, text="  Window step:").pack(side="left")
    step_var = tk.IntVar(value=1)
    ttk.Radiobutton(bar, text="1 (sliding)", variable=step_var, value=1).pack(side="left")
    ttk.Radiobutton(bar, text="3 (reading frame)", variable=step_var, value=3).pack(side="left")
    ai_btn = ttk.Button(bar, text="Ask AI: foods for these amino acids",
                        state="disabled", command=lambda: ask_ai())
    ai_btn.pack(side="right")
    info = ttk.Label(root, text="Open a FASTA file to begin.", padding=(8, 0))
    info.pack(fill="x")

    # --- middle: table + chart
    mid = ttk.Frame(root, padding=8)
    mid.pack(fill="both", expand=True)
    cols = ("rank", "dna", "rna", "aa", "count", "freq")
    heads = ("#", "DNA codon", "RNA codon", "Amino acid", "Count", "Freq %")
    tree = ttk.Treeview(mid, columns=cols, show="headings", height=10)
    for c, h, w in zip(cols, heads, (35, 80, 80, 150, 90, 70)):
        tree.heading(c, text=h)
        tree.column(c, width=w, anchor="center")
    tree.pack(side="left", fill="y")

    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=mid)
    canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

    # --- bottom: AI answer
    ttk.Label(root, text="AI answer:", padding=(8, 0)).pack(anchor="w")
    out = scrolledtext.ScrolledText(root, height=12, wrap="word")
    out.pack(fill="both", expand=False, padx=8, pady=(0, 8))

    def analyze(path):
        try:
            header, seq = read_fasta(path)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return
        counts = count_codons(seq, step_var.get())
        rows, total = top_codons(counts)
        state["rows"] = rows
        info.config(text=f"{header[:90]}  |  {len(seq):,} nt  |  "
                         f"{total:,} windows  |  {len(counts)} distinct codons")
        tree.delete(*tree.get_children())
        for r in rows:
            tree.insert("", "end", values=(r["rank"], r["dna"], r["rna"],
                        f"{r['aa']} ({r['aa_letter']})", f"{r['count']:,}",
                        f"{r['freq']:.3f}"))
        ax.clear()
        labels = [f"{r['dna']}\n{r['aa_letter']}" for r in rows]
        ax.bar(labels, [r["count"] for r in rows], color="#3b7dd8")
        ax.set_title("Top 10 codons (codon / amino acid)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        canvas.draw()
        ai_btn.config(state="normal")
        out.delete("1.0", "end")

    def open_file():
        path = filedialog.askopenfilename(
            filetypes=[("FASTA", "*.fasta *.fa *.fna *.fas *.txt"), ("All", "*.*")])
        if path:
            analyze(path)

    def ask_ai():
        out.delete("1.0", "end")
        out.insert("end", "Asking the AI...\n")
        root.update_idletasks()
        answer = ask_ai_for_foods([r["aa"] for r in state["rows"]])
        out.delete("1.0", "end")
        out.insert("end", answer)

    if initial_path:
        root.after(100, lambda: analyze(initial_path))
    root.mainloop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Codon frequency analyzer")
    p.add_argument("fasta", nargs="?", help="FASTA file with one genome")
    p.add_argument("--cli", action="store_true", help="console output only")
    p.add_argument("--step", type=int, default=1, choices=(1, 3),
                   help="window step: 1 = sliding (default), 3 = reading frame")
    a = p.parse_args()
    if a.cli:
        if not a.fasta:
            sys.exit("Provide a FASTA file for --cli mode.")
        run_cli(a.fasta, a.step)
    else:
        run_gui(a.fasta)