import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(16, 20))
ax.set_xlim(0, 16)
ax.set_ylim(0, 20)
ax.axis('off')
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#0D1117')

def box(ax, x, y, w, h, label, sublabel='', color='#1F6FEB', text_color='white', radius=0.3):
    fancy = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.1",
                           facecolor=color, edgecolor='#30363D', linewidth=1.5)
    ax.add_patch(fancy)
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.18, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color=text_color, fontfamily='monospace')
        ax.text(x + w/2, y + h/2 - 0.22, sublabel, ha='center', va='center',
                fontsize=8.5, color='#8B949E', fontfamily='monospace')
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color=text_color, fontfamily='monospace')

def arrow(ax, x1, y1, x2, y2, color='#58A6FF'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0))

def section_label(ax, x, y, text, color='#58A6FF'):
    ax.text(x, y, text, fontsize=13, fontweight='bold', color=color,
            fontfamily='monospace', va='center')

def divider(ax, y):
    ax.axhline(y=y, color='#30363D', linewidth=1, linestyle='--', xmin=0.02, xmax=0.98)


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(8, 19.3, 'RAG Project — Architecture', ha='center', va='center',
        fontsize=18, fontweight='bold', color='white', fontfamily='monospace')
ax.text(8, 18.85, 'LangChain  •  ChromaDB  •  Ollama  •  Streamlit', ha='center', va='center',
        fontsize=10, color='#8B949E', fontfamily='monospace')

# ── PHASE 1: INDEXING ─────────────────────────────────────────────────────────
divider(ax, 18.5)
section_label(ax, 0.4, 18.1, '  PHASE 1 — INDEXING  (ingest.py)')

# Row 1
box(ax, 0.4, 16.7, 2.8, 1.0, 'PDF / TXT', '.docx files', color='#161B22')
box(ax, 4.0, 16.7, 2.8, 1.0, 'PyPDFLoader', 'langchain-community', color='#1C2E4A')
box(ax, 7.6, 16.7, 2.8, 1.0, 'Text Splitter', '500 chars / 75 overlap', color='#1C2E4A')
box(ax, 11.2, 16.7, 2.8, 1.0, 'Chunks', '~500 char pieces', color='#161B22')

arrow(ax, 3.2, 17.2, 4.0, 17.2)
arrow(ax, 6.8, 17.2, 7.6, 17.2)
arrow(ax, 10.4, 17.2, 11.2, 17.2)

# Row 2
box(ax, 5.8, 15.2, 4.4, 1.0, 'all-MiniLM-L6-v2', 'sentence-transformers  •  384-dim vectors', color='#2D1B69')
box(ax, 11.2, 15.2, 2.8, 1.0, 'ChromaDB', 'chroma_db/ (local)', color='#1A3A1A')

arrow(ax, 12.6, 16.7, 12.6, 16.2)
arrow(ax, 8.0, 16.7, 8.0, 16.2)
arrow(ax, 10.2, 15.7, 11.2, 15.7)

ax.text(8.0, 14.85, 'Vectors stored permanently — run once, reuse forever',
        ha='center', fontsize=9, color='#8B949E', fontfamily='monospace', style='italic')

# ── PHASE 2: QUERYING ─────────────────────────────────────────────────────────
divider(ax, 14.5)
section_label(ax, 0.4, 14.1, '  PHASE 2 — QUERYING  (query.py  or  app.py)')

# User input
box(ax, 5.8, 13.0, 4.4, 0.9, 'User Question', 'typed in CLI or Streamlit UI', color='#161B22')

arrow(ax, 8.0, 13.0, 8.0, 12.4)

# Embed question
box(ax, 5.8, 11.5, 4.4, 0.9, 'all-MiniLM-L6-v2', 'embed question → vector', color='#2D1B69')

arrow(ax, 8.0, 11.5, 8.0, 10.9)

# Vector search
box(ax, 5.8, 10.0, 4.4, 0.9, 'ChromaDB', 'cosine similarity search  →  Top 4 chunks', color='#1A3A1A')

arrow(ax, 8.0, 10.0, 8.0, 9.4)

# Prompt
box(ax, 5.8, 8.5, 4.4, 0.9, 'Prompt Template', '"Answer ONLY from context. If unknown, say so."', color='#3A2000')

arrow(ax, 8.0, 8.5, 8.0, 7.9)

# LLM
box(ax, 5.8, 7.0, 4.4, 0.9, 'llama3.2  (Ollama)', 'local LLM  •  no API key  •  temperature=0', color='#1A0A2E')

arrow(ax, 8.0, 7.0, 8.0, 6.4)

# Output split
box(ax, 2.6, 5.3, 4.4, 0.9, 'query.py', 'Answer + sources in terminal', color='#1C2E4A')
box(ax, 9.0, 5.3, 4.4, 0.9, 'app.py  (Streamlit)', 'Chat UI in browser  :8501', color='#1C2E4A')

ax.annotate('', xy=(4.8, 6.2), xytext=(7.5, 6.4),
            arrowprops=dict(arrowstyle='->', color='#58A6FF', lw=2.0))
ax.annotate('', xy=(11.2, 6.2), xytext=(8.5, 6.4),
            arrowprops=dict(arrowstyle='->', color='#58A6FF', lw=2.0))

# ── TECH STACK FOOTER ─────────────────────────────────────────────────────────
divider(ax, 4.8)
ax.text(8, 4.4, 'TECH STACK', ha='center', fontsize=11, fontweight='bold',
        color='#58A6FF', fontfamily='monospace')

tools = [
    ('LangChain', '#1C2E4A'),
    ('ChromaDB', '#1A3A1A'),
    ('Ollama', '#1A0A2E'),
    ('Sentence\nTransformers', '#2D1B69'),
    ('Streamlit', '#3A2000'),
    ('pypdf', '#161B22'),
]
x_start = 0.5
for i, (name, color) in enumerate(tools):
    bx = x_start + i * 2.55
    box(ax, bx, 3.1, 2.3, 0.9, name, color=color)

# ── Footer ────────────────────────────────────────────────────────────────────
ax.text(8, 2.7, 'github.com/kranthi-prog/RAG-project', ha='center',
        fontsize=9, color='#8B949E', fontfamily='monospace')
ax.text(8, 2.35, 'Built with Claude Code  •  MIT License  •  May 2026', ha='center',
        fontsize=9, color='#8B949E', fontfamily='monospace', style='italic')

plt.tight_layout()
plt.savefig('/home/user/RAG-project/architecture.png', dpi=180, bbox_inches='tight',
            facecolor='#0D1117', edgecolor='none')
print("Saved architecture.png")
