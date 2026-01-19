"""
Train a small LLM from scratch on Wikipedia with GUI
Requirements: pip install torch transformers datasets tokenizers requests tqdm
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import math
from tqdm import tqdm
import os
import requests
import re
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model architecture (SMALL - fits GTX 1650 4GB)
    vocab_size = 16000      # Reduced vocab
    d_model = 256           # Reduced from 768
    n_heads = 4             # Reduced from 12
    n_layers = 6            # Reduced from 12
    d_ff = 1024             # Reduced from 3072
    max_seq_len = 256       # Reduced from 512
    dropout = 0.1
    
    # Training
    batch_size = 2          # Very small batch for 4GB VRAM
    learning_rate = 3e-4
    weight_decay = 0.01
    max_steps = 50000       # Reduced
    warmup_steps = 1000     # Reduced
    save_every = 2500
    
    # Paths
    output_dir = "./llm_output"
    tokenizer_path = "./tokenizer"
    articles_cache = "./wiki_articles.txt"

config = Config()

# ============================================================================
# WIKIPEDIA FETCHER
# ============================================================================

class WikipediaFetcher:
    """Fetch and cache Wikipedia articles"""
    
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.article_count = 0
        
    def fetch_random_articles(self, count=10):
        """Fetch random Wikipedia article titles"""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'random',
                'rnnamespace': 0,
                'rnlimit': count
            }
            
            response = requests.get(url, params=params, timeout=10,
                                   headers={'User-Agent': 'LLMTrainer/1.0'})
            data = response.json()
            return [page['title'] for page in data['query']['random']]
        except Exception as e:
            return []
    
    def fetch_article_text(self, title):
        """Fetch article content"""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'explaintext': True,
                'exsectionformat': 'plain'
            }
            
            response = requests.get(url, params=params, timeout=15,
                                   headers={'User-Agent': 'LLMTrainer/1.0'})
            data = response.json()
            
            pages = data['query']['pages']
            for page_id in pages:
                if 'extract' in pages[page_id]:
                    return pages[page_id]['extract']
            return None
        except:
            return None
    
    def clean_text(self, text):
        """Clean Wikipedia text"""
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def download_articles(self, num_articles, callback=None, append=True):
        """Download articles and save to cache"""
        mode = 'a' if append else 'w'
        
        with open(self.cache_file, mode, encoding='utf-8') as f:
            batch_size = 10
            for i in range(0, num_articles, batch_size):
                titles = self.fetch_random_articles(batch_size)
                
                for title in titles:
                    text = self.fetch_article_text(title)
                    if text and len(text) > 100:
                        cleaned = self.clean_text(text)
                        f.write(cleaned + "\n\n")
                        f.flush()
                        self.article_count += 1
                        
                        if callback:
                            callback(f"Downloaded: {title} ({self.article_count}/{num_articles})")
                
                time.sleep(0.5)
        
        if callback:
            callback(f"✅ Downloaded {self.article_count} articles!")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        bs, seq_len, d_model = x.shape
        
        q = self.q_linear(x).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bs, seq_len, d_model)
        return self.out(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.token_embedding.weight = self.head.weight
        
    def forward(self, x, labels=None):
        bs, seq_len = x.shape
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss

# ============================================================================
# DATASET
# ============================================================================

class WikiDataset(Dataset):
    def __init__(self, text_file, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokens = tokenizer(text, truncation=False, add_special_tokens=False)['input_ids']
        
        self.chunks = []
        for i in range(0, len(tokens) - max_length, max_length // 2):
            chunk = tokens[i:i + max_length]
            if len(chunk) == max_length:
                self.chunks.append(chunk)
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return {
            'input_ids': torch.tensor(chunk[:-1]),
            'labels': torch.tensor(chunk[1:])
        }

# ============================================================================
# GUI
# ============================================================================

class LLMTrainerGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("LLM Training from Wikipedia")
        self.window.geometry("1000x700")
        self.window.configure(bg='#1e1e1e')
        
        self.fetcher = WikipediaFetcher(config.articles_cache)
        self.is_training = False
        self.is_downloading = False
        self.training_thread = None
        
        self._create_widgets()
        self._check_existing_data()
    
    def _create_widgets(self):
        # Title
        title_frame = tk.Frame(self.window, bg='#2d2d2d', pady=15)
        title_frame.pack(fill='x')
        
        tk.Label(title_frame, text="🧠 LLM Training from Wikipedia", 
                bg='#2d2d2d', fg='#00ff00', 
                font=('Consolas', 18, 'bold')).pack()
        
        # Stats frame
        stats_frame = tk.Frame(self.window, bg='#2d2d2d', pady=10)
        stats_frame.pack(fill='x', padx=10)
        
        self.stats_label = tk.Label(stats_frame, 
                                    text="Articles: 0 | Status: Idle", 
                                    bg='#2d2d2d', fg='#00aaff', 
                                    font=('Consolas', 11))
        self.stats_label.pack()
        
        # Main content - notebook
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Tab 1: Data Collection
        data_tab = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(data_tab, text='📥 Data Collection')
        
        tk.Label(data_tab, text="Download Wikipedia Articles", 
                bg='#1e1e1e', fg='#ffffff', 
                font=('Consolas', 12, 'bold')).pack(pady=10)
        
        articles_frame = tk.Frame(data_tab, bg='#1e1e1e')
        articles_frame.pack(pady=5)
        
        tk.Label(articles_frame, text="Number of articles:", 
                bg='#1e1e1e', fg='#ffffff', 
                font=('Consolas', 10)).pack(side='left', padx=5)
        
        self.articles_entry = tk.Entry(articles_frame, 
                                       bg='#2d2d2d', fg='#ffffff', 
                                       font=('Consolas', 10), width=10)
        self.articles_entry.insert(0, "100")
        self.articles_entry.pack(side='left', padx=5)
        
        self.download_btn = tk.Button(data_tab, text="📥 Download Articles", 
                                      command=self.download_articles,
                                      bg='#0066cc', fg='white', 
                                      font=('Consolas', 11, 'bold'),
                                      cursor='hand2', width=20)
        self.download_btn.pack(pady=10)
        
        # Tab 2: Training
        train_tab = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(train_tab, text='🚀 Training')
        
        tk.Label(train_tab, text="Model Training", 
                bg='#1e1e1e', fg='#ffffff', 
                font=('Consolas', 12, 'bold')).pack(pady=10)
        
        # Training params
        params_frame = tk.Frame(train_tab, bg='#1e1e1e')
        params_frame.pack(pady=10)
        
        tk.Label(params_frame, text="Batch Size:", 
                bg='#1e1e1e', fg='#ffffff', 
                font=('Consolas', 10)).grid(row=0, column=0, padx=5, pady=5)
        self.batch_entry = tk.Entry(params_frame, bg='#2d2d2d', fg='#ffffff', 
                                    font=('Consolas', 10), width=10)
        self.batch_entry.insert(0, "2")  # Safe default for 4GB GPU
        self.batch_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(params_frame, text="Max Steps:", 
                bg='#1e1e1e', fg='#ffffff', 
                font=('Consolas', 10)).grid(row=1, column=0, padx=5, pady=5)
        self.steps_entry = tk.Entry(params_frame, bg='#2d2d2d', fg='#ffffff', 
                                    font=('Consolas', 10), width=10)
        self.steps_entry.insert(0, "50000")
        self.steps_entry.grid(row=1, column=1, padx=5, pady=5)
        
        self.train_btn = tk.Button(train_tab, text="🚀 Start Training", 
                                   command=self.start_training,
                                   bg='#00aa00', fg='white', 
                                   font=('Consolas', 11, 'bold'),
                                   cursor='hand2', width=20)
        self.train_btn.pack(pady=10)
        
        self.stop_btn = tk.Button(train_tab, text="⏹ Stop Training", 
                                 command=self.stop_training,
                                 bg='#cc0000', fg='white', 
                                 font=('Consolas', 11, 'bold'),
                                 cursor='hand2', width=20, state='disabled')
        self.stop_btn.pack(pady=5)
        
        # Log display (shared)
        log_frame = tk.Frame(self.window, bg='#1e1e1e')
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        tk.Label(log_frame, text="📋 Log", 
                bg='#1e1e1e', fg='#ffffff', 
                font=('Consolas', 10, 'bold')).pack(anchor='w')
        
        self.log_display = scrolledtext.ScrolledText(log_frame, 
                                                     wrap=tk.WORD,
                                                     bg='#2d2d2d', fg='#00ff00',
                                                     font=('Consolas', 9),
                                                     height=15)
        self.log_display.pack(fill='both', expand=True)
        
        # Bottom buttons
        bottom_frame = tk.Frame(self.window, bg='#1e1e1e')
        bottom_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(bottom_frame, text="🗑️ Clear Log", 
                 command=self.clear_log,
                 bg='#cc6600', fg='white', 
                 font=('Consolas', 9, 'bold'),
                 cursor='hand2').pack(side='left', padx=2)
        
        tk.Button(bottom_frame, text="ℹ️ About", 
                 command=self.show_about,
                 bg='#6600cc', fg='white', 
                 font=('Consolas', 9, 'bold'),
                 cursor='hand2').pack(side='left', padx=2)
    
    def log(self, message):
        """Add message to log"""
        self.log_display.insert('end', f"{message}\n")
        self.log_display.see('end')
        self.window.update()
    
    def _check_existing_data(self):
        """Check for existing data"""
        if os.path.exists(config.articles_cache):
            with open(config.articles_cache, 'r', encoding='utf-8') as f:
                content = f.read()
                articles = content.count('\n\n')
            self.log(f"✅ Found {articles} cached articles")
            self.stats_label.config(text=f"Articles: {articles} | Status: Ready")
        else:
            self.log("💡 No cached articles. Download some to get started!")
    
    def download_articles(self):
        """Download articles in background"""
        if self.is_downloading:
            messagebox.showwarning("Warning", "Already downloading!")
            return
        
        try:
            num_articles = int(self.articles_entry.get())
            if num_articles <= 0:
                raise ValueError()
        except:
            messagebox.showerror("Error", "Enter a valid number of articles")
            return
        
        self.is_downloading = True
        self.download_btn.config(state='disabled')
        
        def download_thread():
            self.log(f"📥 Starting download of {num_articles} articles...")
            self.fetcher.download_articles(num_articles, self.log, 
                                          append=os.path.exists(config.articles_cache))
            self.is_downloading = False
            self.download_btn.config(state='normal')
            self._check_existing_data()
        
        threading.Thread(target=download_thread, daemon=True).start()
    
    def start_training(self):
        """Start training in background"""
        if self.is_training:
            messagebox.showwarning("Warning", "Already training!")
            return
        
        if not os.path.exists(config.articles_cache):
            messagebox.showerror("Error", "No articles downloaded! Use Data Collection tab first.")
            return
        
        try:
            config.batch_size = int(self.batch_entry.get())
            config.max_steps = int(self.steps_entry.get())
            
            # Validate parameters
            if config.batch_size < 1:
                raise ValueError("Batch size must be at least 1")
            if config.max_steps <= config.warmup_steps:
                raise ValueError(f"Max steps ({config.max_steps}) must be greater than warmup steps ({config.warmup_steps})")
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid training parameters: {str(e)}")
            return
        
        self.is_training = True
        self.train_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        self.training_thread = threading.Thread(target=self._train, daemon=True)
        self.training_thread.start()
    
    def stop_training(self):
        """Stop training"""
        if self.is_training:
            self.is_training = False
            self.stop_btn.config(state='disabled')
            self.log("⏹ Stopping training and saving current progress...")
        else:
            self.log("⚠️ Training is not currently running")
    
    def _train(self):
        """Training loop"""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log(f"🔧 Using device: {device}")
            
            # Tokenizer
            if os.path.exists(config.tokenizer_path):
                self.log("📖 Loading tokenizer...")
                tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_path)
            else:
                self.log("🔨 Training tokenizer...")
                tokenizer = Tokenizer(BPE(unk_token="<unk>"))
                trainer = BpeTrainer(
                    vocab_size=config.vocab_size,
                    special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
                )
                tokenizer.pre_tokenizer = Whitespace()
                tokenizer.train([config.articles_cache], trainer=trainer)
                
                os.makedirs(config.tokenizer_path, exist_ok=True)
                tokenizer.save(f"{config.tokenizer_path}/tokenizer.json")
                
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=f"{config.tokenizer_path}/tokenizer.json",
                    pad_token="<pad>", unk_token="<unk>",
                    bos_token="<s>", eos_token="</s>",
                )
                tokenizer.save_pretrained(config.tokenizer_path)
                self.log("✅ Tokenizer trained and saved")
            
            # Dataset
            self.log("📚 Loading dataset...")
            dataset = WikiDataset(config.articles_cache, tokenizer, config.max_seq_len)
            dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
            self.log(f"✅ Dataset ready: {len(dataset)} chunks")
            
            # Model
            self.log("🏗️ Building model...")
            model = GPTModel(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout
            ).to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            self.log(f"✅ Model initialized: {total_params:,} parameters (~25M - fits 4GB GPU)")
            
            # Check GPU memory
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.log(f"🎮 GPU Memory: {gpu_mem:.1f}GB available")
            
            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            
            def get_lr(step):
                if step < config.warmup_steps:
                    return step / config.warmup_steps
                return 0.5 * (1 + math.cos(math.pi * (step - config.warmup_steps) / 
                                          (config.max_steps - config.warmup_steps)))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
            
            # Training
            self.log("🚀 Starting training...")
            model.train()
            step = 0
            data_iter = iter(dataloader)
            
            while step < config.max_steps and self.is_training:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                logits, loss = model(input_ids, labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                if step % 100 == 0:
                    self.log(f"Step {step}/{config.max_steps} | Loss: {loss.item():.4f} | "
                            f"LR: {scheduler.get_last_lr()[0]:.2e}")
                    self.stats_label.config(text=f"Step: {step}/{config.max_steps} | "
                                                f"Loss: {loss.item():.4f}")
                
                if step > 0 and step % config.save_every == 0:
                    checkpoint_path = f"{config.output_dir}/checkpoint-{step}"
                    os.makedirs(checkpoint_path, exist_ok=True)
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f"{checkpoint_path}/model.pt")
                    self.log(f"💾 Checkpoint saved: {checkpoint_path}")
                
                step += 1
            
            if self.is_training:
                # Save final
                final_path = f"{config.output_dir}/final"
                os.makedirs(final_path, exist_ok=True)
                torch.save(model.state_dict(), f"{final_path}/model.pt")
                self.log(f"✅ Training complete! Model saved to {final_path}")
            else:
                # Save on stop
                stop_path = f"{config.output_dir}/stopped-step-{step}"
                os.makedirs(stop_path, exist_ok=True)
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{stop_path}/model.pt")
                self.log(f"💾 Training stopped! Model saved to {stop_path}")
            
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            
            # Try to save on error too
            try:
                error_path = f"{config.output_dir}/error-recovery"
                os.makedirs(error_path, exist_ok=True)
                if 'model' in locals():
                    torch.save(model.state_dict(), f"{error_path}/model.pt")
                    self.log(f"💾 Emergency save to {error_path}")
            except:
                pass
        
        finally:
            self.is_training = False
            self.train_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.stats_label.config(text="Status: Idle")
    
    def clear_log(self):
        """Clear log"""
        self.log_display.delete(1.0, 'end')
    
    def show_about(self):
        """Show about dialog"""
        about_text = """LLM Training from Wikipedia
        
Train a GPT-style language model from scratch
using Wikipedia articles as training data.

Features:
• Download Wikipedia articles
• Train custom tokenizer
• Train transformer model
• Save checkpoints

Model: ~25M parameters (optimized for 4GB GPU)
Architecture: 6-layer decoder-only transformer
Recommended: GTX 1650 or better
"""
        messagebox.showinfo("About", about_text)
    
    def run(self):
        """Run GUI"""
        self.window.mainloop()

if __name__ == "__main__":
    app = LLMTrainerGUI()
    app.run()