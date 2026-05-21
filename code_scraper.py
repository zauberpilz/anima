"""
Code Scraper — GitHub Repos und StackOverflow als Trainingsdaten.
Ressourcen-schonend mit Rate-Limits für paralleles Surfen.
"""
import os
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

class CodeScraper:
    """
    Scrapt Code-Daten von GitHub und StackOverflow.
    Nutzt Rate-Limits um das Surfen nicht zu beeinträchtigen.
    """
    def __init__(self, data_dir='/home/anima/data/code', rate_limit_delay=2.0):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay  # Sekunden zwischen Requests
        self.last_request_time = 0
        self.stats = {'github_files': 0, 'stackoverflow_posts': 0, 'total_chars': 0}
        
    def _rate_limit(self):
        """Warte um Bandbreite für Browser-traffic freizuhalten."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
        
    def scrape_github_repo(self, owner, repo, branch='main', max_files=50):
        """
        Scrapt Python-Dateien aus einem GitHub Repository.
        Nutzt die GitHub API (kein Token nötig für public repos).
        """
        print(f"[SCRAPER] GitHub: {owner}/{repo}")
        self._rate_limit()
        
        # Get repo tree
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Anima-Code-Scraper/1.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                tree = json.loads(response.read().decode())
        except Exception as e:
            print(f"  Fehler: {e}")
            return
            
        # Filter Python files
        py_files = [item['path'] for item in tree.get('tree', []) 
                    if item['path'].endswith('.py') and item['type'] == 'blob']
        
        print(f"  Gefunden: {len(py_files)} Python-Dateien")
        
        # Download files (max_files limit)
        for filepath in py_files[:max_files]:
            self._rate_limit()
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filepath}"
            try:
                req = urllib.request.Request(raw_url, headers={'User-Agent': 'Anima-Code-Scraper/1.0'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    content = response.read().decode('utf-8', errors='ignore')
                    
                if len(content) > 100:  # Nur nicht-leere Dateien
                    # Save to local file
                    safe_name = filepath.replace('/', '_')
                    output_path = self.data_dir / f"github_{owner}_{repo}_{safe_name}"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(f"# Source: https://github.com/{owner}/{repo}/blob/{branch}/{filepath}\n")
                        f.write(content)
                    
                    self.stats['github_files'] += 1
                    self.stats['total_chars'] += len(content)
                    
            except Exception as e:
                continue
                
        print(f"  Gespeichert: {self.stats['github_files']} Dateien")
        
    def scrape_stackoverflow(self, tags=['python', 'machine-learning', 'pytorch'], max_posts=100):
        """
        Scrapt StackOverflow Fragen und Antworten via API.
        """
        print(f"[SCRAPER] StackOverflow: Tags={tags}")
        
        for tag in tags:
            self._rate_limit()
            # StackExchange API
            url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q={tag}&site=stackoverflow&pagesize=30"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Anima-Code-Scraper/1.0'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = json.loads(response.read().decode())
                    
                for item in data.get('items', [])[:max_posts // len(tags)]:
                    # Extract code snippets from body
                    body = item.get('body', '')
                    # Simple code extraction (between <code> tags)
                    import re
                    code_blocks = re.findall(r'<code>(.*?)</code>', body, re.DOTALL)
                    
                    if code_blocks:
                        output_path = self.data_dir / f"so_{tag}_{item['question_id']}.txt"
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(f"# StackOverflow: {item.get('title', 'N/A')}\n")
                            f.write(f"# Tags: {', '.join(item.get('tags', []))}\n")
                            f.write(f"# Score: {item.get('score', 0)}\n\n")
                            for code in code_blocks:
                                # Clean HTML entities
                                code = code.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
                                f.write(code + '\n\n')
                        
                        self.stats['stackoverflow_posts'] += 1
                        self.stats['total_chars'] += len(body)
                        
            except Exception as e:
                print(f"  Fehler bei {tag}: {e}")
                continue
                
        print(f"  StackOverflow Posts: {self.stats['stackoverflow_posts']}")
        
    def scrape_popular_repos(self):
        """Scrapt eine Liste populärer Python/ML Repos."""
        popular_repos = [
            ('karpathy', 'nanoGPT'),
            ('huggingface', 'transformers'),
            ('pytorch', 'pytorch'),
            ('microsoft', 'DeepSpeed'),
            ('google-research', 'bert'),
            ('openai', 'whisper'),
            ('llamafile', 'llamafile'),
            ('ggerganov', 'llama.cpp'),
        ]
        
        for owner, repo in popular_repos:
            try:
                self.scrape_github_repo(owner, repo, max_files=20)
            except Exception as e:
                print(f"Fehler bei {owner}/{repo}: {e}")
                continue
                
    def get_combined_dataset(self, max_chars=None):
        """
        Kombiniert alle gescrapten Dateien zu einem Dataset.
        Kompatibel mit dem bestehenden Training-System.
        """
        import torch
        
        all_text = []
        for file_path in self.data_dir.glob('*'):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        all_text.append(f.read())
                except:
                    continue
                    
        combined = '\n\n'.join(all_text)
        
        if max_chars:
            combined = combined[:max_chars]
            
        # Build vocab
        chars = sorted(list(set(combined)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        
        data = torch.tensor([stoi.get(c, 0) for c in combined], dtype=torch.long)
        
        print(f"[SCRAPER] Dataset: {len(data):,} Tokens, Vocab: {vocab_size}, Dateien: {len(list(self.data_dir.glob('*')))}")
        
        return data, stoi, itos, vocab_size
    
    def get_stats(self):
        return self.stats


if __name__ == '__main__':
    scraper = CodeScraper()
    print("=== Code Scraper Start ===")
    scraper.scrape_popular_repos()
    scraper.scrape_stackoverflow()
    print(f"Stats: {scraper.get_stats()}")
