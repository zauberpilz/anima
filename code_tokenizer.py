"""
Code Tokenizer — Spezieller Tokenizer für Programmiersprachen.
Unterstützt Python, JavaScript, TypeScript, Rust, Go, C++.
Erkennt Syntax-Tokens (Keywords, Operatoren, Klammern) separat.
"""
import re
import json
from collections import Counter

class CodeTokenizer:
    """
    Byte-Pair-Encoding-ähnlicher Tokenizer optimiert für Code.
    Trennt Keywords, Operatoren, Identifier, Literale.
    """
    def __init__(self):
        # Programming language keywords (common across languages)
        self.keywords = {
            'def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif',
            'for', 'while', 'try', 'except', 'finally', 'with', 'as', 'in',
            'not', 'and', 'or', 'is', 'None', 'True', 'False', 'self',
            'function', 'const', 'let', 'var', 'async', 'await', 'yield',
            'fn', 'pub', 'priv', 'impl', 'trait', 'struct', 'enum', 'match',
            'func', 'go', 'package', 'interface', 'type', 'chan', 'select',
            'int', 'float', 'str', 'bool', 'void', 'auto', 'static', 'const',
            'public', 'private', 'protected', 'virtual', 'override', 'new',
            'print', 'len', 'range', 'map', 'filter', 'reduce', 'lambda',
            'torch', 'nn', 'Tensor', 'Module', 'forward', 'backward',
            'cuda', 'cpu', 'device', 'dtype', 'grad', 'backward',
        }
        
        # Operators and symbols
        self.operators = {
            '==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '//=', '%=',
            '**=', '&=', '|=', '^=', '>>=', '<<=', '->', '=>', '::', '..',
            '+', '-', '*', '/', '%', '**', '//', '&', '|', '^', '~',
            '<', '>', '=', '!', '@', '#', '$', '?', ':', ';', ',', '.',
            '(', ')', '[', ']', '{', '}', '<', '>',
        }
        
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
        
    def tokenize_code(self, text):
        """
        Tokenisiert Code-Text in semantische Tokens.
        Returns: list of tokens
        """
        tokens = []
        
        # Regex patterns
        patterns = [
            (r'"""[\s\S]*?"""', 'STRING_MULTI'),  # Multi-line strings
            (r"'''[\s\S]*?'''", 'STRING_MULTI'),
            (r'"[^"\\]*(?:\\.[^"\\]*)*"', 'STRING'),
            (r"'[^'\\]*(?:\\.[^'\\]*)*'", 'STRING'),
            (r'#.*$', 'COMMENT'),  # Comments
            (r'//.*$', 'COMMENT'),
            (r'/\*[\s\S]*?\*/', 'COMMENT'),
            (r'\b\d+\.?\d*(?:e[+-]?\d+)?\b', 'NUMBER'),  # Numbers
            (r'\b\d+\b', 'NUMBER'),
            (r'[a-zA-Z_]\w*', 'IDENTIFIER'),  # Identifiers/keywords
            (r'[^\s\w]', 'SYMBOL'),  # Symbols/operators
        ]
        
        # Combined pattern
        combined = '|'.join(f'(?P<{name}>{pattern})' for pattern, name in patterns)
        regex = re.compile(combined, re.MULTILINE)
        
        for match in regex.finditer(text):
            kind = match.lastgroup
            value = match.group()
            
            if kind == 'IDENTIFIER':
                if value in self.keywords:
                    tokens.append(f'__KW__{value}')  # Keyword token
                elif value[0].isupper() and len(value) > 1:
                    tokens.append(f'__CLS__{value}')  # Class/type token
                else:
                    tokens.append(f'__ID__{value.lower()}')  # Identifier token
            elif kind in ('STRING', 'STRING_MULTI'):
                tokens.append('__STR__')  # Generic string token
            elif kind == 'COMMENT':
                tokens.append('__CMT__')  # Generic comment token
            elif kind == 'NUMBER':
                tokens.append('__NUM__')  # Generic number token
            else:
                tokens.append(value)  # Symbol/operator as-is
                
        return tokens
    
    def build_vocab(self, texts, min_freq=2):
        """
        Baut Vocabular aus Code-Texten.
        """
        counter = Counter()
        for text in texts:
            tokens = self.tokenize_code(text)
            counter.update(tokens)
            
        # Filter by min frequency
        vocab_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
        vocab_tokens.extend([t for t, c in counter.most_common() if c >= min_freq])
        
        self.stoi = {t: i for i, t in enumerate(vocab_tokens)}
        self.itos = {i: t for i, t in enumerate(vocab_tokens)}
        self.vocab_size = len(vocab_tokens)
        
        return self.vocab_size
    
    def encode(self, text):
        """Encode text to token IDs."""
        tokens = self.tokenize_code(text)
        return [self.stoi.get(t, self.stoi.get('<unk>', 0)) for t in tokens]
    
    def decode(self, token_ids):
        """Decode token IDs back to text."""
        tokens = []
        for tid in token_ids:
            token = self.itos.get(tid, '<unk>')
            if token.startswith('__KW__'):
                tokens.append(token[6:])
            elif token.startswith('__CLS__'):
                tokens.append(token[7:])
            elif token.startswith('__ID__'):
                tokens.append(token[6:])
            elif token == '__STR__':
                tokens.append('"..."')
            elif token == '__NUM__':
                tokens.append('0')
            elif token == '__CMT__':
                tokens.append('# ...')
            else:
                tokens.append(token)
        return ' '.join(tokens)
    
    def save(self, path):
        """Save tokenizer vocab."""
        with open(path, 'w') as f:
            json.dump({
                'stoi': self.stoi,
                'itos': {str(k): v for k, v in self.itos.items()},
                'vocab_size': self.vocab_size,
            }, f, indent=2)
            
    def load(self, path):
        """Load tokenizer vocab."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.stoi = data['stoi']
        self.itos = {int(k): v for k, v in data['itos'].items()}
        self.vocab_size = data['vocab_size']


if __name__ == '__main__':
    tokenizer = CodeTokenizer()
    
    # Test with sample code
    sample = '''
def train_model(model, data, lr=0.01):
    """Train a neural network model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(100):
        loss = model.forward(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
'''
    
    tokens = tokenizer.tokenize_code(sample)
    print(f"Tokens: {len(tokens)}")
    print(f"Sample: {tokens[:20]}")
    
    vocab_size = tokenizer.build_vocab([sample])
    print(f"Vocab size: {vocab_size}")
    
    encoded = tokenizer.encode(sample)
    print(f"Encoded: {encoded[:20]}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded[:100]}")
