try:
    import tiktoken; print('tiktoken:', tiktoken.__version__)
except: print('no tiktoken')
try:
    from tokenizers import Tokenizer; print('tokenizers available')
except: print('no tokenizers')
try:
    import transformers; print('transformers:', transformers.__version__)
except: print('no transformers')
