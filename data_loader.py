"""
Erweiterter Multi-Domain Data Loader für große Korpora + Code + Security + Network.
v6: Multi-Domain Dataset Manager mit BPE-Tokenizer (tokenizers) + HuggingFace.
"""
import torch
import os
import urllib.request
import json
import random
import re
from code_scraper import CodeScraper
from code_tokenizer import CodeTokenizer

DATA_DIR = '/home/anima/data'
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_PATH = os.path.join(DATA_DIR, 'input.txt')

# ===== HUGGINGFACE DATASETS INTEGRATION =====
try:
    import datasets
    HF_AVAILABLE = True

    # Monkey-Patch: Ältere Datasets verwenden 'description' in SplitInfo, das in neueren Versionen entfernt wurde
    try:
        from datasets.splits import SplitInfo
        _orig_splitinfo_init = SplitInfo.__init__
        def _patched_splitinfo_init(self, **kwargs):
            kwargs.pop('description', None)
            _orig_splitinfo_init(self, **kwargs)
        SplitInfo.__init__ = _patched_splitinfo_init
    except (ImportError, AttributeError):
        pass

    # HF Cache auf WSL-ext4 (nicht C:) für ausreichend Speicher
    os.environ.setdefault('HF_HOME', '/home/anima/.hf_cache')
    os.environ.setdefault('HUGGINGFACE_HUB_CACHE', '/home/anima/.hf_cache/hub')
    # Kürzere Timeouts für Downloads (verhindert Hängenbleiben bei Netzwerk-Timeout)
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'
    os.environ['HF_HUB_ETAG_TIMEOUT'] = '15'
except ImportError:
    HF_AVAILABLE = False


# =====================================================================
#  UTILITY FUNCTIONS
# =====================================================================

def download_file(url, dest, max_retries=2):
    """Download mit Retry-Logik."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f'OK: {dest} ({os.path.getsize(dest)/1e6:.1f}MB)')
        return True

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(max_retries + 1):
        try:
            print(f'Download {url} (Versuch {attempt+1})...')
            urllib.request.urlretrieve(url, dest)
            size = os.path.getsize(dest)
            if size > 1000:
                print(f'OK: {size/1e6:.1f}MB')
                return True
            os.remove(dest)
        except Exception as e:
            print(f'Fehler: {e}')
    return False


def get_character_mapping(text):
    """Erstellt Character-to-Index Mapping aus Text."""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, vocab_size


def load_shakespeare_fallback(max_chars=None):
    """Ultimativer Fallback: Shakespeare."""
    file_path = SHAKESPEARE_PATH
    if not os.path.exists(file_path):
        download_file(DATASET_URL, file_path)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if max_chars:
            text = text[:max_chars]
        print(f'[FALLBACK] Shakespeare geladen: {len(text):,} Chars')
        return text
    return None


def load_tinystories_fallback(max_chars=None):
    """Fallback auf TinyStories wenn HF nicht verfügbar."""
    if not HF_AVAILABLE:
        return None
    try:
        return download_hf_dataset('roneneldan/TinyStories', 'train', max_chars, 'text')
    except Exception:
        return None


# =====================================================================
#  HUGGINGFACE DOWNLOAD CORE
# =====================================================================

def download_hf_dataset(dataset_name, split='train', max_chars=None, text_column='text'):
    """
    Lädt ein Dataset von HuggingFace und bereitet es als Char-Level Text auf.
    Fallback auf Shakespeare wenn das Dataset nicht verfügbar ist.
    """
    if not HF_AVAILABLE:
        print('[DATA] HuggingFace datasets nicht installiert. Fallback auf Shakespeare.')
        return None

    try:
        print(f'[DATA] Lade HF Dataset: {dataset_name} ({split})...')
        ds = datasets.load_dataset(dataset_name, split=split, streaming=True)

        text = ''
        count = 0
        for example in ds:
            if text_column in example and example[text_column]:
                text += str(example[text_column]) + '\n'
                count += 1
                if max_chars and len(text) >= max_chars:
                    text = text[:max_chars]
                    break
                if count >= 100000:
                    break

        if len(text) < 1000:
            print(f'[DATA] Zu wenig Text von {dataset_name}: {len(text)} Chars')
            return None

        print(f'[DATA] HF Dataset geladen: {len(text):,} Chars aus {count} Beispielen')
        return text
    except Exception as e:
        print(f'[DATA] HF Dataset Fehler ({dataset_name}): {e}')
        return None


# =====================================================================
#  DOMAIN-SPECIFIC LOADERS
# =====================================================================

def _generate_synthetic_code(num_entries=3000, seed=42):
    """Generiert synthetische Code-Daten (Funktionen, Klassen, Algorithmen).
    Zuverlässiger Fallback für Code-Domain."""
    rng = random.Random(seed + 3)
    text = ''

    code_templates = [
        ("python", "def {func_name}({params}):\n    \"\"\"{desc}\"\"\"\n    result = {expr}\n    return result\n"),
        ("python", "class {class_name}:\n    def __init__(self, {params}):\n        self.{attr} = {val}\n\n    def {method}(self):\n        return self.{attr}\n"),
        ("javascript", "function {func_name}({params}) {{\n  // {desc}\n  const result = {expr};\n  return result;\n}}\n"),
        ("c", "int {func_name}({c_params}) {{\n    // {desc}\n    {c_body}\n    return 0;\n}}\n"),
        ("java", "public class {class_name} {{\n    private {type} {attr};\n\n    public {type} {method}() {{\n        return this.{attr};\n    }}\n}}\n"),
    ]

    func_names = ['processData', 'calculateSum', 'validateInput', 'transformMatrix',
                  'parseConfig', 'mergeResults', 'filterItems', 'computeHash',
                  'normalizePath', 'serializeObject', 'deserializeStream',
                  'allocateBuffer', 'traverseGraph', 'sortEntries', 'mapReduce']
    class_names = ['DataProcessor', 'NetworkClient', 'ConfigParser', 'MatrixTransform',
                   'BufferAllocator', 'StreamEncoder', 'GraphTraverser', 'HashValidator',
                   'SessionManager', 'ConnectionPool', 'RequestHandler', 'ResponseBuilder']
    params_list = ['data, options', 'input, config', 'items, key', 'buffer, size, mode',
                   'source, destination', 'value, defaultValue', 'request, response',
                   'array, comparator', 'stream, encoding', 'connection, timeout']
    descs = ['process the input data', 'validate and transform', 'compute the result',
             'parse the configuration', 'merge multiple sources', 'filter and sort',
             'encode the stream', 'allocate memory buffer', 'traverse node graph']

    for i in range(num_entries):
        lang, template = rng.choice(code_templates)
        func_name = rng.choice(func_names)
        class_name = rng.choice(class_names)
        params = rng.choice(params_list)
        desc = rng.choice(descs)
        attr = rng.choice(['data', 'value', 'config', 'buffer', 'state', 'result', 'items'])
        val = rng.choice(['None', '0', '""', '[]', '{}', 'false', 'null'])
        expr = rng.choice(['data + config', 'items.sort()', 'buffer.copy()',
                           'stream.encode()', 'validate(value)', 'process(input)',
                           'merge(source, destination)', 'transform(array)'])
        c_params = params.replace("=", " ")
        c_body = f'if ({attr} == NULL) return -1;\n    {expr};'

        code = template.format(
            func_name=func_name, class_name=class_name,
            params=params, desc=desc, attr=attr, val=val,
            expr=expr, c_params=c_params, c_body=c_body,
            type=rng.choice(["int", "String", "float", "boolean", "byte[]"])
        )
        text += code + '\n'

    print(f'[CODE] Synthetische Daten: {len(text):,} Chars ({num_entries} Einträge)')
    return text


def load_code_datasets(max_chars=None):
    """Lädt Code-Domänen-Datasets mit Fallbacks.
    Primär: HF CodeAlpaca. Fallback: synthetisch oder CodeScraper."""
    all_text = ''

    # Primär: CodeAlpaca von HF (funktioniert zuverlässig)
    if HF_AVAILABLE:
        try:
            text = _load_codealpaca(max_chars)
            if text:
                marker = '\n--- CODE:CodeAlpaca-20k ---\n'
                all_text += marker + text
                print(f'[CODE] CodeAlpaca-20k: {len(text):,} Chars geladen')
        except Exception as e:
            print(f'[CODE] CodeAlpaca Fehler: {e}')

    # Wenn nicht genug Daten, synthetische generieren
    if len(all_text) < 1000:
        print('[CODE] Generiere synthetische Code-Daten...')
        num_entries = max(500, min(20000, max_chars // 200)) if max_chars else 3000
        syn_text = _generate_synthetic_code(num_entries=num_entries)
        all_text = syn_text

    # CodeScraper übersprungen (zu langsam für Initialisierung)
    # Kann manuell via _load_code_scraper_fallback() nachgeladen werden

    if max_chars and len(all_text) > max_chars:
        all_text = all_text[:max_chars]

    return all_text


def _load_codealpaca(max_chars=None):
    """Spezieller Loader für CodeAlpaca-20k (instruction+input+output)."""
    try:
        ds = datasets.load_dataset('sahil2801/CodeAlpaca-20k', split='train', streaming=True)
        text = ''
        for example in ds:
            instr = example.get('instruction', '')
            inp = example.get('input', '')
            out = example.get('output', '')
            combined = f'### Instruction:\n{instr}\n### Input:\n{inp}\n### Output:\n{out}\n\n'
            text += combined
            if max_chars and len(text) >= max_chars:
                text = text[:max_chars]
                break
        if len(text) > 1000:
            print(f'[CODE] CodeAlpaca-20k: {len(text):,} Chars geladen')
            return text
    except Exception as e:
        print(f'[CODE] CodeAlpaca-20k Fehler: {e}')
    return None


def _load_code_scraper_fallback(max_chars=None):
    """Fallback: CodeScraper wenn HF Datasets nicht laden."""
    try:
        scraper = CodeScraper(data_dir=DATA_DIR + '/code')
        code_files = list(scraper.data_dir.glob('*'))
        if len(code_files) < 5:
            print('[CODE] Starte Code-Scraper...')
            scraper.scrape_popular_repos()
            scraper.scrape_stackoverflow()
        return scraper.get_combined_dataset(max_chars=max_chars)
    except Exception as e:
        print(f'[CODE] CodeScraper Fallback Fehler: {e}')
        return None


def _generate_synthetic_security(num_entries=5000, seed=42):
    """Generiert synthetische Sicherheitsdaten (CVE, CWE, Exploits, Patches).
    Zuverlässig, kein Download nötig."""
    import hashlib
    rng = random.Random(seed)
    text = ''

    cwe_templates = [
        ("CWE-79", "XSS", "Cross-Site Scripting"),
        ("CWE-89", "SQLI", "SQL Injection"),
        ("CWE-120", "BOF", "Buffer Overflow"),
        ("CWE-22", "PT", "Path Traversal"),
        ("CWE-78", "CMDI", "OS Command Injection"),
        ("CWE-287", "AUTH", "Improper Authentication"),
        ("CWE-200", "INFO", "Information Exposure"),
        ("CWE-190", "INT", "Integer Overflow"),
        ("CWE-862", "AUTHZ", "Missing Authorization"),
        ("CWE-476", "NPE", "NULL Pointer Dereference"),
        ("CWE-787", "OOB", "Out-of-bounds Write"),
        ("CWE-125", "OOBR", "Out-of-bounds Read"),
        ("CWE-20", "IVAL", "Improper Input Validation"),
        ("CWE-502", "DESER", "Deserialization of Untrusted Data"),
        ("CWE-611", "XXE", "XML External Entity"),
        ("CWE-918", "SSRF", "Server-Side Request Forgery"),
        ("CWE-434", "UPLOAD", "Unrestricted File Upload"),
        ("CWE-798", "HARD", "Hardcoded Credentials"),
        ("CWE-295", "CERT", "Improper Certificate Validation"),
        ("CWE-400", "DOS", "Resource Exhaustion"),
    ]

    code_snippets_vuln = [
        'char buf[64]; strcpy(buf, user_input);',
        'eval(request.GET.get("code"))',
        "SELECT * FROM users WHERE id = '{}'".format(" + user_input + "),
        'System.Runtime.Remoting.Channels.ChannelServices.RegisterChannel(new TcpChannel(port));',
        'Process.Start("cmd.exe", "/c " + userInput);',
        '<?php include("includes/" + $_GET["page"]); ?>',
        'File.ReadAllText("/var/data/" + fileName);',
        'socket.send(data); socket.recv(4096);',
        'int *ptr = malloc(32); ptr[32] = value;',
        'pass = "admin123";',
        'conn = OpenConnection(connStr); cmd.CommandText = "SELECT * FROM users WHERE id=" + id;',
        'with open(filename, "r") as f: return f.read()',
        'var fs = require("fs"); fs.readFile("/etc/passwd", callback);',
        'XMLReader reader = XMLReader.Create(input); reader.Settings.DtdProcessing = DtdProcessing.Parse;',
    ]

    code_snippets_fix = [
        'char buf[64]; strncpy(buf, user_input, sizeof(buf)-1); buf[sizeof(buf)-1] = 0;',
        'import subprocess; subprocess.run(["ls", "-l"], capture_output=True)',
        'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))',
        'ConfigureAwait(false);',
        'Process.Start("cmd.exe", $"/c {EscapeArgument(userInput)}");',
        '<?php $allowed = ["home", "about"]; if (in_array($_GET["page"], $allowed)) include($allowed[$_GET["page"]]); ?>',
        'Path.Combine(basePath, Path.GetFileName(fileName));',
        'import ssl; ctx = ssl.create_default_context(); ctx.check_hostname = True;',
        'int *ptr = malloc(32 * sizeof(int)); if (ptr && idx < 32) ptr[idx] = value;',
        'from cryptography.fernet import Fernet; key = Fernet.generate_key();',
        'cmd.CommandText = "SELECT * FROM users WHERE id=@id"; cmd.Parameters.AddWithValue("@id", id);',
        'import os; allowed = ["file1.txt", "file2.txt"]; basename = os.path.basename(fname); '
        + 'if basename in allowed: open(basename, "r")',
        'let { execFile } = require("child_process"); execFile("/bin/ls", ["-l"], callback);',
        'XmlReaderSettings settings = new XmlReaderSettings(); settings.DtdProcessing = DtdProcessing.Prohibit;',
    ]

    for i in range(num_entries):
        cwe_id, cwe_short, cwe_desc = rng.choice(cwe_templates)
        year = rng.randint(2016, 2024)
        cve_id = f"CVE-{year}-{rng.randint(1000, 99999)}"
        vuln_code = rng.choice(code_snippets_vuln)
        fix_code = rng.choice(code_snippets_fix)
        severity = rng.choice(["CRITICAL", "HIGH", "MEDIUM", "LOW"])
        cvss = round(rng.uniform(4.0, 10.0), 1)

        exploit_type = rng.choice(["remote", "local", "dos", "xss", "sqli", "rce", "priv-esc"])

        entry = (
            f"[CVE: {cve_id}] [CWE: {cwe_id} ({cwe_short})] [SEVERITY: {severity}] "
            f"[CVSS: {cvss}] [TYPE: {exploit_type}]\n"
            f"DESCRIPTION: {cwe_desc} vulnerability in input handling. "
            f"An attacker can exploit this via crafted input to gain {exploit_type} access.\n"
            f"VULNERABLE CODE:\n{vuln_code}\n"
            f"FIXED CODE:\n{fix_code}\n\n"
        )
        text += entry

    print(f'[SECURITY] Synthetische Daten: {len(text):,} Chars ({num_entries} Einträge)')
    return text


def load_security_datasets(max_chars=None):
    """Lädt Security-Domänen-Datasets (Vulnerabilities, CVEs, Patches).
    Primär: synthetische Generierung (zuverlässig). Fallback: HF-Datasets."""
    # Primär synthetische Daten — schnell und zuverlässig
    num_entries = max(500, min(20000, max_chars // 200)) if max_chars else 5000
    text = _generate_synthetic_security(num_entries=num_entries)

    # Optional: HF-Datasets übersprungen (100% zuverlässig nur synthetisch)
    # Bei Bedarf: _load_cwe_patch() manuell aufrufen

    if max_chars and len(text) > max_chars:
        text = text[:max_chars]

    return text


def _load_primevul(max_chars=None):
    """Lädt PrimeVul (vulnerable code) und formatiert es."""
    try:
        ds = datasets.load_dataset('starsofchance/PrimeVul', split='train', streaming=True)
        text = ''
        for example in ds:
            func = example.get('func', '')
            cve = example.get('CVE', '')
            cwe = example.get('CWE', '')
            severity = example.get('severity', '')
            if func:
                header = f'[CVE: {cve}] [CWE: {cwe}] [SEVERITY: {severity}]\n'
                text += header + func + '\n\n'
                if max_chars and len(text) >= max_chars:
                    text = text[:max_chars]
                    break
        if len(text) > 1000:
            return text
    except Exception as e:
        print(f'[SECURITY] PrimeVul Fehler: {e}')
    return None


def _load_cvefixes(max_chars=None):
    """Lädt CVEfixes v1.0.8 und konvertiert in unified Format.
    Dataset-Struktur: cve_id, description, nodes (list of dicts: code_before, code_after, cwe, severity)."""
    try:
        ds = datasets.load_dataset('starsofchance/CVEfixes_v1.0.8', split='train', streaming=True)
        text = ''
        for example in ds:
            cve_id = example.get('cve_id', '')
            description = example.get('description', '')
            nodes = example.get('nodes', [])
            if isinstance(nodes, list):
                for node in nodes:
                    if isinstance(node, dict):
                        code_before = node.get('code_before', '') or node.get('before', '')
                        code_after = node.get('code_after', '') or node.get('after', '')
                        cwe_list = node.get('cwe', [])
                        cwe_str = ', '.join(cwe_list) if isinstance(cwe_list, list) else str(cwe_list)
                        severity = node.get('severity', '')
                        if code_before or code_after:
                            entry = f'[{cve_id}] [CWE: {cwe_str}] [SEVERITY: {severity}]\n'
                            entry += f'DESC: {description[:200]}\n'
                            if code_before:
                                entry += f'BEFORE: {code_before}\n'
                            if code_after:
                                entry += f'AFTER: {code_after}\n'
                            text += entry + '\n'
                            if max_chars and len(text) >= max_chars:
                                return text[:max_chars]
            elif isinstance(nodes, dict):
                # Fallback: nodes ist ein einzelnes Dict
                code_before = nodes.get('code_before', '') or nodes.get('before', '')
                code_after = nodes.get('code_after', '') or nodes.get('after', '')
                cwe_str = str(nodes.get('cwe', ''))
                severity = nodes.get('severity', '')
                if code_before or code_after:
                    entry = f'[{cve_id}] [CWE: {cwe_str}] [SEVERITY: {severity}]\n'
                    entry += f'DESC: {description[:200]}\nBEFORE: {code_before}\nAFTER: {code_after}\n\n'
                    text += entry
                    if max_chars and len(text) >= max_chars:
                        return text[:max_chars]
        if len(text) > 1000:
            return text
    except Exception as e:
        print(f'[SECURITY] CVEfixes Fehler: {e}')
    return None


def _load_cwe_patch(max_chars=None):
    """Lädt CIRCL vulnerability-cwe-patch Dataset (5 shards, streaming)."""
    try:
        ds = datasets.load_dataset('CIRCL/vulnerability-cwe-patch', split='train', streaming=True)
        text = ''
        for example in ds:
            patch = example.get('patch', '')
            cwe = example.get('cwe', '')
            summary = example.get('summary', '')
            if patch:
                entry = f'[CWE: {cwe}] {summary}\nPATCH: {patch}\n\n'
                text += entry
                if max_chars and len(text) >= max_chars:
                    text = text[:max_chars]
                    break
        if len(text) > 1000:
            return text
    except Exception as e:
        print(f'[SECURITY] CWE-Patch Fehler: {e}')
    return None


def _generate_synthetic_network(num_entries=5000, seed=42):
    """Generiert synthetische Netzwerk-Traffic-Daten (Flows, Anomalien).
    Zuverlässig, kein Download nötig."""
    rng = random.Random(seed + 1)
    text = ''

    protocols = ['tcp', 'udp', 'icmp', 'arp', 'dns', 'http', 'https', 'ssh', 'ftp', 'smtp']
    services = ['HTTP', 'DNS', 'SSH', 'FTP', 'SMTP', '-', 'MySQL', 'Redis', 'MQTT']
    states = ['INT', 'FIN', 'CON', 'CLS', 'RST', 'ACC', 'SYN']
    attack_types = ['Normal', 'Fuzzers', 'Analysis', 'Backdoors', 'DoS', 'Exploits',
                    'Generic', 'Reconnaissance', 'Shellcode', 'Worms', 'DDoS', 'PortScan']

    for i in range(num_entries):
        proto = rng.choice(protocols)
        service = rng.choice(services)
        state = rng.choice(states)
        is_anomaly = rng.random() < 0.2  # 20% anomaly rate
        label = rng.choice(attack_types) if is_anomaly else 'Normal'

        # Generate realistic-looking traffic features
        dur = round(rng.uniform(0.0001, 60.0), 6)
        spkts = rng.randint(1, 500)
        dpkts = rng.randint(1, 500) if not is_anomaly else rng.randint(0, 2000)
        sbytes = rng.randint(40, 1500) * spkts
        dbytes = rng.randint(40, 1500) * dpkts
        rate = round(sbytes / max(dur, 0.001), 1)
        sload = round(spkts / max(dur, 0.001), 1)
        dload = round(dpkts / max(dur, 0.001), 1)
        sttl = rng.randint(32, 255)
        dttl = rng.randint(32, 255)

        if is_anomaly:
            # Make anomalous traffic more distinctive
            spkts = rng.randint(100, 10000)
            dpkts = rng.randint(0, 100)
            rate = round(rng.uniform(10000, 1000000), 1)

        entry = (f'[FLOW] proto={proto} service={service} state={state} '
                 f'dur={dur}s spkts={spkts} dpkts={dpkts} '
                 f'sbytes={sbytes} dbytes={dbytes} rate={rate:.1f} '
                 f'sload={sload:.1f} dload={dload:.1f} '
                 f'sttl={sttl} dttl={dttl} '
                 f'anomaly={1 if is_anomaly else 0} label={label}\n')
        text += entry

    print(f'[NETWORK] Synthetische Daten: {len(text):,} Chars ({num_entries} Einträge)')
    return text


def load_network_datasets(max_chars=None):
    """Lädt Network-Domänen-Datasets (Traffic, Anomalien).
    Primär: synthetische Generierung (zuverlässig). Fallback: HF-Datasets."""
    # Primär synthetische Daten
    num_entries = max(500, min(20000, max_chars // 200)) if max_chars else 5000
    text = _generate_synthetic_network(num_entries=num_entries)

    # Optional: HF-Datasets übersprungen (100% zuverlässig nur synthetisch)
    # Bei Bedarf: _load_unsw_nb15() oder _load_cicids2017() manuell aufrufen

    if max_chars and len(text) > max_chars:
        text = text[:max_chars]

    return text


def _load_cicids2017(max_chars=None):
    """Lädt CIC-IDS-2017 (config='machine_learning') und konvertiert in Text-Format."""
    config_name = 'machine_learning'  # for 'bvsam/cic-ids-2017' - 80 features + labels
    try:
        ds = datasets.load_dataset('bvsam/cic-ids-2017', config_name, split='train', streaming=True)
    except Exception as e:
        print(f'[NETWORK] CIC-IDS-2017 Fehler: {e}')
        return None

    text = ''
    count = 0
    try:
        for example in ds:
            # CIC-IDS-2017 machine_learning config: 79 numerische Features + Label
            dst_port = example.get('Destination Port', '')
            flow_dur = example.get('Flow Duration', '')
            fwd_pkts = example.get('Total Fwd Packets', '')
            bwd_pkts = example.get('Total Backward Packets', '')
            fwd_bytes = example.get('Total Length of Fwd Packets', '')
            bwd_bytes = example.get('Total Length of Bwd Packets', '')
            fwd_iat = example.get('Fwd IAT Mean', '')
            fin_flags = example.get('FIN Flag Count', '')
            syn_flags = example.get('SYN Flag Count', '')
            rst_flags = example.get('RST Flag Count', '')
            ack_flags = example.get('ACK Flag Count', '')
            pkt_len_mean = example.get('Packet Length Mean', '')
            label = example.get('Label', '')

            # Formatiere als Flow-Eintrag ohne IPs (CIC-IDS-2017 hat keine IP-Spalten)
            entry = (f'[FLOW] dst_port={dst_port} duration={flow_dur} '
                     f'fwd_pkts={fwd_pkts} bwd_pkts={bwd_pkts} '
                     f'fwd_bytes={fwd_bytes} bwd_bytes={bwd_bytes} '
                     f'fwd_iat={fwd_iat} pkt_len_mean={pkt_len_mean} '
                     f'flags=fin:{fin_flags}/syn:{syn_flags}/rst:{rst_flags}/ack:{ack_flags} '
                     f'label={label}\n')
            text += entry
            count += 1

            if max_chars and len(text) >= max_chars:
                text = text[:max_chars]
                break
            if count >= 50000:
                break

        if len(text) > 1000:
            print(f'[NETWORK] CIC-IDS-2017: {len(text):,} Chars aus {count} Einträgen')
            return text
    except Exception as e:
        print(f'[NETWORK] CIC-IDS-2017 Verarbeitungsfehler: {e}')

    return None


def _load_unsw_nb15(max_chars=None):
    """Lädt UNSW-NB15 (45 Spalten) und konvertiert in strukturiertes Flow-Format.
    Spalten: id, dur, proto, service, state, spkts, dpkts, sbytes, dbytes, rate, ..."""
    try:
        ds = datasets.load_dataset('Mireu-Lab/UNSW-NB15', split='train', streaming=True)
    except Exception as e:
        print(f'[NETWORK] UNSW-NB15 Fehler: {e}')
        return None

    text = ''
    count = 0
    try:
        for example in ds:
            proto = example.get('proto', 'tcp')
            service = example.get('service', '-')
            state = example.get('state', 'INT')
            dur = example.get('dur', 0)
            spkts = example.get('spkts', 0)
            dpkts = example.get('dpkts', 0)
            sbytes = example.get('sbytes', 0)
            dbytes = example.get('dbytes', 0)
            rate = example.get('rate', 0)
            sload = example.get('sload', 0)
            dload = example.get('dload', 0)
            attack_cat = example.get('attack_cat', 'Normal')
            label = example.get('label', 0)
            sttl = example.get('sttl', 64)
            dttl = example.get('dttl', 64)

            # Formatiere als Netzwerk-Flow
            entry = (f'[FLOW] proto={proto} service={service} state={state} '
                     f'dur={dur}s spkts={spkts} dpkts={dpkts} '
                     f'sbytes={sbytes} dbytes={dbytes} rate={rate:.1f} '
                     f'sload={sload:.1f} dload={dload:.1f} '
                     f'sttl={sttl} dttl={dttl} '
                     f'anomaly={label} label={attack_cat}\n')
            text += entry
            count += 1

            if max_chars and len(text) >= max_chars:
                text = text[:max_chars]
                break
            if count >= 50000:
                break

        if len(text) > 1000:
            print(f'[NETWORK] UNSW-NB15: {len(text):,} Chars aus {count} Einträgen')
            return text
    except Exception as e:
        print(f'[NETWORK] UNSW-NB15 Verarbeitungsfehler: {e}')

    return None


def _generate_synthetic_text(num_entries=2000, seed=42):
    """Generiert synthetische Textdaten (Fallback wenn HF nicht verfügbar)."""
    rng = random.Random(seed + 2)
    text = ''

    sentence_templates = [
        "The {adj} {noun} {verb} the {noun2}.",
        "In the {adj} world, every {noun} must {verb}.",
        "A {adj} {noun} is better than a {adj2} {noun2}.",
        "The {noun} {adv} {verb}s through the {adj} landscape.",
        "When {noun} meets {noun2}, {adj} things happen.",
        "The {adj} system processes {noun} data efficiently.",
        "Every {noun} has a {adj} purpose in the grand design.",
        "The {noun2} of {noun} determines its {adj} nature.",
        "A {adj} approach to {noun} yields {adj2} results.",
        "The {noun} {verb}s {adv} while the {noun2} watches.",
    ]

    adjs = ['bright', 'dark', 'complex', 'simple', 'ancient', 'modern',
            'powerful', 'subtle', 'deep', 'shallow', 'vast', 'tiny',
            'colorful', 'monochrome', 'dynamic', 'static', 'fluid', 'rigid']
    nouns = ['mind', 'machine', 'system', 'network', 'algorithm', 'process',
             'pattern', 'signal', 'wave', 'field', 'stream', 'code',
             'data', 'flow', 'node', 'path', 'loop', 'matrix']
    verbs = ['connects', 'transforms', 'processes', 'generates', 'analyzes',
             'synthesizes', 'transmits', 'encodes', 'decodes', 'amplifies']
    advs = ['swiftly', 'slowly', 'efficiently', 'gracefully', 'powerfully',
            'subtly', 'constantly', 'periodically', 'automatically', 'silently']

    for i in range(num_entries):
        adj = rng.choice(adjs)
        adj2 = rng.choice(adjs)
        noun = rng.choice(nouns)
        noun2 = rng.choice(nouns)
        verb = rng.choice(verbs)
        adv = rng.choice(advs)
        template = rng.choice(sentence_templates)
        sentence = template.format(adj=adj, adj2=adj2, noun=noun, noun2=noun2, verb=verb, adv=adv)

        # Build a short paragraph
        paragraph = sentence + ' '
        for _ in range(rng.randint(2, 5)):
            adj = rng.choice(adjs)
            noun = rng.choice(nouns)
            verb = rng.choice(verbs)
            paragraph += f"The {adj} {noun} {verb} " + rng.choice(advs) + ". "

        text += paragraph + '\n\n'

    print(f'[TEXT] Synthetische Daten: {len(text):,} Chars ({num_entries} Absätze)')
    return text


def _load_with_timeout(loader_fn, timeout_sec=30, name='Dataset', **kwargs):
    """Führt eine Loader-Funktion mit Timeout aus (threading-basiert)."""
    import threading
    result = {'value': None, 'error': None}

    def worker():
        try:
            result['value'] = loader_fn(**kwargs)
        except Exception as e:
            result['error'] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        print(f'  ⏱ {name} Timeout ({timeout_sec}s), überspringe.')
        return None
    if result['error']:
        # Leise ignorieren, nur bei Debug ebene
        return None
    return result['value']


def load_text_datasets(max_chars=None):
    """Lädt Text-Domänen-Datasets (synthetisch primär, HF optional).
    Primär: synthetische Generierung (sofort verfügbar).
    Optional: TinyStories von HF (mit 30s Timeout)."""
    # Primär synthetische Daten — sofort verfügbar
    num_entries = max(500, min(20000, max_chars // 200)) if max_chars else 2000
    all_text = _generate_synthetic_text(num_entries=num_entries)
    print(f'[TEXT] Synthetische Daten: {len(all_text):,} Chars ({num_entries} Absätze)')

    # Optional: HF-Datasets übersprungen (100% zuverlässig nur synthetisch)
    # Bei Bedarf: download_hf_dataset('roneneldan/TinyStories', ...) manuell aufrufen

    # Shakespeare als zusätzlicher Fallback
    sh = load_shakespeare_fallback(max_chars // 2 if max_chars else None)
    if sh:
        all_text = all_text + '\n--- TEXT:Shakespeare ---\n' + sh

    if max_chars and len(all_text) > max_chars:
        all_text = all_text[:max_chars]

    return all_text


# =====================================================================
#  MULTI-DOMAIN DATASET MANAGER
# =====================================================================

class MultiDomainDataset:
    """
    Multi-Domain Dataset Manager — Code + Security + Network + Text.
    Verwaltet separate Domänen mit Gewichtung und shared Character Mapping.
    """
    def __init__(self, max_chars_per_domain=5000000, bpe_tokenizer_path=None):
        self.max_chars_per_domain = max_chars_per_domain
        self.domains = {
            'code': {'weight': 0.3, 'enabled': True, 'data': None, 'source': None},
            'security': {'weight': 0.2, 'enabled': True, 'data': None, 'source': None},
            'network': {'weight': 0.1, 'enabled': True, 'data': None, 'source': None},
            'text': {'weight': 0.4, 'enabled': True, 'data': None, 'source': None},
        }
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
        self.data = None
        self.domain_ranges = {}
        self._loaded = False
        # BPE Tokenizer (optional, falls None -> char-level)
        self.bpe_tokenizer = None
        if bpe_tokenizer_path and os.path.exists(bpe_tokenizer_path):
            try:
                from tokenizers import Tokenizer
                self.bpe_tokenizer = Tokenizer.from_file(bpe_tokenizer_path)
                self.vocab_size = self.bpe_tokenizer.get_vocab_size()
                print(f'[BPE] Tokenizer geladen: Vocab={self.vocab_size}')
            except Exception as e:
                print(f'[BPE] Fehler: {e}')

    # -----------------------------------------------------------------
    #  LOADING
    # -----------------------------------------------------------------

    def load_all(self, max_chars_per_domain=None):
        """Lädt alle aktiven Domänen."""
        if max_chars_per_domain is not None:
            self.max_chars_per_domain = max_chars_per_domain

        print('=' * 60)
        print('[MULTI] Multi-Domain Dataset Loader')
        print('=' * 60)

        domain_loaders = {
            'code': load_code_datasets,
            'security': load_security_datasets,
            'network': load_network_datasets,
            'text': load_text_datasets,
        }

        all_texts = {}
        for domain_name, loader_fn in domain_loaders.items():
            if not self.domains[domain_name]['enabled']:
                print(f'[MULTI] Domäne {domain_name} ist deaktiviert, überspringe.')
                continue

            print(f'\n[MULTI] Lade Domäne: {domain_name} ...')
            text = loader_fn(max_chars=self.max_chars_per_domain)
            if text and len(text) > 1000:
                all_texts[domain_name] = text
                self.domains[domain_name]['data'] = text
                print(f'[MULTI] ✓ {domain_name}: {len(text):,} Chars')
            else:
                print(f'[MULTI] ✗ {domain_name}: Keine Daten, fallback auf TinyStories')
                fallback = load_tinystories_fallback(self.max_chars_per_domain)
                if fallback:
                    all_texts[domain_name] = fallback
                    self.domains[domain_name]['data'] = fallback
                    self.domains[domain_name]['source'] = 'fallback'
                else:
                    sh = load_shakespeare_fallback(self.max_chars_per_domain)
                    if sh:
                        all_texts[domain_name] = sh
                        self.domains[domain_name]['data'] = sh
                        self.domains[domain_name]['source'] = 'shakespeare_fallback'

        if not all_texts:
            raise RuntimeError('[MULTI] KEINE DATEN VERFUEGBAR!')

        # BPE oder Character-Level Tokenisierung
        if self.bpe_tokenizer is not None:
            # BPE Tokenizer: encode each domain separately
            print(f'[MULTI] Tokenisiere mit BPE (Vocab={self.vocab_size})...')
            all_tensors = []
            current_offset = 0
            for domain_name in ['code', 'security', 'network', 'text']:
                if domain_name in all_texts:
                    domain_text = all_texts[domain_name]
                    encoded = self.bpe_tokenizer.encode(domain_text)
                    domain_ids = encoded.ids
                    domain_tensor = torch.tensor(domain_ids, dtype=torch.long)
                    start_idx = current_offset
                    end_idx = current_offset + len(domain_tensor)
                    self.domain_ranges[domain_name] = (start_idx, end_idx)
                    all_tensors.append(domain_tensor)
                    current_offset += len(domain_tensor)
                    weight = self.domains[domain_name]['weight']
                    print(f'[MULTI]   {domain_name:10s}: {len(domain_tensor):>10,} Tokens (weight={weight:.1f})')
                else:
                    self.domain_ranges[domain_name] = (current_offset, current_offset)
            # stoi/itos: Dummy-Mappings für BPE (wir nutzen tokenizer direkt)
            self.stoi = {}  # Not used with BPE
            self.itos = {}
        else:
            # Shared Character Mapping über alle Domänen (Legacy)
            combined_text_for_vocab = ''.join(all_texts.values())
            self.stoi, self.itos, self.vocab_size = get_character_mapping(combined_text_for_vocab)
            all_tensors = []
            current_offset = 0
            for domain_name in ['code', 'security', 'network', 'text']:
                if domain_name in all_texts:
                    domain_text = all_texts[domain_name]
                    domain_tensor = torch.tensor(
                        [self.stoi.get(c, 0) for c in domain_text],
                        dtype=torch.long
                    )
                    start_idx = current_offset
                    end_idx = current_offset + len(domain_tensor)
                    self.domain_ranges[domain_name] = (start_idx, end_idx)
                    all_tensors.append(domain_tensor)
                    current_offset += len(domain_tensor)
                    weight = self.domains[domain_name]['weight']
                    print(f'[MULTI]   {domain_name:10s}: {len(domain_tensor):>10,} Tokens (weight={weight:.1f})')
                else:
                    self.domain_ranges[domain_name] = (current_offset, current_offset)

        self.data = torch.cat(all_tensors) if all_tensors else torch.tensor([], dtype=torch.long)
        self._loaded = True

        print(f'\n[MULTI] GESAMT: {len(self.data):,} Tokens, Vocab: {self.vocab_size}')
        print(f'[MULTI] Domain-Ranges: {self.domain_ranges}')
        print('=' * 60)

        return self

    def load_domain(self, domain_name):
        """Lädt eine einzelne Domäne mit Fallbacks."""
        loaders = {
            'code': load_code_datasets,
            'security': load_security_datasets,
            'network': load_network_datasets,
            'text': load_text_datasets,
        }

        if domain_name not in loaders:
            raise ValueError(f'Unbekannte Domäne: {domain_name}. Valid: {list(loaders.keys())}')

        text = loaders[domain_name](max_chars=self.max_chars_per_domain)
        if not text or len(text) < 1000:
            print(f'[MULTI] Fallback für {domain_name} auf TinyStories')
            text = load_tinystories_fallback(self.max_chars_per_domain)
        if not text or len(text) < 1000:
            print(f'[MULTI] Fallback für {domain_name} auf Shakespeare')
            text = load_shakespeare_fallback(self.max_chars_per_domain)

        if text:
            self.domains[domain_name]['data'] = text

            # Stelle sicher dass das mapping existiert
            if not self.stoi:
                self.stoi, self.itos, self.vocab_size = get_character_mapping(text)

            domain_tensor = torch.tensor(
                [self.stoi.get(c, 0) for c in text],
                dtype=torch.long
            )
            return domain_tensor

        return None

    # -----------------------------------------------------------------
    #  BATCHING
    # -----------------------------------------------------------------

    def get_batch(self, domain, batch_size, seq_len, device='cpu'):
        """Holt einen Batch aus einer spezifischen Domäne."""
        if domain not in self.domain_ranges:
            raise ValueError(f'Unbekannte Domäne: {domain}')

        start, end = self.domain_ranges[domain]
        domain_len = end - start

        if domain_len < seq_len + 1:
            print(f'[BATCH] {domain} zu klein ({domain_len}), verwende gesamte Domäne')
            # Wiederhole Daten wenn nötig
            domain_data = self.data[start:end]
            if len(domain_data) < seq_len + 1:
                repeats = (seq_len + 1) // len(domain_data) + 1
                domain_data = domain_data.repeat(repeats)
            domain_data = domain_data[:seq_len + 1]
            ix = 0
        else:
            domain_data = self.data[start:end]
            ix = torch.randint(len(domain_data) - seq_len, (batch_size,))

        if domain_len >= seq_len + 1:
            x = torch.stack([domain_data[i:i+seq_len] for i in ix])
            y = torch.stack([domain_data[i+1:i+seq_len+1] for i in ix])
        else:
            x = domain_data[:seq_len].unsqueeze(0).expand(batch_size, -1)
            y = domain_data[1:seq_len+1].unsqueeze(0).expand(batch_size, -1)

        return x.to(device), y.to(device)

    def get_mixed_batch(self, batch_size, seq_len, device='cpu'):
        """
        Erzeugt einen Batch gemischt nach Domänen-Gewichtung.
        Jede Domäne liefert proportional zu ihrem weight.
        """
        if not self._loaded or self.data is None:
            raise RuntimeError('[MULTI]Dataset noch nicht geladen. Rufe load_all() auf.')

        # Berechne Samples pro Domäne basierend auf Gewichtung
        total_weight = sum(
            d['weight'] for d in self.domains.values()
            if d['enabled'] and d['data'] is not None
        )

        if total_weight == 0:
            raise RuntimeError('[MULTI] Keine aktiven Domänen mit Daten!')

        samples_per_domain = {}
        remaining = batch_size
        active_domains = [
            name for name, d in self.domains.items()
            if d['enabled'] and d['data'] is not None
        ]

        # Erste Zuteilung nach Gewicht
        for i, name in enumerate(active_domains):
            if i == len(active_domains) - 1:
                samples_per_domain[name] = remaining
            else:
                n = max(1, int(batch_size * self.domains[name]['weight'] / total_weight))
                samples_per_domain[name] = n
                remaining -= n

        # Sammle Batches aus jeder Domäne
        x_parts = []
        y_parts = []

        for name, n_samples in samples_per_domain.items():
            if n_samples <= 0:
                continue
            try:
                x_domain, y_domain = self.get_batch(name, n_samples, seq_len, device)
                x_parts.append(x_domain)
                y_parts.append(y_domain)
            except Exception as e:
                print(f'[BATCH] Fehler bei Domäne {name}: {e}')
                continue

        if not x_parts:
            # Fallback: erste verfügbare Domäne
            fallback_domain = active_domains[0] if active_domains else 'text'
            print(f'[BATCH] Fallback auf {fallback_domain}')
            return self.get_batch(fallback_domain, batch_size, seq_len, device)

        x = torch.cat(x_parts, dim=0)
        y = torch.cat(y_parts, dim=0)

        # Shuffle innerhalb des Batches
        perm = torch.randperm(x.size(0))
        x = x[perm]
        y = y[perm]

        return x, y

    def get_domain_weights(self):
        """Gibt die aktuellen Gewichte der Domänen zurück."""
        return {name: d['weight'] for name, d in self.domains.items()}

    def set_domain_weight(self, domain, weight):
        """Setzt das Gewicht einer Domäne (0.0 - 1.0)."""
        if domain in self.domains:
            self.domains[domain]['weight'] = max(0.0, min(1.0, weight))
            print(f'[MULTI] Gewicht {domain} -> {self.domains[domain]["weight"]:.2f}')
        else:
            raise ValueError(f'Unbekannte Domäne: {domain}')

    def enable_domain(self, domain, enabled=True):
        """Aktiviert oder deaktiviert eine Domäne."""
        if domain in self.domains:
            self.domains[domain]['enabled'] = enabled
            print(f'[MULTI] Domäne {domain} {"aktiviert" if enabled else "deaktiviert"}')
        else:
            raise ValueError(f'Unbekannte Domäne: {domain}')

    # -----------------------------------------------------------------
    #  STATISTICS
    # -----------------------------------------------------------------

    def get_vocab_stats(self):
        """Gibt Vokabular-Statistiken aus."""
        if not self.stoi:
            print('[STATS] Kein Vokabular geladen.')
            return

        print('\n' + '=' * 50)
        print('VOCABULARY STATISTICS')
        print('=' * 50)
        print(f'Vocab Size: {self.vocab_size}')
        print(f'Total Tokens: {len(self.data):,}' if self.data is not None else 'Total Tokens: N/A')

        for name in ['code', 'security', 'network', 'text']:
            d = self.domains[name]
            data_len = 0
            start, end = self.domain_ranges.get(name, (0, 0))
            if self.data is not None and end > start:
                data_len = end - start

            status = '✓' if d['data'] is not None else '✗'
            source = d.get('source', 'hf')
            print(f'  {name:10s} [{status}] weight={d["weight"]:.1f} '
                  f'tokens={data_len:>10,} source={source}')

        if self.stoi:
            # Top 10 häufigste Zeichen
            if self.data is not None and len(self.data) > 0:
                counts = torch.bincount(self.data, minlength=self.vocab_size)
                top_chars = counts.topk(min(10, self.vocab_size))
                print('\nTop 10 Characters:')
                for i in range(top_chars.indices.size(0)):
                    idx = top_chars.indices[i].item()
                    char = self.itos.get(idx, repr(idx))
                    count = top_chars.values[i].item()
                    pct = 100.0 * count / len(self.data)
                    print(f'  {i+1}. {repr(char):>6} -> {count:>8,} ({pct:.2f}%)')

        print('=' * 50)

    def get_domain_info(self):
        """Gibt strukturierte Infos über alle Domänen zurück."""
        info = {}
        for name in ['code', 'security', 'network', 'text']:
            d = self.domains[name]
            start, end = self.domain_ranges.get(name, (0, 0))
            info[name] = {
                'weight': d['weight'],
                'enabled': d['enabled'],
                'has_data': d['data'] is not None,
                'tokens': (end - start) if self.data is not None else 0,
                'source': d.get('source', 'hf'),
            }
        info['total_tokens'] = len(self.data) if self.data is not None else 0
        info['vocab_size'] = self.vocab_size
        return info

    def decode(self, token_ids):
        """Decode token IDs back to text (supports both BPE and char-level)."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if self.bpe_tokenizer is not None:
            return self.bpe_tokenizer.decode(token_ids)
        else:
            return ''.join([self.itos.get(i, '?') for i in token_ids])

    def encode(self, text):
        """Encode text to token IDs (supports both BPE and char-level)."""
        if self.bpe_tokenizer is not None:
            return self.bpe_tokenizer.encode(text).ids
        else:
            return [self.stoi.get(c, 0) for c in text]


# =====================================================================
#  CONVENIENCE FUNCTIONS (backward compatible)
# =====================================================================

def get_multi_domain_data(max_chars=3000000):
    """
    Lädt und kombiniert alle Domänen in ein unified Dataset.
    Gibt MultiDomainDataset-Instanz zurück.
    """
    loader = MultiDomainDataset(max_chars_per_domain=max_chars)
    loader.load_all()
    return loader


def get_large_dataset(max_chars=None):
    """
    Lädt großen Text-Korpus via HuggingFace Datasets (Multi-Domain).
    Multi-Source Fallback: TinyStories -> Shakespeare -> ...
    Fällt zurück auf Shakespeare (input.txt) wenn alles fehlschlägt.

    Returns: (data_tensor, stoi, itos, vocab_size)
    """
    try:
        loader = get_multi_domain_data(max_chars=max_chars or 3000000)
        if loader is not None and loader.data is not None and len(loader.data) > 1000:
            print(f'[DATA] Multi-Domain Dataset: {len(loader.data):,} Tokens, '
                  f'Vocab: {loader.vocab_size}')
            return loader.data, loader.stoi, loader.itos, loader.vocab_size
    except Exception as e:
        print(f'[DATA] Multi-Domain Fehler, Fallback auf single-source: {e}')

    # Legacy Fallback
    os.makedirs(DATA_DIR, exist_ok=True)

    HF_DATASETS = [
        ('roneneldan/TinyStories', 'train', 'text'),
        ('shakespeare', 'train', 'text'),
        ('tiny_shakespeare', 'train', 'text'),
        ('stas/openwebtext-10k', 'train', 'text'),
        ('bigcode/the-stack-smol', 'train', 'content'),
        ('roneneldan/TinyStoriesV2', 'train', 'text'),
        ('HuggingFaceFW/fineweb-edu', 'train', 'text'),
    ]

    text = None
    source_name = None

    for dataset_name, split, col in HF_DATASETS:
        result = download_hf_dataset(dataset_name, split, max_chars, col)
        if result:
            text = result
            source_name = dataset_name
            break

    if text is None:
        print('[DATA] HF Datasets nicht verfügbar. Versuche Raw URLs...')
        RAW_URLS = [
            (DATASET_URL, 'input.txt'),
            ("https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt", 'shakespeare2.txt'),
            ("https://gist.githubusercontent.com/phillipj/4944029/raw/75ba2243dd5ec2875febbfa7c2bb1e2e5da5ef3e/shakespeare.txt", 'shakespeare3.txt'),
        ]

        for url, fname in RAW_URLS:
            file_path = os.path.join(DATA_DIR, fname)
            if download_file(url, file_path):
                try:
                    for enc in ['utf-8', 'latin-1']:
                        try:
                            with open(file_path, 'r', encoding=enc) as f:
                                text = f.read()
                            source_name = url
                            break
                        except UnicodeDecodeError:
                            continue
                    if text:
                        break
                except Exception:
                    continue

    if text is None:
        file_path = os.path.join(DATA_DIR, 'input.txt')
        if os.path.exists(file_path):
            print(f'[DATA] Fallback auf vorhandene Datei: {file_path}')
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            source_name = 'shakespeare_fallback'

    if text is None:
        raise RuntimeError("[DATA] KEINE DATASETS VERFUEGBAR! Training kann nicht starten.")

    if max_chars:
        text = text[:max_chars]

    if len(text) < 500000:
        multiplier = min(8, 500000 // max(1, len(text)))
        if multiplier > 1:
            print(f'[DATA] Dataset klein ({len(text):,} Chars). Vervielfache x{multiplier}')
            text = text * multiplier

    stoi, itos, vocab_size = get_character_mapping(text)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    print(f'[DATA] Dataset: {len(data):,} Tokens, Vocab: {vocab_size}, Quelle: {source_name}')

    return data, stoi, itos, vocab_size


def get_code_dataset(max_chars=None, use_code_tokenizer=False):
    """
    Lädt Code-Daten von GitHub/StackOverflow.
    Optional mit speziellem Code-Tokenizer.
    Verwendet jetzt Multi-Domain Code-Loader.
    """
    text = load_code_datasets(max_chars=max_chars)
    if text is None:
        raise RuntimeError('[CODE] Keine Code-Daten verfuegbar.')

    if use_code_tokenizer:
        tokenizer = CodeTokenizer()
        data = tokenizer.encode(text)
        stoi = tokenizer.stoi
        itos = tokenizer.itos
        vocab_size = tokenizer.vocab_size
    else:
        stoi, itos, vocab_size = get_character_mapping(text)
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    print(f'[CODE] Dataset: {len(data):,} Tokens, Vocab: {vocab_size}')
    return data, stoi, itos, vocab_size


def get_mixed_dataset(max_chars=None, code_ratio=0.3):
    """
    Kombiniert Code-Daten mit Text-Daten.
    code_ratio: Anteil Code-Daten (0.0-1.0)
    Verwendet jetzt Multi-Domain Mix.
    """
    if max_chars is None:
        max_chars = 3000000

    try:
        loader = MultiDomainDataset(max_chars_per_domain=max_chars)
        loader.domains['code']['weight'] = code_ratio
        loader.domains['text']['weight'] = 1.0 - code_ratio
        loader.domains['security']['enabled'] = False
        loader.domains['network']['enabled'] = False
        loader.load_all()

        print(f'[DATA] Mixed Dataset: {len(loader.data):,} Tokens, Vocab: {loader.vocab_size}')
        for name in ['code', 'text']:
            start, end = loader.domain_ranges.get(name, (0,0))
            print(f'  {name}: {end-start:,} Tokens')

        return loader.data, loader.stoi, loader.itos, loader.vocab_size
    except Exception as e:
        print(f'[DATA] Multi-Domain Mixed Fehler, fallback: {e}')

    # Legacy Fallback
    text_data, text_stoi, text_itos, text_vocab = get_large_dataset(
        max_chars=int(max_chars * (1 - code_ratio))
    )

    code_data, code_stoi, code_itos, code_vocab = get_code_dataset(
        max_chars=int(max_chars * code_ratio)
    )

    combined_stoi = {**text_stoi, **code_stoi}
    combined_itos = {**text_itos, **code_itos}
    vocab_size = len(combined_stoi)

    combined_data = torch.cat([text_data, code_data])

    print(f'[DATA] Mixed Dataset: {len(combined_data):,} Tokens, Vocab: {vocab_size}')
    print(f'  Text: {len(text_data):,} Tokens, Code: {len(code_data):,} Tokens')

    return combined_data, combined_stoi, combined_itos, vocab_size
