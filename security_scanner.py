"""
PHASE 34: Security Vulnerability Scanner.
Analysiert Code auf Sicherheitslücken (OWASP Top 10, CWE).
"""

import re
import json
import os
from datetime import datetime

class SecurityScanner:
    """Code vulnerability scanner with CWE pattern matching."""
    
    def __init__(self, report_dir='/home/anima/security_reports'):
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
        
        # CWE Pattern Database
        self.cwe_patterns = {
            'CWE-20: Input Validation': {
                'patterns': [
                    r'input\s*\(\s*\)',
                    r'request\.(get|post|form|params)',
                    r'raw_input\s*\(',
                    r'get\(request\)',
                ],
                'severity': 5.0,
                'description': 'Improper Input Validation',
            },
            'CWE-22: Path Traversal': {
                'patterns': [
                    r'\.\.\/',
                    r'\.\.\\\\', 
                    r'open\(\s*.*user\s*',
                    r'os\.path\.join\(\s*.*\/\s*\)',
                ],
                'severity': 7.5,
                'description': 'Path Traversal',
            },
            'CWE-78: OS Command Injection': {
                'patterns': [
                    r'os\.system\s*\(',
                    r'subprocess\.[a-z]+\s*\(.*shell\s*=',
                    r'eval\s*\(',
                    r'exec\s*\(',
                    r'os\.popen\s*\(',
                ],
                'severity': 9.0,
                'description': 'OS Command Injection',
            },
            'CWE-79: Cross-Site Scripting': {
                'patterns': [
                    r'<script[^>]*>',
                    r'innerHTML\s*=',
                    r'document\.write\s*\(',
                    r'\.html\(\s*.*\$',
                ],
                'severity': 6.5,
                'description': 'XSS',
            },
            'CWE-89: SQL Injection': {
                'patterns': [
                    r'SELECT\s+.*FROM\s+.*WHERE\s+.*[+%]',
                    r'execute\s*\(\s*["\'].*[+%]\s*',
                    r'cursor\.execute\(\s*["\'].*['\"]\s*\+',
                    r'SELECT.*\{.*\}',
                ],
                'severity': 8.5,
                'description': 'SQL Injection',
            },
            'CWE-94: Code Injection': {
                'patterns': [
                    r'eval\s*\(',
                    r'exec\s*\(',
                    r'compile\s*\(',
                    r'__import__\s*\(',
                ],
                'severity': 9.5,
                'description': 'Code Injection',
            },
            'CWE-287: Authentication Issues': {
                'patterns': [
                    r'password\s*=\s*["\'][^"\']+["\']',
                    r'secret_key\s*=',
                    r'auth_token\s*=',
                    r'api_key\s*=',
                ],
                'severity': 7.0,
                'description': 'Hardcoded Credentials',
            },
            'CWE-295: SSL/TLS Issues': {
                'patterns': [
                    r'verify\s*=\s*False',
                    r'check_hostname\s*=\s*False',
                    r'cert_reqs\s*=\s*CERT_NONE',
                    r'ssl\._create_default_https_context',
                ],
                'severity': 6.0,
                'description': 'SSL Verification Disabled',
            },
            'CWE-310: Weak Cryptography': {
                'patterns': [
                    r'md5\s*\(',
                    r'sha1\s*\(',
                    r'DES3?',
                    r'ECB\s*\)',
                    r'AES\.new\(.*ECB',
                ],
                'severity': 5.5,
                'description': 'Weak Cryptographic Algorithm',
            },
            'CWE-502: Deserialization': {
                'patterns': [
                    r'pickle\.load\s*\(',
                    r'yaml\.load\s*\(',
                    r'unserialize?\s*\(',
                    r'JSON\.deserialize',
                ],
                'severity': 8.0,
                'description': 'Unsafe Deserialization',
            },
            'CWE-611: XXE': {
                'patterns': [
                    r'ET/XML',
                    r'XMLParser\(.*resolve_entities',
                    r'lxml\.etree\.parse',
                ],
                'severity': 7.0,
                'description': 'XML External Entity',
            },
            'CWE-798: Hardcoded Credentials': {
                'patterns': [
                    r'password\s*=\s*["\'][^"\']{3,}["\']',
                    r'passwd\s*=\s*["\'][^"\']{3,}["\']',
                    r'pwd\s*=\s*["\'][^"\']{3,}["\']',
                ],
                'severity': 7.5,
                'description': 'Hardcoded Password',
            },
        }
        
        self.results = []
    
    def scan_file(self, file_path):
        """Scan a single file for vulnerabilities."""
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
            return self.scan_code(content, file_path)
        except Exception as e:
            return {'file': file_path, 'error': str(e), 'findings': []}
    
    def scan_code(self, code, source='<string>'):
        """Scan code string for vulnerabilities."""
        findings = []
        lines = code.split('\n')
        
        for cwe_name, cwe_info in self.cwe_patterns.items():
            for pattern in cwe_info['patterns']:
                for i, line in enumerate(lines, 1):
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        findings.append({
                            'cwe': cwe_name,
                            'severity': cwe_info['severity'],
                            'description': cwe_info['description'],
                            'file': source,
                            'line': i,
                            'match': match.group(),
                            'context': line.strip()[:100],
                        })
        
        # Deduplicate by line and CWE
        seen = set()
        unique_findings = []
        for f in findings:
            key = (f['cwe'], f['line'], f['match'])
            if key not in seen:
                seen.add(key)
                unique_findings.append(f)
        
        return {
            'file': source,
            'total_findings': len(unique_findings),
            'severity_score': sum(f['severity'] for f in unique_findings) / max(len(unique_findings), 1),
            'findings': unique_findings,
        }
    
    def generate_report(self, scan_result, output_format='text'):
        """Generate a formatted report."""
        if output_format == 'text':
            return self._report_text(scan_result)
        elif output_format == 'json':
            return json.dumps(scan_result, indent=2)
        elif output_format == 'markdown':
            return self._report_markdown(scan_result)
    
    def _report_text(self, result):
        """Generate text report."""
        lines = [f"=== SECURITY SCAN REPORT ===",
                 f"File: {result['file']}",
                 f"Findings: {result['total_findings']}",
                 f"Risk Score: {result['severity_score']:.1f}/10",
                 f"---"]
        for f in sorted(result['findings'], key=lambda x: x['severity'], reverse=True):
            lines.append(f"[{f['severity']:.0f}/10] {f['cwe']}")
            lines.append(f"       Line {f['line']}: {f['context']}")
        return '\n'.join(lines)
    
    def _report_markdown(self, result):
        """Generate markdown report."""
        lines = [f"# Security Scan Report",
                 f"**File:** `{result['file']}`",
                 f"**Total Findings:** {result['total_findings']}",
                 f"**Risk Score:** {result['severity_score']:.1f}/10",
                 f"",
                 f"## Findings",
                 f"| Severity | CWE | Line | Match |",
                 f"|----------|-----|------|-------|"]
        for f in sorted(result['findings'], key=lambda x: x['severity'], reverse=True):
            lines.append(f"| {f['severity']:.0f}/10 | {f['cwe']} | {f['line']} | `{f['match']}` |")
        return '\n'.join(lines)
    
    def scan_directory(self, dir_path, pattern='*.py'):
        """Scan entire directory for vulnerabilities."""
        import glob
        all_results = []
        total_findings = 0
        
        files = glob.glob(os.path.join(dir_path, '**', pattern), recursive=True)
        for file_path in files:
            result = self.scan_file(file_path)
            total_findings += result['total_findings']
            all_results.append(result)
        
        return {
            'directory': dir_path,
            'files_scanned': len(files),
            'total_findings': total_findings,
            'results': all_results,
        }


if __name__ == '__main__':
    print("=== SECURITY SCANNER TEST ===")
    scanner = SecurityScanner()
    
    # Test code with vulnerabilities
    test_code = '''
import os
import pickle

def vulnerable_function(user_input):
    # SQL Injection
    query = "SELECT * FROM users WHERE id = " + user_input
    
    # Command Injection
    os.system("rm -rf " + user_input)
    
    # Unsafe deserialization
    data = pickle.load(open("data.pkl", "rb"))
    
    # Hardcoded password
    password = "super_secret_123"
    
    return data
'''
    
    result = scanner.scan_code(test_code, 'test.py')
    print(scanner.generate_report(result, 'text'))
    
    print("\n\n=== Markdown Report ===")
    print(scanner.generate_report(result, 'markdown'))
