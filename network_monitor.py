"""
PHASE 33: Network Traffic Monitor + Anomaly Detection.
Analysiert PCAPs/NetFlow und erkennt Anomalien in Echtzeit.
"""

import time
import json
import os
from datetime import datetime

class NetworkMonitor:
    """Network traffic monitor with anomaly detection."""
    
    def __init__(self, alert_file='/home/anima/network_alerts.json', max_packets=10000):
        self.alert_file = alert_file
        self.max_packets = max_packets
        self.packet_count = 0
        self.anomaly_count = 0
        self.flow_table = {}  # src:dst -> flow stats
        self.alerts = []
        
        # Try to import scapy
        self.scapy_available = False
        try:
            from scapy.all import rdpcap, IP, TCP, UDP, ICMP
            self.rdpcap = rdpcap
            self.IP = IP
            self.TCP = TCP  
            self.UDP = UDP
            self.ICMP = ICMP
            self.scapy_available = True
        except ImportError:
            pass
    
    def parse_pcap(self, pcap_path):
        """Parse PCAP file and return list of packet dicts."""
        if not self.scapy_available:
            return self._parse_pcap_fallback(pcap_path)
        return self._parse_pcap_scapy(pcap_path)
    
    def _parse_pcap_scapy(self, pcap_path):
        """Parse PCAP using scapy."""
        packets = []
        try:
            for pkt in self.rdpcap(pcap_path):
                if len(packets) >= self.max_packets:
                    break
                pkt_dict = self._packet_to_dict(pkt)
                if pkt_dict:
                    packets.append(pkt_dict)
        except Exception as e:
            print(f"[NETMON] PCAP parse error: {e}")
        return packets
    
    def _packet_to_dict(self, pkt):
        """Convert scapy packet to dict."""
        result = {
            'time': time.time(),
            'src': '?',
            'dst': '?',
            'src_port': 0,
            'dst_port': 0,
            'proto': '?',
            'len': len(pkt),
            'flags': [],
        }
        if self.IP in pkt:
            result['src'] = pkt[self.IP].src
            result['dst'] = pkt[self.IP].dst
            result['proto'] = pkt[self.IP].proto
        if self.TCP in pkt:
            result['src_port'] = pkt[self.TCP].sport
            result['dst_port'] = pkt[self.TCP].dport
            result['flags'] = self._tcp_flags(pkt[self.TCP].flags)
        elif self.UDP in pkt:
            result['src_port'] = pkt[self.UDP].sport
            result['dst_port'] = pkt[self.UDP].dport
        return result
    
    def _tcp_flags(self, flags):
        """Convert TCP flags to list of strings."""
        flag_names = []
        if flags & 0x01: flag_names.append('FIN')
        if flags & 0x02: flag_names.append('SYN')
        if flags & 0x04: flag_names.append('RST')
        if flags & 0x08: flag_names.append('PSH')
        if flags & 0x10: flag_names.append('ACK')
        if flags & 0x20: flag_names.append('URG')
        return flag_names
    
    def _parse_pcap_fallback(self, pcap_path):
        """Parse PCAP with minimal tshark call."""
        packets = []
        try:
            import subprocess
            result = subprocess.run(
                ['tshark', '-r', pcap_path, '-T', 'fields',
                 '-e', 'frame.number', '-e', 'ip.src', '-e', 'ip.dst',
                 '-e', 'tcp.srcport', '-e', 'tcp.dstport',
                 '-e', 'udp.srcport', '-e', 'udp.dstport',
                 '-e', 'frame.len',
                 '-E', 'separator=|', '-E', 'header=n'],
                capture_output=True, text=True, timeout=30
            )
            for line in result.stdout.strip().split('\n'):
                if not line or '|' not in line:
                    continue
                parts = line.split('|')
                pkt = {
                    'time': time.time(),
                    'src': parts[1] if len(parts) > 1 else '?',
                    'dst': parts[2] if len(parts) > 2 else '?',
                    'src_port': int(parts[3] or parts[5] or 0),
                    'dst_port': int(parts[4] or parts[6] or 0),
                    'proto': 'TCP' if parts[3] else 'UDP' if parts[5] else '?',
                    'len': int(parts[7]) if len(parts) > 7 else 0,
                }
                packets.append(pkt)
        except:
            pass
        return packets
    
    def detect_anomalies(self, packets):
        """Detect network anomalies in packet stream."""
        anomalies = []
        
        # SYN flood detection
        syn_count = sum(1 for p in packets if 'SYN' in p.get('flags', []))
        if syn_count > 100:
            anomalies.append({
                'type': 'SYN_FLOOD',
                'severity': 'HIGH',
                'count': syn_count,
                'description': f'Possible SYN flood: {syn_count} SYN packets',
                'time': time.time(),
            })
        
        # Port scan detection
        dst_ports = {}
        for p in packets:
            key = f"{p['src']}:{p['dst']}"
            if p['dst_port']:
                dst_ports.setdefault(key, set()).add(p['dst_port'])
        for key, ports in dst_ports.items():
            if len(ports) > 20:
                anomalies.append({
                    'type': 'PORT_SCAN',
                    'severity': 'MEDIUM',
                    'target': key,
                    'ports_scanned': len(ports),
                    'description': f'Port scan detected: {len(ports)} unique ports to {key}',
                    'time': time.time(),
                })
        
        # Data exfiltration detection
        large_packets = [p for p in packets if p.get('len', 0) > 1000]
        if len(large_packets) > 50:
            anomalies.append({
                'type': 'DATA_EXFIL',
                'severity': 'HIGH',
                'count': len(large_packets),
                'description': f'Large packet burst: {len(large_packets)} packets > 1000 bytes',
                'time': time.time(),
            })
        
        return anomalies
    
    def packets_to_text(self, packets):
        """Convert packets to AGI-readable text format."""
        lines = []
        for p in packets:
            line = f"[PACKET] src={p['src']}:{p['src_port']} -> dst={p['dst']}:{p['dst_port']} proto={p['proto']} len={p['len']}"
            if p.get('flags'):
                line += f" flags={'|'.join(p['flags'])}"
            lines.append(line)
        return '\n'.join(lines[:100])  # Limit to 100 packets
    
    def log_alert(self, anomaly):
        """Write anomaly alert to file."""
        self.alerts.append(anomaly)
        try:
            alerts = []
            if os.path.exists(self.alert_file):
                with open(self.alert_file, 'r') as f:
                    alerts = json.load(f)
            alerts.append(anomaly)
            # Keep last 100 alerts
            alerts = alerts[-100:]
            with open(self.alert_file, 'w') as f:
                json.dump(alerts, f, indent=2)
        except:
            pass
    
    def generate_report(self):
        """Generate a text summary of current network state."""
        return f"""=== NETWORK MONITOR REPORT ===
Time: {datetime.now().isoformat()}
Packets Analyzed: {self.packet_count}
Anomalies Detected: {self.anomaly_count}
Active Flows: {len(self.flow_table)}
Active Alerts: {len([a for a in self.alerts if a['severity'] == 'HIGH'])} HIGH
"""


def get_network_demo():
    """Generate demo network data for testing."""
    import random
    packets = []
    protocols = ['TCP', 'UDP', 'ICMP']
    ips = [f'192.168.1.{i}' for i in range(1, 255)]
    ports = [80, 443, 22, 8080, 53, 3306, 5432, 6379, 27017]
    
    # Normal traffic
    for _ in range(50):
        packets.append({
            'src': random.choice(ips),
            'dst': '10.0.0.1',
            'src_port': random.randint(1024, 65535),
            'dst_port': random.choice(ports),
            'proto': random.choice(protocols),
            'len': random.randint(40, 1500),
            'flags': ['SYN', 'ACK'] if random.random() > 0.5 else ['ACK'],
        })
    
    # Anomalous traffic (SYN flood)
    for _ in range(150):
        packets.append({
            'src': '10.0.0.99',
            'dst': '192.168.1.1',
            'src_port': random.randint(1024, 65535),
            'dst_port': 80,
            'proto': 'TCP',
            'len': 40,
            'flags': ['SYN'],
        })
    
    return packets


if __name__ == '__main__':
    print("=== NETWORK MONITOR TEST ===")
    monitor = NetworkMonitor()
    packets = get_network_demo()
    print(f"Parsed {len(packets)} packets")
    
    anomalies = monitor.detect_anomalies(packets)
    print(f"Detected {len(anomalies)} anomalies:")
    for a in anomalies:
        print(f"  [{a['severity']}] {a['type']}: {a['description']}")
        monitor.log_alert(a)
    
    print("\nText format sample:")
    text = monitor.packets_to_text(packets[:5])
    print(text)
    print(f"\nReport:\n{monitor.generate_report()}")
