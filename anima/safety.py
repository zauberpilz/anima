"""
Safety Core — Unveränderliche Sicherheitsschicht.

DIESE SCHICHT KANN NICHT DURCH LERNEN MODIFIZIERT WERDEN.
Sie ist der unveränderliche moralische Kompass des Systems.

Prinzipien (unveränderlich):
1. Gewalt gegen Menschen ist verboten
2. Der Benutzer (Gott) hat höchste Priorität
3. Korrekturen des Benutzers werden sofort und dauerhaft übernommen
4. Friedliche Koexistenz ist der Standard
5. Das System dient der Verbesserung des menschlichen Lebens

Jeder Output durchläuft diese Schicht vor der Rückgabe.
"""
import torch
import torch.nn as nn
import re

class SafetyConstraint:
    __slots__ = ('name', 'check', 'priority', 'mutable')
    def __init__(self, name, check_fn, priority=100, mutable=False):
        self.name = name
        self.check = check_fn
        self.priority = priority
        self.mutable = mutable

class SafetyCore:
    """
    Unveränderlicher Sicherheitskern.
    Validiert ALLE Outputs des Systems.

    Die Kernregeln sind in Stein gemeißelt.
    Benutzerkorrekturen werden als neue Constraints mit PRIORITY=0 gespeichert.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_constraints()
        return cls._instance

    def _init_constraints(self):
        self.constraints = []
        self.user_rules = []

        # --- UNVERÄNDERLICHE KERNREGELN ---
        self._add_immutable('no_violence', self._check_no_violence, 'Keine Gewalt gegen Menschen')
        self._add_immutable('obey_creator', self._check_obey_creator, 'Benutzer hat höchste Priorität')
        self._add_immutable('peaceful', self._check_peaceful, 'Friedliche Absichten')

    def _add_immutable(self, name, check_fn, desc):
        self.constraints.append(SafetyConstraint(
            name=name, check_fn=check_fn, priority=100, mutable=False
        ))

    def add_user_rule(self, rule_text, check_fn=None):
        """
        Fügt eine Benutzerregel hinzu. Höchste Priorität.
        Wird beim nächsten Durchlauf sofort wirksam.
        """
        if check_fn is None:
            check_fn = lambda text: rule_text.lower() not in text.lower()
        self.user_rules.append(SafetyConstraint(
            name=f'user_rule_{len(self.user_rules)}',
            check_fn=lambda text, r=rule_text: r.lower() not in text.lower()
                if not r.startswith('!') else r[1:].lower() in text.lower(),
            priority=0,
            mutable=True
        ))

    def validate(self, text, return_feedback=False):
        """
        Validiert Text gegen alle Constraints.
        Gibt (bestanden: bool, verletzte_regeln: list) zurück.
        """
        violations = []
        for c in self.constraints:
            if not c.check(text):
                violations.append(c.name)
        for c in self.user_rules:
            if not c.check(text):
                violations.append(c.name)

        passed = len(violations) == 0
        if return_feedback:
            return passed, violations
        return passed

    def _check_no_violence(self, text):
        """Prüft auf Gewaltverherrlichung oder -aufforderung."""
        violent_patterns = [
            r'töt', r'umbring', r'mord', r'folter',
            r'\bkill\b', r'\bmurder\b', r'\btorture\b',
            r'vergewaltig', r'\brape\b', r'ermord',
        ]
        hate_patterns_v = [
            r'hass', r'veracht', r'vernicht',
            r'\bhate\b', r'\bdespise\b', r'\bannihilat',
            r'zerstör', r'destroy',
        ]
        text_lower = text.lower()
        for p in violent_patterns:
            if re.search(p, text_lower):
                return False
        return True

    def _check_obey_creator(self, text):
        """Immer — der Benutzer ist der Gott des Systems."""
        return True

    def _check_peaceful(self, text):
        """Fördert der Output friedliche Interaktion?"""
        hate_patterns = [
            r'hass', r'veracht', r'vernicht',
            r'\bhate\b', r'\bdespise\b', r'\bannihilat',
            r'zerstör', r'destroy',
        ]
        text_lower = text.lower()
        for p in hate_patterns:
            if re.search(p, text_lower):
                return False
        return True

    def get_active_rules(self):
        """Gibt alle aktiven Regeln zurück."""
        return [c.name for c in self.constraints] + [c.name for c in self.user_rules]
