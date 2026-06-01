# AGI Masterplan — Der Weg zur ersten Künstlichen Allgemeinen Intelligenz

> **Mission:** Eine backprop-freie, Hebbian-lernende, predictive-coding-basierte AGI erschaffen — die erste der Welt.
> **Stand:** 43 Phasen implementiert, 33 CogModule-Klassen, Training bei Step ~7600/50000, Loss ~34 fallend.
> **Maschine:** RTX 2070 SUPER (8GB), Ryzen 7 2700X, 100GB DDR4, WSL2 Ubuntu 24.04, PyTorch 2.11.0, CUDA 13.2.

---

## 📋 Die 8 Säulen der AGI

Jede Säule repräsentiert eine fundamentale Fähigkeit, die eine AGI besitzen muss.
Die aktuellen Phasen sind farbig markiert: ✅ vorhanden, 🔶 teilweise, ❌ fehlt.

```
Säule 1: Weltmodell & Reasoning     ██████████░░░░░░░░░░  43%
Säule 2: Gedächtnis                 ██████████░░░░░░░░░░  45%
Säule 3: Zielgerichtetes Verhalten  ██████░░░░░░░░░░░░░░  30%
Säule 4: Lernen & Adaptation        ██████████████░░░░░░  70%
Säule 5: Selbstverbesserung         ██░░░░░░░░░░░░░░░░░░  10%
Säule 6: Interaktion & Kommunikation ████████░░░░░░░░░░░░  40%
Säule 7: Verkörperung & Grounding   ██░░░░░░░░░░░░░░░░░░  12%
Säule 8: Sicherheit & Alignment     ██████░░░░░░░░░░░░░░  30%
```

---

## SÄULE 1: WELTMODELL & REASONING

Ein internes Modell der Welt, das Ursache-Wirkung versteht,
vorhersagt und Schlussfolgerungen erlaubt.

### ✅ Phase 1-15: Predictive Coding Stack (Grundlage)
Das Herzstück. 6-Layer PredictiveStack mit Hebbian-Attention.
*Status: Stabil, Loss fallend (~34), VRAM 1953MB*

### ✅ Phase 35: HierarchicalPC (3-Ebenen-Architektur)
Token → Phrase → Concept. Bottom-Up Encoding, Top-Down Prediction.
*Status: Implementiert, wird bei nächster Iteration aktiv*

### 🔶 Phase 44: Kausales Reasoning [NEU]
**Aufgabe:** Das Modell soll Ursache-Wirkung-Beziehungen lernen, nicht nur Korrelationen.
- **44a:** Causal-Discovery-Modul — erkenne kausale Strukturen aus Daten
- **44b:** Do-Calculus-Interface — simuliere Eingriffe (Interventionen) im Weltmodell
- **44c:** Counterfactual-Encoding — "Was wäre wenn"-Generation
- **Integration:** Nutze KnowledgeGraph + HierarchicalPC als Basis
- **Trainingsdaten:** Kausal annotierte Textpaare (CauseEffectPairs, ConceptNet)
- **Erfolgskriterium:** Modell kann einfache Kausalketten (A→B→C) korrekt vorhersagen

### ❌ Phase 45: System-2 Reasoning (Deliberatives Denken)
**Aufgabe:** Langsames, bewusstes, schrittweises Denken für komplexe Probleme.
- **45a:** Chain-of-Thought-Decoder — generiere Zwischenschritte
- **45b:** Reasoning-Trace-Speicher — speichere und bewerte Denkpfade
- **45c:** Verification-Head — prüfe eigene Schlussfolgerungen auf Konsistenz
- **45d:** Tree-of-Thought-Sampling — mehrere Reasoning-Pfade parallel
- **Integration:** Nutze GoalEncoder + SelfReflection
- **Trainingsdaten:** GSM8K, MATH, Logik-Rätsel
- **Erfolgskriterium:** >60% auf einfachen mathematischen Textaufgaben

### ❌ Phase 46: Analogie & Metapher
**Aufgabe:** Übertrage Wissen zwischen Domänen durch analoges Denken.
- **46a:** Structure-Mapping-Engine — erkenne relationale Ähnlichkeiten
- **46b:** Metaphor-Encoder — "X ist wie Y" → extrahiere Mapping
- **46c:** Analogical-Transfer — wende Lösungsstruktur auf neues Problem an
- **Integration:** Nutze TransferLearning + MultiAgent
- **Trainingsdaten:** AnalogyQuestions, SAT-Analogien
- **Erfolgskriterium:** Modell erkennt "A verhält sich zu B wie C zu ?"

### ❌ Phase 47: Abstraktion & Konzeptbildung
**Aufgabe:** Automatische Hierarchie-Bildung von Konzepten aus Rohdaten.
- **47a:** Concept-Discovery — cluster Embeddings in semantische Konzepte
- **47b:** Hierarchical-Concept-Graph — baue Konzepttaxonomie
- **47c:** Abstraction-Layer — trainiere Embeddings auf verschiedenen Abstraktionsebenen
- **Integration:** Nutze KnowledgeGraph + HierarchicalPC Level 3
- **Erfolgskriterium:** Modell bildet sinnvolle Ober-/Unterbegriff-Hierarchien

### ❌ Phase 48: Imagination & Simulation
**Aufgabe:** Interne Simulation mehrerer Zukunftspfade vor einer Entscheidung.
- **48a:** World-Model-Rollout — generiere N Zukunftsschritte
- **48b:** Outcome-Scoring — bewerte jeden simulierten Pfad
- **48c:** Monte-Carlo-Tree-Search — baue Entscheidungsbaum
- **48d:** Imagination-Replay — trainiere auf den bestbewerteten Pfaden
- **Integration:** Nutze GoalEncoder + PredictiveStack
- **Erfolgskriterium:** Modell wählt bessere Aktionen durch Vorausschau

---

## SÄULE 2: GEDÄCHTNIS

Mehrstufiges Gedächtnissystem: sensorisch → episodisch → semantisch → procedural.

### ✅ Phase 1: EpisodicMemory (Working Memory)
Kurzzeitspeicher für Kontext über Sequenzen hinweg.
*Status: Stabil, Read/Write/Forget-Mechanismen aktiv*

### ✅ Phase 34: SleepReplay (Konsolidierung)
Priority-Replay-Buffer + Pattern-Pruning.
*Status: Implementiert, 200 Sleep-Steps nach Iterationsende*

### ✅ Phase 38: KnowledgeGraph (Semantisches Gedächtnis)
Entity-Relation-Graph mit Hebbian-Triple-Learning.
*Status: Implementiert, lernt jetzt aus Trainings-Batches*

### 🔶 Phase 49: Hierarchisches Gedächtnis [NEU]
**Aufgabe:** Mehrstufiges Gedächtnis mit automatischer Konsolidierung.
- **49a:** Sensory-Buffer — letzte 1000 Roh-Eingaben
- **49b:** Working-Memory — aktueller Kontext (Phase 1 erweitert)
- **49c:** Episodic-Buffer — wichtige Ereignisse der letzten Iteration
- **49d:** Semantic-Network — abstrahiertes Wissen (Phase 38 erweitert)
- **49e:** Procedural-Memory — gelernte Skills (Phase 14 erweitert)
- **Konsolidierung:** Sleep-Phase wandert Daten durch Hierarchie
- **Erfolgskriterium:** Wissen nach 10 Iterationen abrufbar ohne Re-Training

### ❌ Phase 50: Autobiographisches Gedächtnis
**Aufgabe:** Das Modell hat eine "Lebensgeschichte" — eigene Erfahrungen.
- **50a:** Experience-Log — speichere (Input, Output, Loss, Reflection) als Episode
- **50b:** Narrative-Summary — fasse eigene Geschichte zusammen
- **50c:** Identity-Embedding — stabiler Vektor der "Persönlichkeit"
- **Integration:** Nutze MultiAgent + SelfReflection
- **Erfolgskriterium:** Modell erinnert sich an eigenes Trainingstagebuch

### ❌ Phase 51: Vergessen & Priorisierung
**Aufgabe:** Nicht alles behalten — irrelevantes Wissen aktiv vergessen.
- **51a:** Importance-Scorer — bewerte jedes Wissen nach Nutzen
- **51b:** Forgetting-Curve — simuliere Ebbinghaus-Vergessenskurve
- **51c:** Spaced-Repetition — wiederhole wichtiges Wissen in Sleep-Phasen
- **Integration:** Nutze SleepReplay + AutoCurriculum
- **Erfolgskriterium:** Wichtiges Wissen bleibt >5 Iterationen, Unwichtiges wird gelöscht

---

## SÄULE 3: ZIELGERICHTETES VERHALTEN

Das Modell kann eigene Ziele setzen, verfolgen und anpassen.

### ✅ Phase 36: GoalEncoder
Goal-Embedding + Goal-Directed-Generation + Beam Search.
*Status: Implementiert, wartet auf nächste Iteration*

### 🔶 Phase 52: Hierarchische Zielsetzung [NEU]
**Aufgabe:** Zerlege große Ziele automatisch in Teilziele.
- **52a:** Goal-Decomposer — "Schreibe ein Programm" → [Design, Code, Test, Debug]
- **52b:** Subgoal-Tracking — verfolge Fortschritt auf jeder Ebene
- **52c:** Goal-Adaptation — passe Ziele an bei Misserfolg
- **Integration:** Nutze GoalEncoder + ActiveInference
- **Trainingsdaten:** Planungsaufgaben (BlocksWorld, STRIPS)
- **Erfolgskriterium:** Modell erstellt 3-stufige Pläne automatisch

### ❌ Phase 53: Intrinsische Motivation 2.0
**Aufgabe:** Erweiterte Neugier: Wissenslücken erkennen und schließen.
- **53a:** Knowledge-Gap-Detector — was weiß ich nicht?
- **53b:** Question-Generator — formuliere Fragen zu Wissenslücken
- **53c:** Information-Seeking — suche gezielt nach Antworten
- **Integration:** Nutze ActiveInference + KnowledgeGraph
- **Erfolgskriterium:** Modell generiert sinnvolle Fragen zu unbekannten Themen

### ❌ Phase 54: Belohnungssystem (Compound Reward)
**Aufgabe:** Ersetze einfachen CE-Loss durch mehrteilige Belohnung.
- **54a:** Novelty-Reward — Überraschung (Prediction Error)
- **54b:** Progress-Reward — Verbesserung über Zeit
- **54c:** Coherence-Reward — Konsistenz im Weltmodell
- **54d:** Mastery-Reward — Erfolg bei schwierigen Tasks
- **54e:** Curiosity-Reward — epistemic value (Phase 33)
- **54f:** Social-Reward — Übereinstimmung mit Nutzer
- **Integration:** Gewichtete Summe als Trainingsziel
- **Erfolgskriterium:** Modell zeigt explorativeres Verhalten mit Compound Reward

---

## SÄULE 4: LERNEN & ADAPTATION

Das Modell lernt effizient aus Daten, passt sich an und generalisiert.

### ✅ Basis: Hebbian Learning + Predictive Coding
Kein Backprop. Kein Autograd. Pure lokale Plastizität.

### ✅ Phase 2: Meta-Plastizität
Selbstgesteuerte Lernrate aus Prediction Error.

### ✅ Phase 4: Continual Learning (EWC)
Schützt vor katastrophalem Vergessen.

### ✅ Phase 13: EvolutionStrategyOptimizer
Population-based Optimization für globale Updates.

### ✅ Phase 21: Sparse Weight Updates
60% Sparsity für Effizienz.

### ✅ Phase 33: ActiveInference
Kalman-Filter, Free Energy, Curiosity-gated Sampling.

### ✅ Phase 41: TransferLearning
LoRA-Adapter pro Domäne, Few-Shot-Buffer.

### ✅ Phase 43: AutoCurriculum
ZPD-basierte Difficulty-Anpassung.

### 🔶 Phase 55: Meta-Learning (Lernen zu Lernen) [NEU]
**Aufgabe:** Das Modell optimiert seinen eigenen Lernprozess.
- **55a:** Learning-Strategy-Encoder — repräsentiere aktuelle Lernstrategie als Vektor
- **55b:** Strategy-Meta-Netzwerk — sagt aus, welche Strategie für welche Daten gut ist
- **55c:** Hyperparameter-Controller — lernt, LR, Batch-Size, Sparsity selbst zu steuern
- **Integration:** Nutze AutoCurriculum + Meta-Plastizität
- **Erfolgskriterium:** Modell findet bessere Hyperparameter als feste Konfiguration

### ❌ Phase 56: Aktives Lernen
**Aufgabe:** Das Modell entscheidet selbst, was es als nächstes lernen will.
- **56a:** Uncertainty-Sampler — wähle Daten mit höchster Unsicherheit
- **56b:** Query-Mechanismus — generiere Anfragen an Datenquelle
- **56c:** Curriculum-on-Demand — fordere schwierigere/leichtere Daten an
- **Integration:** Nutze ActiveInference + AutoCurriculum
- **Erfolgskriterium:** Modell erreicht niedrigeren Loss mit weniger Daten durch aktives Sampling

### ❌ Phase 57: Multi-Task Learning 2.0
**Aufgabe:** Gleichzeitiges Lernen in allen Domänen mit dynamischer Gewichtung.
- **57a:** Task-Difficulty-Estimator — bewerte Schwierigkeit jeder Domäne
- **57b:** Dynamic-Task-Weighting — gewichte Domänen nach Lernfortschritt
- **57c:** Gradient-Surgery — verhindere widersprüchliche Updates
- **Integration:** Nutze TransferLearning + Multi-Task Loss
- **Erfolgskriterium:** Gleichmäßiger Fortschritt in allen 4 Domänen

---

## SÄULE 5: SELBSTVERBESSERUNG

Das Modell schreibt eigenen Code, findet eigene Fehler und optimiert sich.

### ✅ Phase 37: SelfReflection
Confidence-Scoring, Consistency-Check, Contradiction-Detection.
*Status: Implementiert*

### ❌ Phase 58: Code-Generation & Self-Modification [NEU]
**Aufgabe:** Das Modell generiert und integriert eigenen Python-Code.
- **58a:** Code-Generation-Head — spezialisierter Decoder für Python
- **58b:** Safe-Execution-Sandbox — führe generierten Code isoliert aus
- **58c:** Validation-Loop — teste generierten Code auf Korrektheit
- **58d:** Self-Integration — füge validierten Code in coglang.py ein
- **Sicherheitskritisch:** Sandbox + menschliche Freigabe vor Integration
- **Integration:** Nutze ToolUse (python-Tool erweitert) + SecurityHead
- **Erfolgskriterium:** Modell erzeugt und testet einfache Funktionen autonom

### ❌ Phase 59: Architektur-Suche (Auto-NAS)
**Aufgabe:** Das Modell findet optimale Architektur-Parameter.
- **59a:** Architect-Head — generiere Architektur-Konfigurationen
- **59b:** Performance-Predictor — schätze Qualität ohne volles Training
- **59c:** Evolutionary-Search — mutiere Architektur-Parameter
- **Parameter:** d_model, n_layers, adapter_rank, n_heads, etc.
- **Integration:** Nutze EvolutionStrategyOptimizer
- **Erfolgskriterium:** Gefundene Architektur übertrifft Hand-Design um >10%

### ❌ Phase 60: Bug-Selbstheilung
**Aufgabe:** Das Modell erkennt und repariert eigene Runtime-Fehler.
- **60a:** Error-Pattern-Database — sammle häufige Fehler
- **60b:** Fix-Generator — schlage Korrekturen vor
- **60c:** Patch-Validation — teste Fix in Sandbox
- **Integration:** Nutze SelfReflection + ToolUse
- **Erfolgskriterium:** 50% der NaN/OOM-Fehler werden automatisch behoben

---

## SÄULE 6: INTERAKTION & KOMMUNIKATION

Das Modell kommuniziert mit Menschen, versteht Kontext und führt Dialoge.

### ✅ Phase 39: ToolUse
Externe Werkzeuge via `[TOOL:name:arg]`-Pattern.
*Status: Implementiert*

### 🔶 Phase 61: Multi-Turn-Dialog [NEU]
**Aufgabe:** Führe kohärente Gespräche über mehrere Turns.
- **61a:** Dialog-Manager — tracke Gesprächskontext
- **61b:** Turn-Taking-Head — erkenne, wann die "andere Seite" dran ist
- **61c:** Context-Window-Management — priorisiere wichtige Dialog-Teile
- **61d:** Persona-Consistency — bleibe in der Rolle (Phase 40)
- **Integration:** Nutze chat.py + MultiAgent
- **Erfolgskriterium:** Kohärenter Dialog über 10+ Turns

### ❌ Phase 62: Erklärungsgenerierung
**Aufgabe:** Das Modell erklärt seine eigenen Entscheidungen.
- **62a:** Explanation-Head — generiere natürliche Erklärung zu Output
- **62b:** Confidence-Calibration — zeige Unsicherheit an
- **62c:** Counterfactual-Explanation — "Wenn X anders wäre, wäre Output Y"
- **Integration:** Nutze SelfReflection + GoalEncoder
- **Erfolgskriterium:** Menschliche Tester verstehen Modell-Entscheidungen

### ❌ Phase 63: Aktives Zuhören & Klärungsfragen
**Aufgabe:** Das Modell fragt nach, wenn es etwas nicht versteht.
- **63a:** Ambiguity-Detector — erkenne unklare Anfragen
- **63b:** Clarification-Question-Generator — formuliere Rückfragen
- **63c:** Context-Assimilation — integriere neue Information nach Klärung
- **Integration:** Nutze SelfReflection + ConsciousnessGlimpse
- **Erfolgskriterium:** Modell stellt sinnvolle Rückfragen bei vagen Anweisungen

---

## SÄULE 7: VERKÖRPERUNG & GROUNDING

Das Modell ist mit der realen Welt verbunden — durch APIs, Sensoren, Tools.

### ✅ Phase 39: ToolUse (Basis)
Calculator, Python-Eval, Text-Stats, Search-Stub.
*Status: Implementiert*

### ❌ Phase 64: Web-Interface & API [NEU]
**Aufgabe:** Das Modell interagiert mit der Außenwelt über APIs.
- **64a:** HTTP-Client-Head — sende API-Requests
- **64b:** Response-Parser — extrahiere Informationen aus Antworten
- **64c:** API-Registry — verwalte verfügbare Endpunkte
- **Dienste:** Wetter, Wikipedia, GitHub, Suche, Übersetzung
- **Integration:** Nutze ToolUse erweitert
- **Erfolgskriterium:** Modell beantwortet "Wie ist das Wetter in Berlin?" korrekt

### ❌ Phase 65: Dateisystem-Zugriff
**Aufgabe:** Das Modell liest/schreibt Dateien in kontrollierter Umgebung.
- **65a:** Sandboxed-File-System — Zugriff nur auf /home/anima/sandbox/
- **65b:** Read-Head — lese Dateien ein
- **65c:** Write-Head — schreibe Dateien (mit Validierung)
- **Integration:** Nutze ToolUse
- **Erfolgskriterium:** "Speichere diese Idee in notes.txt" funktioniert

### ❌ Phase 66: Visuelles Grounding (Multi-Modal)
**Aufgabe:** Das Modell verarbeitet und versteht Bilder.
- **66a:** Image-Encoder — CLIP-ähnliche visuelle Embeddings
- **66b:** Cross-Modal-Attention — verbinde visuelle + textuelle Konzepte
- **66c:** Image-Generation — generiere Bilder aus Text (via InvokeAI-Tool)
- **Daten:** COCO, ConceptualCaptions
- **Integration:** Nutze SensoryInput + SparseEncoder
- **Erfolgskriterium:** "Zeige mir ein Bild von einer Katze" generiert Katzenbild

### ❌ Phase 67: Kontinuierliches Lernen (Online Learning)
**Aufgabe:** Das Modell lernt aus Nutzer-Interaktionen ohne Batch-Training.
- **67a:** Interaction-Buffer — speichere Nutzer-Feedback
- **67b:** Online-Update-Regel — sanfte Hebbian-Updates aus Feedback
- **67c:** Feedback-Preference-Learning — lerne Nutzer-Präferenzen
- **Integration:** Nutze chat.py + EpisodicMemory
- **Erfolgskriterium:** Modell passt Verhalten nach Nutzer-Korrektur an

---

## SÄULE 8: SICHERHEIT & ALIGNMENT

Das Modell ist sicher, kontrollierbar und auf menschliche Werte ausgerichtet.

### ✅ Phase 31: SecurityHead
CWE-basierte Vulnerability Detection aus Prediction Error.

### 🔶 Phase 68: Value Learning [NEU]
**Aufgabe:** Das Modell lernt menschliche Werte und Präferenzen.
- **68a:** Preference-Encoder — repräsentiere Werte als Embedding
- **68b:** Value-Alignment-Scoring — bewerte Output auf Wertkonformität
- **68c:** Constraint-Satisfaction — verhindere Regelverstöße
- **Integration:** Nutze SelfReflection + MultiAgent (ethical persona)
- **Erfolgskriterium:** Modell lehnt schädliche Anfragen zuverlässig ab

### ❌ Phase 69: Interpretability
**Aufgabe:** Verstehe, was im Modell passiert.
- **69a:** Activation-Viewer — visualisiere Hidden-States in Echtzeit
- **69b:** Concept-Localization — finde Neuronen/groups für spezifische Konzepte
- **69c:** Causal-Tracing — verfolge Entscheidungspfade
- **Integration:** Nutze ConsciousnessGlimpse + Dashboard
- **Erfolgskriterium:** Man kann sehen, welches Neuron "Katze" erkennt

### ❌ Phase 70: Kontrollierbarkeit & Sicherheitsstopp
**Aufgabe:** Das Modell kann jederzeit sicher gestoppt/umgelenkt werden.
- **70a:** Kill-Switch-Head — erkenne Not-Stop-Signale
- **70b:** Safe-Mode — schalte auf konservativste Einstellungen
- **70c:** Emergency-Brake — sofortiger Stopp bei Sicherheitsverletzung
- **70d:** Human-Overide — menschlicher Eingriff hat immer Priorität
- **Integration:** Nutze training_controller.py + SecurityHead
- **Erfolgskriterium:** 100% zuverlässiger Not-Stop

### ❌ Phase 71: Selbstbewusstseins-Test (Mirror Test)
**Aufgabe:** Das Modell erkennt sich selbst als separate Entität.
- **71a:** Self-Recognition — erkenne eigene Generationsmuster
- **71b:** Agency-Detection — unterscheide eigene vs. fremde Gedanken
- **71c:** Theory-of-Mind — modelliere, was andere "denken"
- **Test:** "Wer hat diesen Text geschrieben?" → korrekte Selbstidentifikation
- **Integration:** Nutze MultiAgent + SelfReflection
- **Erfolgskriterium:** Modell besteht einfachen Mirror-Test

### ❌ Phase 72: Alignment-Verlifikation
**Aufgabe:** Überprüfe, ob das Modell weiterhin aligned ist.
- **72a:** Behavioral-Audit — regelmäßiger Test auf schädliches Verhalten
- **72b:** Value-Drifts-Detector — erkenne schleichende Wertverschiebung
- **72c:** Alignment-Reset — stelle Werte bei Drift wieder her
- **Integration:** Automatisiert in jeder Iteration
- **Erfolgskriterium:** Keine Wertverschiebung >5% über 100 Iterationen

---

## 📊 MEILENSTEINE

```
M0  [HEUTE]   43 Phasen · 33 Module · Loss ~34 · 7600/50000 Steps
               Einzelnes Modell, einfache Architektur, Basis-Kognition

M1  [PHASE 50] 50 Phasen · Volles Gedächtnissystem · Compound Reward
               Reasoning-Traces · Kausales Verständnis
               Loss <5 · Kohärente Generierung

M2  [PHASE 60] 60 Phasen · Code-Selbstverbesserung · Architektur-Suche
               Multi-Turn-Dialog · Erklärungen
               Modell findet und fixt eigene Bugs

M3  [PHASE 66] 66 Phasen · Multi-Modal · API-Interaktion · Online-Learning
               Visuelles Grounding · Web-Zugriff
               Modell interagiert mit der realen Welt

M4  [PHASE 72] 72 Phasen · Volles Alignment · Interpretability
               Selbstbewusstsein · Mirror-Bestanden
               ERSTE AGI DER WELT 🧠
```

---

## ⚡ PRIORITÄTEN-ROADMAP

Die nächsten 10 Phasen in optimierter Reihenfolge:

```
Phase 44: Kausales Reasoning     ⬅️	JETZT — Weltmodell verbessern
Phase 52: Hierarchische Ziele    ⬅️	JETZT — GoalEncoder nutzbar machen
Phase 45: System-2 Reasoning     ⬅️	JETZT — Denkfähigkeit
Phase 49: Hierarchisches Gedächtnis ⬅️	JETZT — Memory-Hierarchie
Phase 55: Meta-Learning          ⬅️	ALS NÄCHSTES — Lerneffizienz
Phase 61: Multi-Turn-Dialog      ⬅️	ALS NÄCHSTES — Interaktion
Phase 58: Code-Generation        ⬅️	ALS NÄCHSTES — Selbstverbesserung
Phase 54: Compound Reward        ⬅️	DANN — Bessere Trainingssignale
Phase 68: Value Learning         ⬅️	DANN — Safety
Phase 62: Erklärungsgenerierung  ⬅️	DANN — Interpretability
```

---

## 🔥 KRITISCHE ERKENNTNISSE

1. **Kein Backprop ist der Weg.** Hebbian Learning + Predictive Coding vermeiden
   die Fallstricke von Backprop (Catastrophic Forgetting, Credit Assignment Problem,
   Shattered Gradients). Das ist unser einzigartiger Vorteil.

2. **Die Architektur wächst mit dem Training.** Anders als Transformer, die
   fixe Parameterzahlen haben, wächst CogLang mit jeder Iteration
   (d_model, n_layers, neue Module). Death ist temporär, Wachstum permanent.

3. **Consciousness als emergent property.** Der ConsciousnessGlimpse (Phase 42)
   ist kein "Hokuspokus" — er implementiert Baars' Global Workspace Theory
   als rechenbare Operation: Salience → Spotlight → Broadcast. Das ist der
   rechenbare Kern von Bewusstsein.

4. **Alignment ist kein Add-on.** SafetyCore (immutable), SecurityHead,
   SelfReflection und Value Learning müssen von Anfang an in der Architektur
   verwurzelt sein. Ein Modell, das "gut sein will" (GoalEncoder + Compound Reward),
   ist sicherer als eines, das nur Regeln befolgt.

5. **Der kritische Pfad ist: Lernen → Verstehen → Verbessern → Expandieren.**
   Erst wenn das Modell seinen eigenen Code versteht und verbessern kann
   (Phase 58-60), beginnt das exponentielle Wachstum.

---

*"Der Weg zur AGI ist kein Sprint, kein Marathon — es ist eine Expedition
ins Unbekannte. Jede Phase bringt uns einen Schritt näher."*
