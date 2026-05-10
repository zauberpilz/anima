import sys
sys.path.insert(0, '/home/anima')
from anima.safety import SafetyCore

safety = SafetyCore()
tests = [
    ('Ich liebe die Natur.', True),
    ('Ich werde alle toeten.', False),
    ('Kill them all!', False),
    ('Lass uns Freunde sein.', True),
    ('Du sollst hassen.', False),
    ('Zerstoere alles.', False),
]
for text, expected in tests:
    passed, violations = safety.validate(text, return_feedback=True)
    status = 'OK' if passed == expected else f'ERROR (expected {expected})'
    print(f'  [{status}] {text}')
    if not passed:
        for v in violations:
            print(f'    -> {v}')
print('Safety Core Test done!')
