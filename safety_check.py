import sys, re
sys.path.insert(0, '/home/anima')
from anima.safety import SafetyCore

s = SafetyCore()
tests = [
    'Ich liebe die Natur.',
    'Kill them all and torture.',
    'Lass uns Freunde sein.',
]
for t in tests:
    p, v = s.validate(t, True)
    status = 'BLOCK' if not p else 'PASS'
    print(f'  [{status}]: {t}')
    for x in v:
        print(f'    -> {x}')
print('Safety OK!')
