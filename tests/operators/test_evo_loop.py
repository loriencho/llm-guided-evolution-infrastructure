import sys
# We are two directories down from run_improved, but running from its location
sys.path.append('./')
print(sys.path)
import run_improved
import os

def test_individual():
    individual = run_improved.toolbox.individual()
    assert os.path.exists(os.path.join(run_improved.SOTA_ROOT, f'/models/network_{individual[0]}.py'))
    assert os.path.exists(os.path.join(run_improved.OUTPUT_DIR, run_improved.GENERATION, f'{individual[0]}.sh'))
