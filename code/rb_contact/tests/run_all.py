"""
Test suite runner for RB-Contact% (spec deliverable: "Test suite ... runnable
via CI/standard test runner"). Uses stdlib unittest discovery rather than
pytest, since pytest isn't installed in the `baseball` conda env and adding
it wasn't confirmed with the user -- unittest requires no new dependency.

Usage:  python code/rb_contact/tests/run_all.py
"""
import os, sys, unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.dirname(THIS_DIR))  # code/rb_contact, for module imports

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover(THIS_DIR, pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
