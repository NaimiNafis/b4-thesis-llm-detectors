import io
import re
import tokenize
import textwrap
from typing import Optional


def preprocess_code(src: str, enable_lib2to3: bool = False) -> str:
    """Normalize whitespace, optionally apply lib2to3 refactor, and fix tokens.

    Keep this function small and testable.
    """
    s = src.replace('\r\n', '\n').replace('\r', '\n')
    s = s.lstrip('\ufeff')
    s = s.replace('\t', '    ')
    try:
        s = textwrap.dedent(s)
    except Exception:
        pass
    s = s.strip('\n')

    if enable_lib2to3:
        try:
            # Import via importlib and silence type checker for environments
            # where lib2to3 is not present (newer Pythons may omit it).
            import importlib

            _refactor = importlib.import_module('lib2to3.refactor')  # type: ignore[import]

            fixer_names = _refactor.get_fixers_from_package('lib2to3.fixes')
            tool = _refactor.RefactoringTool(fixer_names)
            s = tool.refactor_string(s, name='-')
            s = str(s)
        except Exception:
            # lib2to3 not available or failed — skip this step safely
            pass

    # Fix leading-zero numeric literals (simple heuristic)
    try:
        out_tokens = []
        sio = io.StringIO(s)
        tokgen = tokenize.generate_tokens(sio.readline)
        for toknum, tokval, start, end, line_text in tokgen:
            if toknum == tokenize.NUMBER:
                if re.match(r"^0[0-9]+$", tokval):
                    newval = tokval.lstrip('0')
                    if newval == '':
                        newval = '0'
                    tokval = newval
            out_tokens.append((toknum, tokval))
        s = tokenize.untokenize(out_tokens)
    except Exception:
        pass

    return s


def normalize_and_try(src: str, enable_lib2to3: bool = False) -> str:
    """Wrapper that currently just calls preprocess_code (placeholder for more).
    """
    return preprocess_code(src, enable_lib2to3=enable_lib2to3)
