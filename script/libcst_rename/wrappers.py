import textwrap
import libcst as cst
from libcst.metadata import ParentNodeProvider
from .transformer import LibCSTRenameTransformer
from typing import Tuple, Optional, Dict


def attempt_wrappers(preprocessed: str) -> Tuple[str, Optional[str], Dict[str, str], Optional[str]]:
    """Try a series of wrapper strategies.

    Returns (status, new_code or None, mapping, error_message or None)
    status is one of: 'wrapped', 'if_wrapped', 'try_wrapped', or 'failed'
    """
    # attempt 1: def wrapper
    try:
        wrapped = 'def __b4_wrap__():\n' + textwrap.indent(preprocessed, '    ')
        tree = cst.parse_module(wrapped)
        wrapper = cst.metadata.MetadataWrapper(tree, cache={ParentNodeProvider})
        transformer = LibCSTRenameTransformer()
        modified_tree = wrapper.visit(transformer)
        func_node = None
        for node in modified_tree.body:
            if isinstance(node, cst.FunctionDef) and node.name.value == '__b4_wrap__':
                func_node = node
                break
        if func_node is not None:
            new_module = cst.Module(body=func_node.body.body)
            new_code = new_module.code
            cst.parse_module(new_code)
            return 'wrapped', new_code, transformer.collected, None
    except Exception as e:
        err1 = str(e)

    # attempt 2: if True wrapper
    try:
        wrapped2 = 'if True:\n' + textwrap.indent(preprocessed, '    ')
        tree2 = cst.parse_module(wrapped2)
        wrapper2 = cst.metadata.MetadataWrapper(tree2, cache={ParentNodeProvider})
        transformer = LibCSTRenameTransformer()
        modified_tree2 = wrapper2.visit(transformer)
        if_node = None
        for n in modified_tree2.body:
            if isinstance(n, cst.If):
                if_node = n
                break
        if if_node is not None:
            new_module = cst.Module(body=if_node.body.body)
            new_code = new_module.code
            cst.parse_module(new_code)
            return 'if_wrapped', new_code, transformer.collected, None
    except Exception as e:
        err2 = str(e)

    # attempt 3: try/except wrapper
    try:
        wrapped3 = 'try:\n' + textwrap.indent(preprocessed, '    ') + '\nexcept Exception:\n    pass'
        tree3 = cst.parse_module(wrapped3)
        wrapper3 = cst.metadata.MetadataWrapper(tree3, cache={ParentNodeProvider})
        transformer = LibCSTRenameTransformer()
        modified_tree3 = wrapper3.visit(transformer)
        try_node = None
        for n3 in modified_tree3.body:
            if isinstance(n3, cst.Try):
                try_node = n3
                break
        if try_node is not None:
            new_module = cst.Module(body=try_node.body.body)
            new_code = new_module.code
            cst.parse_module(new_code)
            return 'try_wrapped', new_code, transformer.collected, None
    except Exception as e:
        err3 = str(e)

    # If none succeeded, return failed with a combined error
    combined = '\n'.join([s for s in (locals().get('err1'), locals().get('err2'), locals().get('err3')) if s])
    return 'failed', None, {}, combined or 'unknown'
