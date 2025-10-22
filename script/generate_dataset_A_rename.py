import libcst as cst
from libcst.metadata import (
    MetadataWrapper,
    ParentNodeProvider,
)
import json
import random # Needed if we add changes percentage % logic later
import textwrap

class LibCSTRenameTransformer(cst.CSTTransformer):
    """
    LibCST transformer to rename function parameters and local variables.
    It is scope-aware and preserves all formatting and comments.
    """
    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(self):
        super().__init__()
        # Stack for rename maps: each entry maps original identifier -> new name
        # e.g. {'x': 'var_0', 'y': 'param_0'} with one map per function scope
        self.rename_map_stack = []
        # Stack for counters: {'param': int, 'var': int}
        self.counters_stack = []
        # Collected mapping for the current snippet: original identifier -> new name
        self.collected = {}

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        # --- ENTERING A NEW SCOPE ---
        # push a fresh scope map and counters for this function
        new_rename_map = {}
        new_counters = {'param': 0, 'var': 0}

        # Map parameters by their identifier string (include posonly/kwonly/star args)
        def _maybe_add_param(p):
            if p is None:
                return
            try:
                name_str = p.name.value
            except Exception:
                return
            new_name = f"param_{new_counters['param']}"
            new_rename_map[name_str] = new_name
            # record in collected mapping
            self.collected[name_str] = new_name
            new_counters['param'] += 1

        # positional-or-keyword
        for param in node.params.params:
            _maybe_add_param(param)
        # positional-only (PEP 570)
        for param in getattr(node.params, 'posonly_params', []):
            _maybe_add_param(param)
        # keyword-only
        for param in getattr(node.params, 'kwonly_params', []):
            _maybe_add_param(param)
        # star args / kwargs
        _maybe_add_param(getattr(node.params, 'star_arg', None))
        _maybe_add_param(getattr(node.params, 'star_kwarg', None))

        self.rename_map_stack.append(new_rename_map)
        self.counters_stack.append(new_counters)
        return True # Continue visiting children

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        # --- EXITING A SCOPE ---
        if self.rename_map_stack:
            self.rename_map_stack.pop()
        if self.counters_stack:
            self.counters_stack.pop()
        return updated_node

    def visit_Assign(self, node: cst.Assign) -> bool:
        # 2. Map all new local variable assignments (e.g., var_0 = ...)
        if not self.rename_map_stack:
            return True

        current_rename_map = self.rename_map_stack[-1]
        current_counters = self.counters_stack[-1]

        # Reuse a class-level helper to collect target names
        names = self._collect_target_names(node.targets[0].target) if node.targets else []
        # For multiple assignment targets (unlikely for simple code), collect each
        if len(node.targets) > 1:
            for t in node.targets:
                names.extend(self._collect_target_names(t.target))

        for nm in names:
            if nm not in current_rename_map:
                new_name = f"var_{current_counters['var']}"
                current_rename_map[nm] = new_name
                # record in collected mapping
                self.collected[nm] = new_name
                current_counters['var'] += 1

        return True

    def _collect_target_names(self, target_expr):
        """Recursively collect simple Name identifiers from a target expression.
        Skip attributes (obj.attr) because attr names are not local variable names.
        Return a list of identifier strings.
        """
        names = []
        try:
            if isinstance(target_expr, cst.Name):
                names.append(target_expr.value)
            elif isinstance(target_expr, (cst.Tuple, cst.List)):
                # elements are Element nodes with .value
                for elt in target_expr.elements:
                    inner = getattr(elt, 'value', None) or getattr(elt, 'target', None)
                    if inner is not None:
                        names.extend(self._collect_target_names(inner))
            elif isinstance(target_expr, cst.StarredElement):
                inner = getattr(target_expr, 'value', None)
                if inner is not None:
                    names.extend(self._collect_target_names(inner))
            # ignore Attribute, Subscript, Call, etc. (not local name definitions)
        except Exception:
            pass
        return names

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        # annotated assignment: target: Type = value
        if not self.rename_map_stack:
            return True
        current_rename_map = self.rename_map_stack[-1]
        current_counters = self.counters_stack[-1]
        names = self._collect_target_names(node.target)
        for nm in names:
            if nm not in current_rename_map:
                new_name = f"var_{current_counters['var']}"
                current_rename_map[nm] = new_name
                self.collected[nm] = new_name
                current_counters['var'] += 1
        return True

    def visit_For(self, node: cst.For) -> bool:
        if not self.rename_map_stack:
            return True
        current_rename_map = self.rename_map_stack[-1]
        current_counters = self.counters_stack[-1]
        names = self._collect_target_names(node.target)
        for nm in names:
            if nm not in current_rename_map:
                new_name = f"var_{current_counters['var']}"
                current_rename_map[nm] = new_name
                self.collected[nm] = new_name
                current_counters['var'] += 1
        return True

    def visit_AsyncFor(self, node) -> bool:
        return self.visit_For(node)

    def visit_With(self, node: cst.With) -> bool:
        if not self.rename_map_stack:
            return True
        current_rename_map = self.rename_map_stack[-1]
        current_counters = self.counters_stack[-1]
        for item in node.items:
            asname = getattr(item, 'asname', None)
            if asname is not None:
                # asname is an AsName with .name
                name_node = getattr(asname, 'name', None)
                if isinstance(name_node, cst.Name):
                    nm = name_node.value
                    if nm not in current_rename_map:
                        new_name = f"var_{current_counters['var']}"
                        current_rename_map[nm] = new_name
                        self.collected[nm] = new_name
                        current_counters['var'] += 1
        return True

    def visit_ExceptHandler(self, node: cst.ExceptHandler) -> bool:
        if not self.rename_map_stack:
            return True
        name_node = getattr(node, 'name', None)
        if isinstance(name_node, cst.Name):
            current_rename_map = self.rename_map_stack[-1]
            current_counters = self.counters_stack[-1]
            nm = name_node.value
            if nm not in current_rename_map:
                new_name = f"var_{current_counters['var']}"
                current_rename_map[nm] = new_name
                self.collected[nm] = new_name
                current_counters['var'] += 1
        return True

    def _map_names_from_comprehensions(self, generators):
        # generators: sequence of Comprehension objects
        if not self.rename_map_stack:
            return
        current_rename_map = self.rename_map_stack[-1]
        current_counters = self.counters_stack[-1]
        if generators is None:
            return

        # generators may be a single CompFor/Comprehension object or an iterable
        if isinstance(generators, (list, tuple)):
            gens = generators
        else:
            gens = (generators,)

        for gen in gens:
            target = getattr(gen, 'target', None) or getattr(gen, 'comp_for', None)
            if target is not None:
                names = self._collect_target_names(target)
                for nm in names:
                    if nm not in current_rename_map:
                        new_name = f"var_{current_counters['var']}"
                        current_rename_map[nm] = new_name
                        self.collected[nm] = new_name
                        current_counters['var'] += 1

    def visit_ListComp(self, node: cst.ListComp) -> bool:
        self._map_names_from_comprehensions(getattr(node, 'generators', getattr(node, 'for_in', None)))
        return True

    def visit_SetComp(self, node: cst.SetComp) -> bool:
        self._map_names_from_comprehensions(getattr(node, 'generators', getattr(node, 'for_in', None)))
        return True

    def visit_DictComp(self, node: cst.DictComp) -> bool:
        self._map_names_from_comprehensions(getattr(node, 'generators', getattr(node, 'for_in', None)))
        return True

    def visit_GeneratorExp(self, node: cst.GeneratorExp) -> bool:
        self._map_names_from_comprehensions(getattr(node, 'generators', getattr(node, 'for_in', None)))
        return True

    def visit_NamedExpr(self, node: cst.NamedExpr) -> bool:
        # walrus operator: (x := expr)
        if not self.rename_map_stack:
            return True
        names = self._collect_target_names(node.target)
        current_rename_map = self.rename_map_stack[-1]
        current_counters = self.counters_stack[-1]
        for nm in names:
            if nm not in current_rename_map:
                new_name = f"var_{current_counters['var']}"
                current_rename_map[nm] = new_name
                self.collected[nm] = new_name
                current_counters['var'] += 1
        return True

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.BaseExpression:
        # 3. Rename all USAGES of the variable (e.g., print(var_0))
        name_str = original_node.value

        # Use ParentNodeProvider to decide whether this Name is a definition target
        try:
            parent = self.get_metadata(ParentNodeProvider, original_node)
        except Exception:
            parent = None

        # Skip renaming attribute accessors (the attribute name itself)
        if isinstance(parent, cst.Attribute) and parent.attr is original_node:
            return updated_node

        # Skip other definition contexts: Param, Function name, Import alias, AssignTarget
        if isinstance(parent, (cst.Param, cst.FunctionDef, cst.ClassDef, cst.ImportAlias, cst.Attribute, cst.ParamStar)):
            # Param handled in leave_Param; others should not be renamed here
            return updated_node

        # Finally, walk the rename maps from inner to outer and rename the first match.
        # Note: we intentionally DO rename in definition positions (Assign targets, For targets, etc.)
        # because we want to change both definitions and usages. We still skip attribute names and
        # other declaration contexts handled elsewhere (Param, FunctionDef name, ImportAlias).
        for rename_map in reversed(self.rename_map_stack):
            if name_str in rename_map:
                return updated_node.with_changes(value=rename_map[name_str])

        return updated_node

    def leave_Param(
        self, original_node: cst.Param, updated_node: cst.Param
    ) -> cst.Param:
        # 4. Rename the parameter in the function DEFINITION line (e.g., def my_func(param_0):)
        if not self.rename_map_stack:
            return updated_node
        current_rename_map = self.rename_map_stack[-1]
        orig = original_node.name.value
        if orig in current_rename_map:
            new_name_node = updated_node.name.with_changes(value=current_rename_map[orig])
            return updated_node.with_changes(name=new_name_node)
        return updated_node

# --- Main processing function and execution block remain the same ---

def create_rename_attack_dataset_libcst(input_file, output_file):
    """
    Reads a .jsonl file, applies the rename transformation to the 'code' field,
    and writes the result to a new .jsonl file, preserving formatting.
    """
    debug_file = output_file.replace('.jsonl', '_debug.jsonl')

    processed_count = 0
    error_count = 0
    wrapped_count = 0
    rescued_count = 0

    # Ensure files are opened with UTF-8 encoding
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         open(debug_file, 'w', encoding='utf-8') as dbgout:

        for line_num, line in enumerate(infile, 1):
            # Load JSON data from the current line
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Warning: invalid JSON at line {line_num}, copying original. {e}")
                outfile.write(line)
                error_count += 1
                dbgout.write(json.dumps({"line": line_num, "status": "bad_json", "error": str(e)}) + "\n")
                continue

            original_code = data.get("code", "")
            index = data.get('index')

            # Reset transformer state for each snippet so mappings don't leak
            transformer = LibCSTRenameTransformer()
            status = 'unprocessed'
            error_msg = None
            mapping = {}

            if not original_code:
                # nothing to do
                outfile.write(json.dumps(data) + '\n')
                processed_count += 1
                dbgout.write(json.dumps({"line": line_num, "index": index, "status": "empty_code"}) + "\n")
                continue

            # Try parsing and transforming as-is
            try:
                tree = cst.parse_module(original_code)
                wrapper = cst.metadata.MetadataWrapper(tree, cache={ParentNodeProvider})
                modified_tree = wrapper.visit(transformer)
                new_code = modified_tree.code
                status = 'ok'
                mapping = transformer.collected
                rescued = True

            except cst.ParserSyntaxError as e:
                # Try wrapper fallback: wrap snippet in a function to fix indent/fragments
                error_msg = str(e)
                wrapped_count += 1
                try:
                    wrapped = 'def __b4_wrap__():\n' + textwrap.indent(original_code, '    ')
                    tree = cst.parse_module(wrapped)
                    wrapper = cst.metadata.MetadataWrapper(tree, cache={ParentNodeProvider})
                    transformer = LibCSTRenameTransformer()
                    modified_tree = wrapper.visit(transformer)

                    # find the wrapper function in modified_tree
                    func_node = None
                    for node in modified_tree.body:
                        if isinstance(node, cst.FunctionDef) and node.name.value == '__b4_wrap__':
                            func_node = node
                            break

                    if func_node is not None:
                        # Build a Module from the wrapped function's body statements
                        new_module = cst.Module(body=func_node.body.body)
                        new_code = new_module.code
                        # final syntax check
                        try:
                            cst.parse_module(new_code)
                            status = 'wrapped'
                            mapping = transformer.collected
                            rescued = True
                            rescued_count += 1
                        except Exception as e2:
                            error_msg = f"unwrap_failed: {e2}"
                            rescued = False
                    else:
                        error_msg = 'wrapper_func_not_found'
                        rescued = False

                except Exception as e2:
                    error_msg = f"wrapper_parse_failed: {e2}"
                    rescued = False

            except Exception as e:
                error_msg = str(e)
                rescued = False

            if 'rescued' in locals() and rescued:
                data['code'] = new_code
                outfile.write(json.dumps(data) + '\n')
                processed_count += 1
                dbgout.write(json.dumps({"line": line_num, "index": index, "status": status, "mapping": mapping}) + "\n")
            else:
                # Couldn't transform (either parser error or other). Copy original.
                outfile.write(line)
                processed_count += 1
                error_count += 1
                dbgout.write(json.dumps({"line": line_num, "index": index, "status": "failed", "error": error_msg}) + "\n")

    # Print summary after processing all lines
    print(f"\nFinished creating attack dataset A at: {output_file}")
    print(f"Processed (total lines): {processed_count}")
    print(f"Rescued (wrapped & transformed): {rescued_count}")
    print(f"Wrapped attempts: {wrapped_count}")
    print(f"Failed/left unchanged: {error_count} (see {debug_file})")

if __name__ == '__main__':
    # Define input and output file paths
    input_dataset_path = './script/python_test.jsonl'
    output_dataset_path = './script/dataset_A_rename.jsonl'
    # Run the main processing function
    create_rename_attack_dataset_libcst(input_dataset_path, output_dataset_path)