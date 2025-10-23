import libcst as cst
from libcst.metadata import ParentNodeProvider
from typing import Dict


class LibCSTRenameTransformer(cst.CSTTransformer):
    """Transformer to rename function parameters and local variables.

    This version is intentionally small and self-contained so it can be
    unit-tested independently of I/O and preprocessing.
    """
    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(self):
        super().__init__()
        self.rename_map_stack: list[Dict[str, str]] = []
        self.counters_stack: list[Dict[str, int]] = []
        self.collected: Dict[str, str] = {}

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        new_rename_map: Dict[str, str] = {}
        new_counters = {"param": 0, "var": 0}

        def _maybe_add_param(p):
            if p is None:
                return
            try:
                name_str = p.name.value
            except Exception:
                return
            new_name = f"param_{new_counters['param']}"
            new_rename_map[name_str] = new_name
            self.collected[name_str] = new_name
            new_counters["param"] += 1

        for param in node.params.params:
            _maybe_add_param(param)
        for param in getattr(node.params, "posonly_params", []):
            _maybe_add_param(param)
        for param in getattr(node.params, "kwonly_params", []):
            _maybe_add_param(param)
        _maybe_add_param(getattr(node.params, "star_arg", None))
        _maybe_add_param(getattr(node.params, "star_kwarg", None))

        self.rename_map_stack.append(new_rename_map)
        self.counters_stack.append(new_counters)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if self.rename_map_stack:
            self.rename_map_stack.pop()
        if self.counters_stack:
            self.counters_stack.pop()
        return updated_node

    def _collect_target_names(self, target_expr):
        names: list[str] = []
        try:
            if isinstance(target_expr, cst.Name):
                names.append(target_expr.value)
            elif isinstance(target_expr, (cst.Tuple, cst.List)):
                for elt in target_expr.elements:
                    inner = getattr(elt, "value", None) or getattr(elt, "target", None)
                    if inner is not None:
                        names.extend(self._collect_target_names(inner))
            elif isinstance(target_expr, cst.StarredElement):
                inner = getattr(target_expr, "value", None)
                if inner is not None:
                    names.extend(self._collect_target_names(inner))
        except Exception:
            pass
        return names

    def _maybe_map_var(self, names: list[str]):
        if not self.rename_map_stack:
            return
        current_rename_map = self.rename_map_stack[-1]
        current_counters = self.counters_stack[-1]
        for nm in names:
            if nm not in current_rename_map:
                new_name = f"var_{current_counters['var']}"
                current_rename_map[nm] = new_name
                self.collected[nm] = new_name
                current_counters['var'] += 1

    def visit_Assign(self, node: cst.Assign) -> bool:
        if not self.rename_map_stack:
            return True
        names = self._collect_target_names(node.targets[0].target) if node.targets else []
        if len(node.targets) > 1:
            for t in node.targets:
                names.extend(self._collect_target_names(t.target))
        self._maybe_map_var(names)
        return True

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        if not self.rename_map_stack:
            return True
        self._maybe_map_var(self._collect_target_names(node.target))
        return True

    def visit_For(self, node: cst.For) -> bool:
        if not self.rename_map_stack:
            return True
        self._maybe_map_var(self._collect_target_names(node.target))
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
        if not self.rename_map_stack:
            return
        if generators is None:
            return
        gens = generators if isinstance(generators, (list, tuple)) else (generators,)
        for gen in gens:
            target = getattr(gen, 'target', None) or getattr(gen, 'comp_for', None)
            if target is not None:
                self._maybe_map_var(self._collect_target_names(target))

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
        if not self.rename_map_stack:
            return True
        self._maybe_map_var(self._collect_target_names(node.target))
        return True

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
        name_str = original_node.value
        try:
            parent = self.get_metadata(ParentNodeProvider, original_node)
        except Exception:
            parent = None
        if isinstance(parent, cst.Attribute) and parent.attr is original_node:
            return updated_node
        if isinstance(parent, (cst.Param, cst.FunctionDef, cst.ClassDef, cst.ImportAlias, cst.Attribute, cst.ParamStar)):
            return updated_node
        for rename_map in reversed(self.rename_map_stack):
            if name_str in rename_map:
                return updated_node.with_changes(value=rename_map[name_str])
        return updated_node

    def leave_Param(self, original_node: cst.Param, updated_node: cst.Param) -> cst.Param:
        if not self.rename_map_stack:
            return updated_node
        current_rename_map = self.rename_map_stack[-1]
        orig = original_node.name.value
        if orig in current_rename_map:
            new_name_node = updated_node.name.with_changes(value=current_rename_map[orig])
            return updated_node.with_changes(name=new_name_node)
        return updated_node
