"""
Python AST Intelligence Module.

Extracts language-level structural entities (functions, classes, methods,
nested functions, parameters, return types, decorators, docstrings,
constants, and imports) using Python's built-in ast module and computes
fine-grained semantic deltas between file versions.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParameterInfo:
    """Represents a single function or method parameter."""

    name: str
    type_annotation: str | None = None
    default_value: str | None = None
    kind: str = "POSITIONAL_OR_KEYWORD"  # POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, VAR_POSITIONAL, KEYWORD_ONLY, VAR_KEYWORD


@dataclass(frozen=True)
class SymbolInfo:
    """Represents a code symbol (function, method, nested function, or class) in Python AST."""

    name: str  # Fully qualified name, e.g. "func_name" or "Outer.Inner.method"
    kind: str  # "function", "async_function", "method", "async_method", "nested_function", "async_nested_function", "class"
    line_number: int
    end_line_number: int | None = None
    docstring: str | None = None
    is_public: bool = True
    parameters: list[ParameterInfo] = field(default_factory=list)
    return_type: str | None = None
    decorators: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)  # For classes
    body_hash: str | None = None


@dataclass(frozen=True)
class ConstantInfo:
    """Represents a module-level or class-level constant or variable assignment."""

    name: str  # e.g. "DEFAULT_PORT" or "Service.TIMEOUT"
    value: str | None = None
    type_annotation: str | None = None
    line_number: int = 1
    is_public: bool = True


@dataclass(frozen=True)
class ImportInfo:
    """Represents a module-level import or from-import statement."""

    module: str | None = (
        None  # e.g. "service" for "from service import User" or None for "import os"
    )
    name: str = ""  # e.g. "User" or "os"
    alias: str | None = None  # e.g. "u" for "import ... as u"
    is_from_import: bool = False
    line_number: int = 1

    @property
    def display_name(self) -> str:
        """Formatted representation for display and logging."""
        if self.is_from_import:
            mod = f"from {self.module} " if self.module else ""
            res = f"{mod}import {self.name}"
        else:
            res = f"import {self.name}"
        if self.alias:
            res += f" as {self.alias}"
        return res


@dataclass
class ASTDelta:
    """Captures structural and semantic differences between two versions of a Python file."""

    file_path: str = ""
    has_semantic_changes: bool = False
    has_signature_changes: bool = False
    has_parameter_changes: bool = False
    has_return_type_changes: bool = False
    has_decorator_changes: bool = False
    has_docstring_changes: bool = False
    has_inheritance_changes: bool = False
    has_constant_changes: bool = False
    has_import_changes: bool = False
    has_module_level_stmt_changes: bool = False
    has_declaration_changes: bool = False
    has_implementation_changes: bool = False
    has_public_interface_changes: bool = False
    syntax_error: str | None = None

    added_symbols: list[SymbolInfo] = field(default_factory=list)
    removed_symbols: list[SymbolInfo] = field(default_factory=list)
    modified_symbols: list[tuple[SymbolInfo, SymbolInfo]] = field(default_factory=list)

    added_parameters: list[tuple[str, ParameterInfo]] = field(default_factory=list)
    removed_parameters: list[tuple[str, ParameterInfo]] = field(default_factory=list)
    modified_parameters: list[tuple[str, ParameterInfo, ParameterInfo]] = field(
        default_factory=list
    )

    modified_return_types: list[tuple[str, str | None, str | None]] = field(default_factory=list)
    modified_docstrings: list[tuple[str, str | None, str | None]] = field(default_factory=list)
    modified_decorators: list[tuple[str, list[str], list[str]]] = field(default_factory=list)
    modified_bases: list[tuple[str, list[str], list[str]]] = field(default_factory=list)

    added_constants: list[ConstantInfo] = field(default_factory=list)
    removed_constants: list[ConstantInfo] = field(default_factory=list)
    modified_constants: list[tuple[ConstantInfo, ConstantInfo]] = field(default_factory=list)

    added_imports: list[ImportInfo] = field(default_factory=list)
    removed_imports: list[ImportInfo] = field(default_factory=list)
    modified_imports: list[tuple[ImportInfo, ImportInfo]] = field(default_factory=list)

    implementation_only_changes: list[str] = field(default_factory=list)


def _is_name_public(dotted_name: str) -> bool:
    """
    Check if a dotted symbol name is public.

    A symbol is public if no component in its dotted path has a leading underscore,
    except for standard Python dunder methods (e.g. __init__, __enter__).
    """
    for part in dotted_name.split("."):
        if part.startswith("__") and part.endswith("__"):
            continue
        if part.startswith("_"):
            return False
    return True


def _compute_function_body_hash(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """
    Compute a deterministic SHA-256 hash of function body statements,
    excluding docstrings, comments, and line numbers.
    """
    stmts = node.body
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        stmts = stmts[1:]

    normalized = ast.Module(body=stmts, type_ignores=[])
    ast_dump = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(ast_dump.encode("utf-8")).hexdigest()


def _compute_class_body_hash(class_node: ast.ClassDef) -> str:
    """
    Compute a deterministic SHA-256 hash of class-level statements,
    excluding method and nested class implementations.

    This ensures method implementation edits do not trigger duplicate
    class-level implementation changes.
    """
    stmts = class_node.body
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        stmts = stmts[1:]

    # Filter out function and class definitions so method bodies don't pollute class hash
    class_level_stmts = [
        s for s in stmts if not isinstance(s, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]

    normalized = ast.Module(body=class_level_stmts, type_ignores=[])
    ast_dump = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(ast_dump.encode("utf-8")).hexdigest()


def _compute_module_statements_hash(tree: ast.Module) -> str:
    """
    Compute a deterministic SHA-256 hash of module-level executable statements,
    excluding functions, classes, imports, assignments/constants, and docstrings.
    """
    stmts = [
        s
        for s in tree.body
        if not isinstance(
            s,
            ast.FunctionDef
            | ast.AsyncFunctionDef
            | ast.ClassDef
            | ast.Import
            | ast.ImportFrom
            | ast.Assign
            | ast.AnnAssign,
        )
    ]
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        stmts = stmts[1:]

    normalized = ast.Module(body=stmts, type_ignores=[])
    ast_dump = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(ast_dump.encode("utf-8")).hexdigest()


def _extract_parameters(args_node: ast.arguments) -> list[ParameterInfo]:
    """Extract ordered parameter metadata from an ast.arguments node."""
    params: list[ParameterInfo] = []

    posonly = list(getattr(args_node, "posonlyargs", []))
    standard_args = list(args_node.args)
    all_positional = posonly + standard_args

    defaults = list(args_node.defaults)
    num_positional = len(all_positional)
    num_defaults = len(defaults)
    first_default_idx = num_positional - num_defaults

    for idx, arg in enumerate(all_positional):
        is_posonly = idx < len(posonly)
        kind = "POSITIONAL_ONLY" if is_posonly else "POSITIONAL_OR_KEYWORD"
        type_ann = ast.unparse(arg.annotation) if arg.annotation else None
        default_val = None
        if idx >= first_default_idx:
            default_expr = defaults[idx - first_default_idx]
            default_val = ast.unparse(default_expr)

        params.append(
            ParameterInfo(
                name=arg.arg,
                type_annotation=type_ann,
                default_value=default_val,
                kind=kind,
            )
        )

    # Vararg (*args)
    if args_node.vararg:
        params.append(
            ParameterInfo(
                name="*" + args_node.vararg.arg,
                type_annotation=ast.unparse(args_node.vararg.annotation)
                if args_node.vararg.annotation
                else None,
                default_value=None,
                kind="VAR_POSITIONAL",
            )
        )

    # Keyword-only args
    for arg, default in zip(args_node.kwonlyargs, args_node.kw_defaults, strict=True):
        type_ann = ast.unparse(arg.annotation) if arg.annotation else None
        default_val = ast.unparse(default) if default is not None else None
        params.append(
            ParameterInfo(
                name=arg.arg,
                type_annotation=type_ann,
                default_value=default_val,
                kind="KEYWORD_ONLY",
            )
        )

    # Kwarg (**kwargs)
    if args_node.kwarg:
        params.append(
            ParameterInfo(
                name="**" + args_node.kwarg.arg,
                type_annotation=ast.unparse(args_node.kwarg.annotation)
                if args_node.kwarg.annotation
                else None,
                default_value=None,
                kind="VAR_KEYWORD",
            )
        )

    return params


class _SymbolExtractor(ast.NodeVisitor):
    """AST visitor extracting classes, functions, methods, constants, and imports."""

    def __init__(self) -> None:
        self.scope_stack: list[tuple[str, str]] = []  # List of (name, kind)
        self.symbols: dict[str, SymbolInfo] = {}
        self.constants: dict[str, ConstantInfo] = {}
        self.imports: list[ImportInfo] = []

    def _current_prefix(self) -> str:
        if not self.scope_stack:
            return ""
        return ".".join(name for name, _ in self.scope_stack) + "."

    def visit_Import(self, node: ast.Import) -> Any:
        if not self.scope_stack:
            for alias in node.names:
                self.imports.append(
                    ImportInfo(
                        module=None,
                        name=alias.name,
                        alias=alias.asname,
                        is_from_import=False,
                        line_number=node.lineno,
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        if not self.scope_stack:
            mod_prefix = ("." * node.level) + (node.module or "")
            for alias in node.names:
                self.imports.append(
                    ImportInfo(
                        module=mod_prefix,
                        name=alias.name,
                        alias=alias.asname,
                        is_from_import=True,
                        line_number=node.lineno,
                    )
                )
        self.generic_visit(node)

    def _record_constant(
        self,
        name: str,
        val_node: ast.expr | None,
        ann_node: ast.expr | None,
        lineno: int,
    ) -> None:
        full_name = self._current_prefix() + name
        val_str = ast.unparse(val_node) if val_node is not None else None
        ann_str = ast.unparse(ann_node) if ann_node is not None else None
        is_pub = _is_name_public(full_name)
        self.constants[full_name] = ConstantInfo(
            name=full_name,
            value=val_str,
            type_annotation=ann_str,
            line_number=lineno,
            is_public=is_pub,
        )

    def visit_Assign(self, node: ast.Assign) -> Any:
        # Only capture module-level or class-level assignments; ignore local variables inside functions
        is_module_level = len(self.scope_stack) == 0
        is_class_level = len(self.scope_stack) > 0 and self.scope_stack[-1][1] == "class"

        if is_module_level or is_class_level:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._record_constant(target.id, node.value, None, node.lineno)
                elif isinstance(target, ast.Tuple | ast.List):
                    for idx, elt in enumerate(target.elts):
                        if isinstance(elt, ast.Name):
                            val_node = None
                            if isinstance(node.value, ast.Tuple | ast.List) and len(
                                node.value.elts
                            ) == len(target.elts):
                                val_node = node.value.elts[idx]
                            else:
                                val_node = node.value
                            self._record_constant(elt.id, val_node, None, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        is_module_level = len(self.scope_stack) == 0
        is_class_level = len(self.scope_stack) > 0 and self.scope_stack[-1][1] == "class"

        if is_module_level or is_class_level:
            if isinstance(node.target, ast.Name):
                self._record_constant(node.target.id, node.value, node.annotation, node.lineno)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        full_name = self._current_prefix() + node.name
        docstring = ast.get_docstring(node)
        bases = [ast.unparse(b) for b in node.bases]
        decorators = [f"@{ast.unparse(d)}" for d in node.decorator_list]
        body_hash = _compute_class_body_hash(node)

        symbol = SymbolInfo(
            name=full_name,
            kind="class",
            line_number=node.lineno,
            end_line_number=getattr(node, "end_lineno", None),
            docstring=docstring,
            is_public=_is_name_public(full_name),
            parameters=[],
            return_type=None,
            decorators=decorators,
            bases=bases,
            body_hash=body_hash,
        )
        self.symbols[full_name] = symbol

        self.scope_stack.append((node.name, "class"))
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._record_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._record_function(node, is_async=True)

    def _record_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool
    ) -> None:
        full_name = self._current_prefix() + node.name
        docstring = ast.get_docstring(node)
        decorators = [f"@{ast.unparse(d)}" for d in node.decorator_list]
        return_type = ast.unparse(node.returns) if node.returns else None
        parameters = _extract_parameters(node.args)
        body_hash = _compute_function_body_hash(node)

        # Precise classification:
        # - Direct child of class -> method / async_method
        # - Top-level -> function / async_function
        # - Child of function or method -> nested_function / async_nested_function
        if not self.scope_stack:
            kind = "async_function" if is_async else "function"
            is_pub = _is_name_public(full_name)
        elif self.scope_stack[-1][1] == "class":
            kind = "async_method" if is_async else "method"
            is_pub = _is_name_public(full_name)
        else:
            kind = "async_nested_function" if is_async else "nested_function"
            is_pub = False  # Local to the parent function

        symbol = SymbolInfo(
            name=full_name,
            kind=kind,
            line_number=node.lineno,
            end_line_number=getattr(node, "end_lineno", None),
            docstring=docstring,
            is_public=is_pub,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            bases=[],
            body_hash=body_hash,
        )
        self.symbols[full_name] = symbol

        self.scope_stack.append((node.name, kind))
        self.generic_visit(node)
        self.scope_stack.pop()


def extract_python_symbols(source_code: str) -> dict[str, SymbolInfo]:
    """
    Parse Python source code and extract all classes, functions, methods, and nested functions.

    Returns a dictionary mapping fully qualified symbol names to SymbolInfo objects.
    Safely handles empty strings, comments-only files, and syntax errors.
    """
    if not source_code or not source_code.strip():
        return {}

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return {}

    extractor = _SymbolExtractor()
    extractor.visit(tree)
    return dict(sorted(extractor.symbols.items()))


def extract_python_constants(source_code: str) -> dict[str, ConstantInfo]:
    """
    Parse Python source code and extract all module-level and class-level constants/assignments.

    Safely handles empty strings, comments-only files, and syntax errors.
    """
    if not source_code or not source_code.strip():
        return {}

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return {}

    extractor = _SymbolExtractor()
    extractor.visit(tree)
    return dict(sorted(extractor.constants.items()))


def extract_python_imports(source_code: str) -> list[ImportInfo]:
    """
    Parse Python source code and extract all module-level import statements.

    Safely handles empty strings, comments-only files, and syntax errors.
    """
    if not source_code or not source_code.strip():
        return []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    extractor = _SymbolExtractor()
    extractor.visit(tree)
    return sorted(
        extractor.imports,
        key=lambda imp: (imp.is_from_import, imp.module or "", imp.name, imp.alias or ""),
    )


def compute_ast_delta(old_code: str, new_code: str, file_path: str = "") -> ASTDelta:
    """
    Compute fine-grained structural and semantic differences between two versions of a Python file.

    Identifies added/removed symbols, modified parameters, changed return types,
    updated docstrings, modified decorators, class inheritance changes,
    constants/variables changes, import changes, and pure implementation changes.
    """
    delta = ASTDelta(file_path=file_path)

    old_code_clean = old_code or ""
    new_code_clean = new_code or ""

    if old_code_clean == new_code_clean:
        return delta

    # Check for syntax errors
    old_syntax_err = None
    new_syntax_err = None
    old_tree = None
    new_tree = None

    try:
        old_tree = ast.parse(old_code_clean)
    except SyntaxError as e:
        old_syntax_err = f"Old syntax error: {e}"

    try:
        new_tree = ast.parse(new_code_clean)
    except SyntaxError as e:
        new_syntax_err = f"New syntax error: {e}"

    if old_syntax_err or new_syntax_err:
        err_msg = "; ".join(filter(None, [old_syntax_err, new_syntax_err]))
        delta.syntax_error = err_msg
        delta.has_semantic_changes = True
        return delta

    # 1. Compare Functions, Classes, and Methods
    old_symbols = extract_python_symbols(old_code_clean)
    new_symbols = extract_python_symbols(new_code_clean)

    for name, sym in new_symbols.items():
        if name not in old_symbols:
            delta.added_symbols.append(sym)

    for name, sym in old_symbols.items():
        if name not in new_symbols:
            delta.removed_symbols.append(sym)

    for name, new_sym in new_symbols.items():
        if name not in old_symbols:
            continue
        old_sym = old_symbols[name]

        symbol_signature_changed = False

        # Compare parameters
        old_params = {p.name: p for p in old_sym.parameters}
        new_params = {p.name: p for p in new_sym.parameters}

        for p_name, p_new in new_params.items():
            if p_name not in old_params:
                delta.added_parameters.append((name, p_new))
                symbol_signature_changed = True
            else:
                p_old = old_params[p_name]
                if (
                    p_old.type_annotation != p_new.type_annotation
                    or p_old.default_value != p_new.default_value
                    or p_old.kind != p_new.kind
                ):
                    delta.modified_parameters.append((name, p_old, p_new))
                    symbol_signature_changed = True

        for p_name, p_old in old_params.items():
            if p_name not in new_params:
                delta.removed_parameters.append((name, p_old))
                symbol_signature_changed = True

        # Check parameter reordering
        old_names = [p.name for p in old_sym.parameters]
        new_names = [p.name for p in new_sym.parameters]
        if set(old_names) == set(new_names) and old_names != new_names:
            symbol_signature_changed = True
            delta.has_parameter_changes = True

        # Compare return types
        if old_sym.return_type != new_sym.return_type:
            delta.modified_return_types.append((name, old_sym.return_type, new_sym.return_type))
            symbol_signature_changed = True

        # Compare docstrings
        if old_sym.docstring != new_sym.docstring:
            delta.modified_docstrings.append((name, old_sym.docstring, new_sym.docstring))
            symbol_signature_changed = True

        # Compare decorators
        if old_sym.decorators != new_sym.decorators:
            delta.modified_decorators.append((name, old_sym.decorators, new_sym.decorators))
            symbol_signature_changed = True

        # Compare class bases
        if old_sym.bases != new_sym.bases:
            delta.modified_bases.append((name, old_sym.bases, new_sym.bases))
            symbol_signature_changed = True

        if symbol_signature_changed:
            delta.modified_symbols.append((old_sym, new_sym))
        elif old_sym.body_hash != new_sym.body_hash:
            delta.implementation_only_changes.append(name)

    # 2. Compare Constants and Variable Assignments
    old_constants = extract_python_constants(old_code_clean)
    new_constants = extract_python_constants(new_code_clean)

    for name, c_new in new_constants.items():
        if name not in old_constants:
            delta.added_constants.append(c_new)
        else:
            c_old = old_constants[name]
            if c_old.value != c_new.value or c_old.type_annotation != c_new.type_annotation:
                delta.modified_constants.append((c_old, c_new))

    for name, c_old in old_constants.items():
        if name not in new_constants:
            delta.removed_constants.append(c_old)

    # 3. Compare Imports
    old_imports = extract_python_imports(old_code_clean)
    new_imports = extract_python_imports(new_code_clean)

    old_imp_map = {(imp.is_from_import, imp.module, imp.name): imp for imp in old_imports}
    new_imp_map = {(imp.is_from_import, imp.module, imp.name): imp for imp in new_imports}

    for key, imp_new in new_imp_map.items():
        if key not in old_imp_map:
            delta.added_imports.append(imp_new)
        else:
            imp_old = old_imp_map[key]
            if imp_old.alias != imp_new.alias:
                delta.modified_imports.append((imp_old, imp_new))

    for key, imp_old in old_imp_map.items():
        if key not in new_imp_map:
            delta.removed_imports.append(imp_old)

    # 4. Check Module-Level Executable Statements
    old_mod_hash = _compute_module_statements_hash(old_tree)
    new_mod_hash = _compute_module_statements_hash(new_tree)
    if old_mod_hash != new_mod_hash:
        delta.has_module_level_stmt_changes = True
        delta.implementation_only_changes.append("__module__")

    # Sort all delta collections deterministically
    delta.added_symbols.sort(key=lambda s: s.name)
    delta.removed_symbols.sort(key=lambda s: s.name)
    delta.modified_symbols.sort(key=lambda pair: pair[0].name)
    delta.added_parameters.sort(key=lambda pair: (pair[0], pair[1].name))
    delta.removed_parameters.sort(key=lambda pair: (pair[0], pair[1].name))
    delta.modified_parameters.sort(key=lambda item: (item[0], item[1].name))
    delta.modified_return_types.sort(key=lambda item: item[0])
    delta.modified_docstrings.sort(key=lambda item: item[0])
    delta.modified_decorators.sort(key=lambda item: item[0])
    delta.modified_bases.sort(key=lambda item: item[0])
    delta.added_constants.sort(key=lambda c: c.name)
    delta.removed_constants.sort(key=lambda c: c.name)
    delta.modified_constants.sort(key=lambda pair: pair[0].name)
    delta.added_imports.sort(
        key=lambda imp: (imp.is_from_import, imp.module or "", imp.name, imp.alias or "")
    )
    delta.removed_imports.sort(
        key=lambda imp: (imp.is_from_import, imp.module or "", imp.name, imp.alias or "")
    )
    delta.modified_imports.sort(
        key=lambda pair: (
            pair[0].is_from_import,
            pair[0].module or "",
            pair[0].name,
            pair[0].alias or "",
        )
    )
    delta.implementation_only_changes.sort()

    # Assign comprehensive and well-separated delta flags
    delta.has_parameter_changes = bool(
        delta.has_parameter_changes
        or delta.added_parameters
        or delta.removed_parameters
        or delta.modified_parameters
    )
    delta.has_return_type_changes = bool(delta.modified_return_types)
    delta.has_decorator_changes = bool(delta.modified_decorators)
    delta.has_docstring_changes = bool(delta.modified_docstrings)
    delta.has_inheritance_changes = bool(delta.modified_bases)
    delta.has_constant_changes = bool(
        delta.added_constants or delta.removed_constants or delta.modified_constants
    )
    delta.has_import_changes = bool(
        delta.added_imports or delta.removed_imports or delta.modified_imports
    )

    delta.has_signature_changes = delta.has_parameter_changes or delta.has_return_type_changes

    delta.has_declaration_changes = (
        delta.has_signature_changes
        or delta.has_decorator_changes
        or delta.has_docstring_changes
        or delta.has_inheritance_changes
        or delta.has_constant_changes
        or delta.has_import_changes
        or bool(delta.added_symbols or delta.removed_symbols or delta.modified_symbols)
    )

    delta.has_implementation_changes = bool(delta.implementation_only_changes)

    delta.has_semantic_changes = (
        delta.has_declaration_changes
        or delta.has_implementation_changes
        or delta.has_module_level_stmt_changes
        or bool(delta.syntax_error)
    )

    delta.has_public_interface_changes = (
        any(s.is_public for s in delta.added_symbols)
        or any(s.is_public for s in delta.removed_symbols)
        or any(new_s.is_public for _, new_s in delta.modified_symbols)
        or any(c.is_public for c in delta.added_constants)
        or any(c.is_public for c in delta.removed_constants)
        or any(new_c.is_public for _, new_c in delta.modified_constants)
    )

    return delta


def is_semantic_change(delta: ASTDelta) -> bool:
    """
    Determine if an ASTDelta represents a semantic change.

    Returns True if:
    - Any symbol was added or removed
    - Any signature changed (parameters, return types)
    - Any declaration changed (decorators, bases, docstrings, constants, imports)
    - Any implementation body changed
    - Any module-level executable statements changed
    - A syntax error was detected in changed code

    Returns False if changes are purely non-semantic (e.g., comments, whitespace, formatting).
    """
    return delta.has_semantic_changes
