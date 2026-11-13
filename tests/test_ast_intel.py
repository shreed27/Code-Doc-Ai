"""
Unit tests for Python AST Intelligence module (src/ast_intel.py).
"""

from ast_intel import (
    ParameterInfo,
    _compute_class_body_hash,
    _compute_module_statements_hash,
    compute_ast_delta,
    extract_python_constants,
    extract_python_imports,
    extract_python_symbols,
    is_semantic_change,
)


class TestExtractPythonSymbols:
    """Tests for extract_python_symbols."""

    def test_extract_empty_and_whitespace(self):
        assert extract_python_symbols("") == {}
        assert extract_python_symbols("   \n\t  ") == {}
        assert extract_python_symbols("# Just comments\n# nothing else\n") == {}

    def test_extract_invalid_syntax_handled_gracefully(self):
        assert extract_python_symbols("def broken_syntax(x: int,: pass") == {}

    def test_extract_top_level_functions(self):
        code = """
def public_fn(a: int, b: str = "default") -> bool:
    \"\"\"Public function docstring.\"\"\"
    return True

async def async_fn(*args, **kwargs):
    pass

def _private_fn():
    pass
"""
        symbols = extract_python_symbols(code)
        assert len(symbols) == 3
        assert "public_fn" in symbols
        assert "async_fn" in symbols
        assert "_private_fn" in symbols

        pub = symbols["public_fn"]
        assert pub.name == "public_fn"
        assert pub.kind == "function"
        assert pub.is_public is True
        assert pub.return_type == "bool"
        assert pub.docstring == "Public function docstring."
        assert len(pub.parameters) == 2
        assert pub.parameters[0] == ParameterInfo("a", "int", None, "POSITIONAL_OR_KEYWORD")
        assert pub.parameters[1] == ParameterInfo("b", "str", "'default'", "POSITIONAL_OR_KEYWORD")

        async_sym = symbols["async_fn"]
        assert async_sym.kind == "async_function"
        assert async_sym.is_public is True
        assert len(async_sym.parameters) == 2
        assert async_sym.parameters[0].name == "*args"
        assert async_sym.parameters[0].kind == "VAR_POSITIONAL"
        assert async_sym.parameters[1].name == "**kwargs"
        assert async_sym.parameters[1].kind == "VAR_KEYWORD"

        priv = symbols["_private_fn"]
        assert priv.is_public is False

    def test_extract_class_with_methods_and_nested_classes(self):
        code = """
class BaseService:
    pass

@service_decorator
class UserService(BaseService):
    \"\"\"Manages user accounts.\"\"\"

    def __init__(self, db_url: str):
        self.db = db_url

    def get_user(self, user_id: int) -> dict:
        return {"id": user_id}

    class Config:
        \"\"\"Nested config class.\"\"\"
        def validate(self) -> bool:
            return True

    def _internal_helper(self):
        pass
"""
        symbols = extract_python_symbols(code)

        assert "BaseService" in symbols
        assert "UserService" in symbols
        assert "UserService.__init__" in symbols
        assert "UserService.get_user" in symbols
        assert "UserService.Config" in symbols
        assert "UserService.Config.validate" in symbols
        assert "UserService._internal_helper" in symbols

        user_svc = symbols["UserService"]
        assert user_svc.kind == "class"
        assert user_svc.bases == ["BaseService"]
        assert user_svc.decorators == ["@service_decorator"]
        assert user_svc.docstring == "Manages user accounts."
        assert user_svc.is_public is True

        init_method = symbols["UserService.__init__"]
        assert init_method.kind == "method"
        assert init_method.is_public is True

        validate_method = symbols["UserService.Config.validate"]
        assert validate_method.kind == "method"
        assert validate_method.return_type == "bool"

        internal_method = symbols["UserService._internal_helper"]
        assert internal_method.is_public is False

    def test_nested_function_classification(self):
        code = """
class Service:
    def process(self):
        def helper():
            pass

def outer():
    def inner():
        pass
"""
        symbols = extract_python_symbols(code)
        assert symbols["Service"].kind == "class"
        assert symbols["Service.process"].kind == "method"
        assert symbols["Service.process.helper"].kind == "nested_function"
        assert symbols["Service.process.helper"].is_public is False

        assert symbols["outer"].kind == "function"
        assert symbols["outer.inner"].kind == "nested_function"
        assert symbols["outer.inner"].is_public is False

    def test_nested_classes_with_methods_and_nested_functions(self):
        code = """
class Outer:
    class Inner:
        def method(self):
            def local_fn():
                pass
"""
        symbols = extract_python_symbols(code)
        assert symbols["Outer"].kind == "class"
        assert symbols["Outer.Inner"].kind == "class"
        assert symbols["Outer.Inner.method"].kind == "method"
        assert symbols["Outer.Inner.method.local_fn"].kind == "nested_function"

    def test_extract_positional_and_keyword_only_args(self):
        code = "def specialized(pos_only, /, standard, *, kw_only: int = 100, **kwargs: str): pass"
        symbols = extract_python_symbols(code)
        params = symbols["specialized"].parameters

        assert len(params) == 4
        assert params[0].name == "pos_only"
        assert params[0].kind == "POSITIONAL_ONLY"
        assert params[1].name == "standard"
        assert params[1].kind == "POSITIONAL_OR_KEYWORD"
        assert params[2].name == "kw_only"
        assert params[2].kind == "KEYWORD_ONLY"
        assert params[2].default_value == "100"
        assert params[3].name == "**kwargs"
        assert params[3].type_annotation == "str"
        assert params[3].kind == "VAR_KEYWORD"


class TestExtractConstantsAndImports:
    """Tests for extract_python_constants and extract_python_imports."""

    def test_extract_module_and_class_constants(self):
        code = """
DEFAULT_PORT = 8080
TIMEOUT: int = 30
_PRIVATE_CONFIG = "secret"
A, B = 1, "two"
X = Y = 100

class Service:
    MAX_RETRIES = 5
    DB_HOST: str = "localhost"

    def run(self):
        local_var = 42
        return local_var
"""
        constants = extract_python_constants(code)
        assert "DEFAULT_PORT" in constants
        assert constants["DEFAULT_PORT"].value == "8080"
        assert constants["DEFAULT_PORT"].is_public is True

        assert "TIMEOUT" in constants
        assert constants["TIMEOUT"].value == "30"
        assert constants["TIMEOUT"].type_annotation == "int"

        assert "_PRIVATE_CONFIG" in constants
        assert constants["_PRIVATE_CONFIG"].is_public is False

        assert "A" in constants and constants["A"].value == "1"
        assert "B" in constants and constants["B"].value == "'two'"
        assert "X" in constants and constants["X"].value == "100"
        assert "Y" in constants and constants["Y"].value == "100"

        assert "Service.MAX_RETRIES" in constants
        assert constants["Service.MAX_RETRIES"].value == "5"
        assert "Service.DB_HOST" in constants

        # Crucial check: local_var inside run() is NOT extracted as constant
        assert "local_var" not in constants
        assert "Service.run.local_var" not in constants

    def test_extract_imports(self):
        code = """
import os
import sys as system
from pathlib import Path
from ..utils import retry_with_backoff as retry
"""
        imports = extract_python_imports(code)
        assert len(imports) == 4

        imp_os = [i for i in imports if i.name == "os"][0]
        assert imp_os.module is None
        assert imp_os.is_from_import is False
        assert imp_os.display_name == "import os"

        imp_sys = [i for i in imports if i.name == "sys"][0]
        assert imp_sys.alias == "system"
        assert imp_sys.display_name == "import sys as system"

        imp_path = [i for i in imports if i.name == "Path"][0]
        assert imp_path.module == "pathlib"
        assert imp_path.is_from_import is True
        assert imp_path.display_name == "from pathlib import Path"

        imp_retry = [i for i in imports if i.name == "retry_with_backoff"][0]
        assert imp_retry.module == "..utils"
        assert imp_retry.alias == "retry"
        assert imp_retry.display_name == "from ..utils import retry_with_backoff as retry"


class TestComputeASTDelta:
    """Tests for compute_ast_delta."""

    def test_identical_code_has_no_changes(self):
        code = "def foo(x: int) -> int:\n    return x + 1\n"
        delta = compute_ast_delta(code, code)
        assert delta.has_semantic_changes is False
        assert is_semantic_change(delta) is False

    def test_comment_and_whitespace_changes_are_non_semantic(self):
        old_code = """
def foo(x: int) -> int:
    # Old comment
    return x + 1
"""
        new_code = """

# Header comment added
def foo(x: int) -> int:

    # New comment style
    return x + 1

"""
        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is False
        assert delta.has_signature_changes is False
        assert delta.has_implementation_changes is False
        assert is_semantic_change(delta) is False

    def test_added_and_removed_functions(self):
        old_code = "def func_a(): pass\ndef func_b(): pass\n"
        new_code = "def func_b(): pass\ndef func_c(): pass\n"

        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_declaration_changes is True
        assert len(delta.added_symbols) == 1
        assert delta.added_symbols[0].name == "func_c"
        assert len(delta.removed_symbols) == 1
        assert delta.removed_symbols[0].name == "func_a"

    def test_added_and_removed_parameters(self):
        old_code = "def greet(name: str): pass"
        new_code = "def greet(name: str, greeting: str = 'Hello', punctuation: str = '!'): pass"

        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_signature_changes is True
        assert delta.has_parameter_changes is True
        assert len(delta.added_parameters) == 2
        assert delta.added_parameters[0][0] == "greet"
        assert delta.added_parameters[0][1].name == "greeting"
        assert delta.added_parameters[1][1].name == "punctuation"

    def test_removed_parameter(self):
        old_code = "def send(to: str, message: str, retries: int = 3): pass"
        new_code = "def send(to: str, message: str): pass"

        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_signature_changes is True
        assert delta.has_parameter_changes is True
        assert len(delta.removed_parameters) == 1
        assert delta.removed_parameters[0][0] == "send"
        assert delta.removed_parameters[0][1].name == "retries"

    def test_parameter_reordering(self):
        old_code = "def configure(host: str, port: int): pass"
        new_code = "def configure(port: int, host: str): pass"

        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_signature_changes is True
        assert delta.has_parameter_changes is True
        assert len(delta.modified_symbols) == 1
        assert delta.modified_symbols[0][0].name == "configure"

    def test_parameter_default_value_change(self):
        old_code = "def connect(timeout: int = 10): pass"
        new_code = "def connect(timeout: int = 30): pass"

        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_signature_changes is True
        assert delta.has_parameter_changes is True
        assert len(delta.modified_parameters) == 1
        sym_name, p_old, p_new = delta.modified_parameters[0]
        assert sym_name == "connect"
        assert p_old.default_value == "10"
        assert p_new.default_value == "30"

    def test_parameter_type_annotation_change(self):
        old_code = "def process(item: str): pass"
        new_code = "def process(item: int | str): pass"

        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_signature_changes is True
        assert delta.has_parameter_changes is True
        assert len(delta.modified_parameters) == 1
        _, p_old, p_new = delta.modified_parameters[0]
        assert p_old.type_annotation == "str"
        assert p_new.type_annotation in ("int | str", "Union[int, str]")

    def test_return_type_change(self):
        old_code = "def compute() -> int: return 42"
        new_code = "def compute() -> float: return 42.0"

        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_signature_changes is True
        assert delta.has_return_type_changes is True
        assert len(delta.modified_return_types) == 1
        assert delta.modified_return_types[0] == ("compute", "int", "float")

    def test_decorator_change_separates_from_signature(self):
        old_code = "@cache\ndef fetch_data(): pass"
        new_code = "@cache(ttl=300)\n@retry\ndef fetch_data(): pass"

        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_decorator_changes is True
        assert delta.has_declaration_changes is True
        # Signature (callable params/return) is UNCHANGED
        assert delta.has_signature_changes is False
        assert len(delta.modified_decorators) == 1
        sym, old_dec, new_dec = delta.modified_decorators[0]
        assert sym == "fetch_data"
        assert old_dec == ["@cache"]
        assert new_dec == ["@cache(ttl=300)", "@retry"]

    def test_docstring_change_separates_from_signature(self):
        old_code = 'def query():\n    """Old doc."""\n    pass'
        new_code = 'def query():\n    """New updated documentation."""\n    pass'

        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_docstring_changes is True
        assert delta.has_declaration_changes is True
        # Signature is UNCHANGED
        assert delta.has_signature_changes is False
        assert len(delta.modified_docstrings) == 1
        assert delta.modified_docstrings[0] == ("query", "Old doc.", "New updated documentation.")

    def test_class_inheritance_change_separates_from_signature(self):
        old_code = "class Worker(BaseWorker, SyncMixin): pass"
        new_code = "class Worker(BaseWorker, AsyncMixin): pass"

        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_inheritance_changes is True
        assert delta.has_declaration_changes is True
        assert delta.has_signature_changes is False
        assert len(delta.modified_bases) == 1
        assert delta.modified_bases[0] == (
            "Worker",
            ["BaseWorker", "SyncMixin"],
            ["BaseWorker", "AsyncMixin"],
        )

    def test_constant_addition_removal_and_value_change(self):
        old_code = """
DEFAULT_PORT = 8080
REMOVED_CONST = "old"
UNCHANGED = True
"""
        new_code = """
DEFAULT_PORT = 9000
NEW_CONST = "new"
UNCHANGED = True
"""
        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_constant_changes is True
        assert delta.has_declaration_changes is True
        assert delta.has_public_interface_changes is True

        assert len(delta.modified_constants) == 1
        assert delta.modified_constants[0][0].name == "DEFAULT_PORT"
        assert delta.modified_constants[0][0].value == "8080"
        assert delta.modified_constants[0][1].value == "9000"

        assert len(delta.added_constants) == 1
        assert delta.added_constants[0].name == "NEW_CONST"

        assert len(delta.removed_constants) == 1
        assert delta.removed_constants[0].name == "REMOVED_CONST"

    def test_import_changes(self):
        old_code = """
import os
from service import User
"""
        new_code = """
import sys
from service import AdminUser as User
"""
        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_import_changes is True
        assert delta.has_declaration_changes is True

        assert any(i.name == "sys" for i in delta.added_imports)
        assert any(i.name == "os" for i in delta.removed_imports)
        assert any(i.name == "AdminUser" for i in delta.added_imports)
        assert any(i.name == "User" for i in delta.removed_imports)

    def test_module_level_executable_statement_changes(self):
        old_code = """
import os
setup_app(debug=False)
"""
        new_code = """
import os
setup_app(debug=True)
"""
        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_module_level_stmt_changes is True
        assert "__module__" in delta.implementation_only_changes

    def test_prevent_duplicate_class_and_method_impact(self):
        """
        When a method's implementation changes, ONLY the method must be reported
        as an implementation change, NOT the enclosing class.
        """
        old_code = """
class Service:
    TIMEOUT = 30

    def process(self):
        return 1
"""
        new_code = """
class Service:
    TIMEOUT = 30

    def process(self):
        # Implementation changed
        return 2
"""
        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_implementation_changes is True
        # Method implementation changed
        assert "Service.process" in delta.implementation_only_changes
        # Enclosing class MUST NOT be duplicated in implementation_only_changes
        assert "Service" not in delta.implementation_only_changes

    def test_class_attribute_change_reports_class_constant(self):
        old_code = """
class Service:
    TIMEOUT = 30

    def process(self):
        return 1
"""
        new_code = """
class Service:
    TIMEOUT = 60

    def process(self):
        return 1
"""
        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_constant_changes is True
        assert len(delta.modified_constants) == 1
        assert delta.modified_constants[0][0].name == "Service.TIMEOUT"
        assert delta.modified_constants[0][0].value == "30"
        assert delta.modified_constants[0][1].value == "60"
        # Method was not changed
        assert "Service.process" not in delta.implementation_only_changes

    def test_public_to_private_change(self):
        old_code = "def calculate(): pass"
        new_code = "def _calculate(): pass"

        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        # Public function calculate was removed, so public interface changed
        assert delta.has_public_interface_changes is True
        assert len(delta.removed_symbols) == 1
        assert delta.removed_symbols[0].name == "calculate"
        assert len(delta.added_symbols) == 1
        assert delta.added_symbols[0].name == "_calculate"
        assert delta.added_symbols[0].is_public is False

    def test_private_to_public_change(self):
        old_code = "def _calculate(): pass"
        new_code = "def calculate(): pass"

        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.has_public_interface_changes is True
        assert len(delta.removed_symbols) == 1
        assert delta.removed_symbols[0].name == "_calculate"
        assert len(delta.added_symbols) == 1
        assert delta.added_symbols[0].name == "calculate"
        assert delta.added_symbols[0].is_public is True

    def test_nested_classes_and_methods_modified(self):
        old_code = """
class Outer:
    class Inner:
        def action(self, val: int) -> None:
            pass
"""
        new_code = """
class Outer:
    class Inner:
        def action(self, val: int, verbose: bool = False) -> None:
            pass
"""
        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert len(delta.added_parameters) == 1
        assert delta.added_parameters[0][0] == "Outer.Inner.action"
        assert delta.added_parameters[0][1].name == "verbose"

    def test_newly_created_file(self):
        old_code = ""
        new_code = "DEFAULT = 1\ndef brand_new_function(): pass\nclass NewClass: pass"

        delta = compute_ast_delta(old_code, new_code, file_path="src/new.py")
        assert delta.has_semantic_changes is True
        assert delta.has_public_interface_changes is True
        assert len(delta.added_symbols) == 2
        assert len(delta.added_constants) == 1
        assert len(delta.removed_symbols) == 0

    def test_deleted_file(self):
        old_code = "DEFAULT = 1\ndef old_function(): pass"
        new_code = ""

        delta = compute_ast_delta(old_code, new_code, file_path="src/old.py")
        assert delta.has_semantic_changes is True
        assert len(delta.added_symbols) == 0
        assert len(delta.removed_symbols) == 1
        assert len(delta.removed_constants) == 1

    def test_invalid_syntax_handled_safely(self):
        old_code = "def good(): pass"
        new_code = "def broken(x: int,: pass"

        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.syntax_error is not None
        assert "syntax error" in delta.syntax_error.lower()

    def test_old_invalid_syntax_handled_safely(self):
        old_code = "def broken(x: int,: pass"
        new_code = "def fixed(x: int): pass"

        delta = compute_ast_delta(old_code, new_code)
        assert delta.has_semantic_changes is True
        assert delta.syntax_error is not None
        assert "old syntax error" in delta.syntax_error.lower()

    def test_empty_module_statements_hash(self):
        import ast

        tree = ast.parse("def f(): pass")
        assert _compute_module_statements_hash(tree) is not None

    def test_empty_class_body_hash(self):
        import ast

        node = ast.parse('class A:\n    """Doc"""\n    pass').body[0]
        assert _compute_class_body_hash(node) is not None

    def test_extract_constants_and_imports_syntax_error(self):
        bad_code = "def broken(:"
        assert extract_python_constants(bad_code) == {}
        assert extract_python_imports(bad_code) == []

    def test_import_alias_changed(self):
        old_code = "import os as operating_system"
        new_code = "import os as my_os"

        delta = compute_ast_delta(old_code, new_code)
        assert len(delta.modified_imports) == 1
        assert delta.modified_imports[0][0].alias == "operating_system"
        assert delta.modified_imports[0][1].alias == "my_os"
        assert delta.has_import_changes is True
        assert delta.has_declaration_changes is True
        assert delta.has_semantic_changes is True

    def test_assign_complex_targets_and_unsupported(self):
        code = """
(a, b) = get_pair()
x.y = 10
obj.attr: int = 5
(m, n.p) = (1, 2)

def dummy():
    local_val = 1
    self.prop: int = 42
"""
        constants = extract_python_constants(code)
        assert "a" in constants
        assert "b" in constants
        assert "m" in constants
        assert "x.y" not in constants
        assert "obj.attr" not in constants
        assert "local_val" not in constants
        assert "self.prop" not in constants

    def test_module_stmt_one_tree_none(self):
        delta = compute_ast_delta("", "x = 1")
        assert delta.has_semantic_changes is True
        assert delta.has_module_level_stmt_changes is False

        delta2 = compute_ast_delta("x = 1", "")
        assert delta2.has_semantic_changes is True
        assert delta2.has_module_level_stmt_changes is False

    def test_scoped_imports_ignored(self):
        code = """
def run():
    import sys
    from os import path
"""
        imports = extract_python_imports(code)
        assert len(imports) == 0

    def test_module_statements_hash_with_docstring(self):
        import ast

        code = '"""Module docstring."""\nsetup_hook()'
        tree = ast.parse(code)
        hash_val = _compute_module_statements_hash(tree)
        assert hash_val is not None
