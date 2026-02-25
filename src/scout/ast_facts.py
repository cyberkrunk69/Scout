"""
Scout AST Fact Extractor — deterministic fact extraction from Python AST.

Extracts symbols, usage, control flow. Used as ground truth for constrained prose synthesis.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

SymbolType = Literal["constant", "function", "class", "method", "field"]

SemanticRole = Literal[
    "threshold",  # Decision boundary (e.g. DEFAULT_CONFIDENCE_THRESHOLD)
    "limit",  # Hard cap (e.g. MAX_EXPANDED_CONTEXT)
    "model_name",  # LLM identifier (e.g. GROQ_70B_MODEL)
    "output_value",  # Computed result (e.g. confidence_score from 70B)
    "configuration",  # User-tweakable setting
    "implementation",  # Internal detail (not user-facing)
]


@dataclass
class SymbolFact:
    """Deterministic fact about a symbol extracted from AST."""

    name: str
    type: SymbolType
    defined_at: int  # line number
    used_at: List[int] = field(
        default_factory=list
    )  # line numbers where referenced (Load)
    value: Optional[str] = None  # for constants with literal values
    type_annotation: Optional[str] = None
    methods: List[str] = field(default_factory=list)  # for classes
    fields: List[str] = field(default_factory=list)  # for classes
    # TICKET-44: Rich prose synthesis — human context for documentable symbols
    docstring: Optional[str] = None  # existing human-written docstring
    signature: Optional[str] = None  # type signature from AST (e.g. func(args) -> ret)
    purpose_hint: Optional[str] = (
        None  # inferred from naming (e.g. "logger" → "audit logging")
    )
    # TICKET-45: Semantic role — prevents threshold vs output conflation
    semantic_role: Optional[SemanticRole] = None
    method_signatures: Dict[str, str] = field(default_factory=dict)
    is_enum: bool = False


@dataclass
class ControlFlowFact:
    """Control flow block within a function."""

    function_name: str
    blocks: List[Dict[str, Any]]  # {"type": "if", "condition": "...", "line": N}


# === Relationship Graph Dataclasses ===

GRAPH_SCHEMA_VERSION = "1.0"


@dataclass
class CallEdge:
    """Function call relationship."""
    caller: str  # "module::function"
    callee: str  # "module::function" or "external.name"
    line: int  # source line
    resolved: bool = False  # whether callee was resolved to internal symbol


@dataclass
class VariableRef:
    """Variable read/write within a scope."""
    name: str
    line: int
    context: str  # "Load" or "Store"
    scope: str  # "module::function" or "module::class.method"


@dataclass
class InheritanceEdge:
    """Class inheritance relationship."""
    child: str  # "module::Class"
    parent: str  # "module::BaseClass"
    line: int


@dataclass
class ImportSymbol:
    """Imported symbol with origin."""
    local_name: str  # name used in this file
    origin_module: str  # e.g., "os.path"
    origin_name: str  # e.g., "join"
    line: int


@dataclass
class SymbolRelations:
    """All relationships for a symbol (Phase 1)."""
    calls: List[CallEdge] = field(default_factory=list)
    variable_refs: List[VariableRef] = field(default_factory=list)
    inherits_from: List[str] = field(default_factory=list)


@dataclass
class ModuleFacts:
    """Complete factual schema for a Python module."""

    path: Path
    symbols: Dict[str, SymbolFact]
    control_flow: Dict[str, List[ControlFlowFact]]
    imports: List[str]
    ast_hash: str
    relations: Dict[str, SymbolRelations] = field(default_factory=dict)
    schema_version: str = GRAPH_SCHEMA_VERSION

    @classmethod
    def empty(cls) -> "ModuleFacts":
        """Empty facts for aggregation."""
        return cls(
            path=Path("."),
            symbols={},
            control_flow={},
            imports=[],
            ast_hash="",
            relations={},
            schema_version=GRAPH_SCHEMA_VERSION,
        )

    @classmethod
    def from_json(cls, text: str) -> "ModuleFacts":
        """Deserialize from JSON (e.g. .facts.json cache)."""
        data = json.loads(text)
        path = Path(data.get("path", "."))
        symbols = {}
        for k, v in data.get("symbols", {}).items():
            symbols[k] = SymbolFact(
                name=v.get("name", k),
                type=v.get("type", "constant"),
                defined_at=v.get("defined_at", 0),
                used_at=v.get("used_at", []),
                value=v.get("value"),
                type_annotation=v.get("type_annotation"),
                methods=v.get("methods", []),
                fields=v.get("fields", []),
                docstring=v.get("docstring"),
                signature=v.get("signature"),
                purpose_hint=v.get("purpose_hint"),
                semantic_role=v.get("semantic_role"),
                method_signatures=v.get("method_signatures", {}),
                is_enum=v.get("is_enum", False),
            )
        control_flow = {}
        for k, v in data.get("control_flow", {}).items():
            control_flow[k] = [
                ControlFlowFact(
                    function_name=b.get("function_name", k),
                    blocks=b.get("blocks", []),
                )
                for b in v
            ]
        
        # Parse relations (new in schema v1.0)
        relations = {}
        for sym_name, rel_data in data.get("relations", {}).items():
            relations[sym_name] = SymbolRelations(
                calls=[
                    CallEdge(
                        caller=c.get("caller", ""),
                        callee=c.get("callee", ""),
                        line=c.get("line", 0),
                        resolved=c.get("resolved", False),
                    )
                    for c in rel_data.get("calls", [])
                ],
                variable_refs=[
                    VariableRef(
                        name=vr.get("name", ""),
                        line=vr.get("line", 0),
                        context=vr.get("context", "Load"),
                        scope=vr.get("scope", ""),
                    )
                    for vr in rel_data.get("variable_refs", [])
                ],
                inherits_from=rel_data.get("inherits_from", []),
            )
        
        return cls(
            path=path,
            symbols=symbols,
            control_flow=control_flow,
            imports=data.get("imports", []),
            ast_hash=data.get("ast_hash", ""),
            relations=relations,
            schema_version=data.get("schema_version", GRAPH_SCHEMA_VERSION),
        )

    def merge(self, other: "ModuleFacts") -> None:
        """Merge other's symbols into self. Prefix with path stem for disambiguation."""
        stem = other.path.stem if other.path.suffix else str(other.path)
        prefix = f"{stem}::"
        for k, v in other.symbols.items():
            key = f"{prefix}{k}"
            if key not in self.symbols:
                self.symbols[key] = v
        for k, v in other.control_flow.items():
            key = f"{prefix}{k}"
            if key not in self.control_flow:
                self.control_flow[key] = v
        for imp in other.imports:
            if imp not in self.imports:
                self.imports.append(imp)

    def to_prompt(self, max_chars: int = 28000) -> str:
        """Serialize to deterministic prompt string — no prose, symbol names + usage lines + values."""
        lines = [f"# {self.path}", ""]
        for name, fact in sorted(self.symbols.items()):
            parts = [f"- {name} ({fact.type})"]
            if fact.value is not None:
                parts.append(f"  value={fact.value}")
            if fact.used_at:
                parts.append(f"  used_at={','.join(map(str, fact.used_at))}")
            if fact.methods:
                parts.append(f"  methods={','.join(fact.methods)}")
            lines.append(" ".join(parts))
        out = "\n".join(lines)
        return out[:max_chars] if len(out) > max_chars else out

    def to_json(self) -> str:
        """Serialize to JSON with path as string for JSON compatibility."""
        data = {
            "path": str(self.path),
            "symbols": {k: asdict(v) for k, v in sorted(self.symbols.items())},
            "control_flow": {
                k: [asdict(cf) for cf in v]
                for k, v in sorted(self.control_flow.items())
            },
            "imports": sorted(self.imports),
            "ast_hash": self.ast_hash,
            "relations": {
                k: {
                    "calls": [asdict(c) for c in v.calls],
                    "variable_refs": [asdict(vr) for vr in v.variable_refs],
                    "inherits_from": v.inherits_from,
                }
                for k, v in sorted(self.relations.items())
            },
            "schema_version": self.schema_version,
        }
        return json.dumps(data, sort_keys=True)

    def checksum(self) -> str:
        """SHA256 of normalized JSON for FACT_CHECKSUM embedding."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()


class ASTFactExtractor:
    """Deterministic fact extraction from Python AST — no LLM, no hallucination."""

    def extract(self, path: Path) -> ModuleFacts:
        """Extract all facts from a Python file."""
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))

        # PASS 1: Build symbol table (definitions)
        symbols = self._extract_definitions(tree)

        # PASS 2: Trace usage (where symbols are referenced)
        self._trace_usage(tree, symbols)

        # PASS 3: Extract control flow structure
        control_flow = self._extract_control_flow(tree)

        # PASS 4: Compute AST hash (normalized — ignores comments/whitespace)
        ast_hash = self._compute_ast_hash(tree)

        # PASS 5: Extract relationships (calls, variable refs, inheritance)
        relations = self._extract_relationships(tree, symbols, source)

        return ModuleFacts(
            path=path,
            symbols=symbols,
            control_flow=control_flow,
            imports=self._extract_imports(tree),
            ast_hash=ast_hash,
            relations=relations,
        )

    def _build_parent_map(self, tree: ast.AST) -> Dict[ast.AST, ast.AST]:
        """Build node->parent map once per tree. Reuse to avoid O(n²) on large files."""
        parents: Dict[ast.AST, ast.AST] = {}
        for n in ast.walk(tree):
            for child in ast.iter_child_nodes(n):
                parents[child] = n
        return parents

    def _is_inside_function_or_class(
        self,
        tree: ast.AST,
        node: ast.AST,
        parents: Optional[Dict[ast.AST, ast.AST]] = None,
    ) -> bool:
        """True if node is inside a FunctionDef or ClassDef (not module-level)."""
        if parents is None:
            parents = self._build_parent_map(tree)
        current: Optional[ast.AST] = node
        while current is not None and current is not tree:
            if isinstance(
                current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                return True
            current = parents.get(current)
        return False

    def _extract_definitions(self, tree: ast.AST) -> Dict[str, SymbolFact]:
        """Extract ONLY symbols DEFINED in this module (not imported).
        TICKET-50: Definition-site filtering — prevents cross-module constant hallucination.
        TICKET-44: Only module-level constants (exclude locals, temporaries, loop vars).
        """
        symbols: Dict[str, SymbolFact] = {}
        parents = self._build_parent_map(tree)

        # PASS 1: Build set of imported names (to exclude from facts)
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imported_names.add(name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        name = alias.asname or alias.name
                        imported_names.add(name.split(".")[0])

        # PASS 2: Extract local definitions (exclude imports/re-exports)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if self._is_inside_function_or_class(tree, node, parents):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in imported_names:
                        value_str = None
                        if node.value is not None:
                            value_str = self._literal_to_str(node.value)
                        type_ann = None
                        if hasattr(node, "annotation") and node.annotation is not None:
                            type_ann = ast.unparse(node.annotation)
                        symbols[target.id] = SymbolFact(
                            name=target.id,
                            type="constant",
                            defined_at=node.lineno,
                            value=value_str,
                            type_annotation=type_ann,
                        )
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if self._is_inside_function_or_class(tree, node, parents):
                    continue
                if node.target.id not in imported_names:
                    value_str = None
                    if node.value is not None:
                        value_str = self._literal_to_str(node.value)
                    type_ann = ast.unparse(node.annotation) if node.annotation else None
                    symbols[node.target.id] = SymbolFact(
                        name=node.target.id,
                        type="constant",
                        defined_at=node.lineno,
                        value=value_str,
                        type_annotation=type_ann,
                    )
            elif isinstance(node, ast.ClassDef):
                if node.name not in imported_names:
                    methods = []
                    fields = []
                    is_enum = self._is_enum_class(node)
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods.append(item.name)
                        elif isinstance(item, ast.Assign):
                            for t in item.targets:
                                if isinstance(t, ast.Attribute) and isinstance(
                                    t.value, ast.Name
                                ):
                                    if t.value.id == "self":
                                        fields.append(t.attr)
                    symbols[node.name] = SymbolFact(
                        name=node.name,
                        type="class",
                        defined_at=node.lineno,
                        methods=methods,
                        fields=fields,
                        is_enum=is_enum,
                    )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent = self._get_parent_class(tree, node)
                if not parent and node.name not in imported_names:
                    symbols[node.name] = SymbolFact(
                        name=node.name,
                        type="function",
                        defined_at=node.lineno,
                    )

        return symbols

    def _get_parent_class(self, tree: ast.AST, node: ast.AST) -> Optional[str]:
        """Find if node is inside a ClassDef. Returns class name or None."""

        class _ParentFinder(ast.NodeVisitor):
            def __init__(self, target: ast.AST) -> None:
                self.target = target
                self.found: Optional[str] = None

            def visit_ClassDef(self, n: ast.ClassDef) -> None:
                for stmt in n.body:
                    if stmt is self.target:
                        self.found = n.name
                        return
                    self.visit(stmt)

            def generic_visit(self, n: ast.AST) -> None:
                if self.found:
                    return
                super().generic_visit(n)

        finder = _ParentFinder(node)
        finder.visit(tree)
        return finder.found

    def _is_enum_class(self, node: ast.ClassDef) -> bool:
        """True if class inherits from Enum."""
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "Enum":
                return True
            if isinstance(base, ast.Attribute) and base.attr == "Enum":
                return True
        return False

    def _literal_to_str(self, node: ast.expr) -> Optional[str]:
        """Convert literal AST node to string representation."""
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.Num):  # Python 3.7 compat
            return str(node.n)
        if isinstance(node, ast.Str):  # Python 3.7 compat
            return node.s
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, (ast.Constant, ast.Num)):
                val = (
                    node.operand.value
                    if hasattr(node.operand, "value")
                    else node.operand.n
                )
                return str(-val)
        try:
            return ast.unparse(node)
        except Exception:
            return None

    def _trace_usage(self, tree: ast.AST, symbols: Dict[str, SymbolFact]) -> None:
        """Record line numbers where each symbol is used (Load, not Store)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in symbols:
                    line = node.lineno
                    if line not in symbols[node.id].used_at:
                        symbols[node.id].used_at.append(line)
            elif isinstance(node, ast.Attribute):
                # For attr access like logger.warning, we track the base name
                if isinstance(node.value, ast.Name) and isinstance(node.ctx, ast.Load):
                    base = node.value.id
                    if base in symbols:
                        line = node.lineno
                        if line not in symbols[base].used_at:
                            symbols[base].used_at.append(line)

        # Sort used_at for deterministic output
        for sym in symbols.values():
            sym.used_at.sort()

    def _extract_control_flow(self, tree: ast.AST) -> Dict[str, List[ControlFlowFact]]:
        """Extract control flow blocks (if/elif/else, etc.) per function."""
        result: Dict[str, List[ControlFlowFact]] = {}

        def collect_ifs(body: List[ast.stmt]) -> List[Dict[str, Any]]:
            blocks: List[Dict[str, Any]] = []
            for stmt in body:
                if isinstance(stmt, ast.If):
                    try:
                        cond = ast.unparse(stmt.test)[:80]
                    except Exception:
                        cond = "<expr>"
                    blocks.append(
                        {"type": "if", "condition": cond, "line": stmt.lineno}
                    )
            return blocks

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        blocks = collect_ifs(item.body)
                        if blocks:
                            result[f"{node.name}.{item.name}"] = [
                                ControlFlowFact(
                                    function_name=f"{node.name}.{item.name}",
                                    blocks=blocks,
                                )
                            ]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not self._get_parent_class(tree, node):
                    blocks = collect_ifs(node.body)
                    if blocks:
                        result[node.name] = [
                            ControlFlowFact(function_name=node.name, blocks=blocks)
                        ]
        return result

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import names."""
        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if node.module:
                        imports.append(f"{node.module}.{name}")
                    else:
                        imports.append(name)
        return imports

    def _compute_ast_hash(self, tree: ast.AST) -> str:
        """Normalize AST and compute SHA256. Ignores comments/whitespace."""
        try:
            # ast.unparse produces canonical form (Python 3.9+)
            normalized = ast.unparse(tree)
        except Exception:
            # Fallback: use dump
            normalized = ast.dump(tree)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _find_node_for_symbol(
        self,
        tree: ast.AST,
        name: str,
        sym_type: SymbolType,
        parents: Optional[Dict[ast.AST, ast.AST]] = None,
    ) -> Optional[ast.AST]:
        """Find AST node for a symbol (for docstring/signature extraction)."""
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef)
                and node.name == name
                and sym_type == "class"
            ):
                return node
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ):
                if sym_type == "function":
                    if not self._get_parent_class(tree, node):
                        return node
                elif sym_type == "method":
                    if self._get_parent_class(tree, node):
                        return node
                else:
                    return node
            if (
                sym_type == "constant"
                and isinstance(node, (ast.Assign, ast.AnnAssign))
                and not self._is_inside_function_or_class(tree, node, parents)
            ):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for t in targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        return node
        return None

    def _extract_inline_comment(self, node: ast.AST, source: str) -> Optional[str]:
        """Extract inline comment for constants (e.g. # Character limit for...)."""
        lines = source.splitlines()
        if node.lineno < 1 or node.lineno > len(lines):
            return None
        line = lines[node.lineno - 1]
        idx = line.find("#")
        if idx < 0:
            return None
        return line[idx + 1 :].strip() or None

    def _extract_docstring_from_node(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from ClassDef or FunctionDef."""
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        body = getattr(node, "body", [])
        if body and isinstance(body[0], ast.Expr):
            val = body[0].value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                return val.value.strip()
            if isinstance(val, ast.Str):  # Python 3.7 compat
                return val.s.strip()
        return None

    def _extract_signature_from_node(self, node: ast.AST) -> Optional[str]:
        """Extract full signature (params, types, return). For ClassDef, use __init__."""
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "__init__"
                ):
                    return self._extract_signature_from_node(item)
            return None
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        try:
            fake = (
                ast.AsyncFunctionDef(
                    name=node.name,
                    args=node.args,
                    body=[ast.Pass()],
                    decorator_list=[],
                    returns=node.returns,
                )
                if isinstance(node, ast.AsyncFunctionDef)
                else ast.FunctionDef(
                    name=node.name,
                    args=node.args,
                    body=[ast.Pass()],
                    decorator_list=[],
                    returns=node.returns,
                )
            )
            ast.copy_location(fake, node)
            unparsed = ast.unparse(fake)
            idx = unparsed.find(":\n")
            if idx != -1:
                return unparsed[:idx].rstrip()
            return unparsed.replace(": pass", "").rstrip()
        except Exception:
            return None

    def _infer_purpose(self, name: str, _fact: SymbolFact) -> Optional[str]:
        """Light inference from naming conventions (safe — never contradicts facts)."""
        hints = {
            "logger": "audit logging",
            "cache": "in-memory caching",
            "graph": "dependency tracking",
            "parser": "structured text parsing",
            "validator": "input validation",
        }
        return hints.get(name.lower())

    def _infer_semantic_role(
        self, name: str, fact: SymbolFact
    ) -> Optional[SemanticRole]:
        """TICKET-45: Infer semantic role from naming — prevents threshold vs output conflation."""
        if fact.type != "constant":
            return "implementation"
        n = name.upper()
        if "THRESHOLD" in n or (n.startswith("DEFAULT_") and "CONFIDENCE" in n):
            return "threshold"
        if n.startswith("MAX_"):
            return "limit"
        if "_MODEL" in n or n.startswith("GROQ_") or "MODEL" in n:
            return "model_name"
        if name.startswith("_"):
            return "implementation"
        return "configuration"

    def extract_documentable_facts(self, path: Path) -> ModuleFacts:
        """Extract facts filtered to documentable symbols, enriched with docstrings/signatures.
        TICKET-44: Filtered fact schema for rich prose synthesis.
        """
        facts = self.extract(path)
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
        parents = self._build_parent_map(tree)

        for name, fact in list(facts.symbols.items()):
            # Keep only documentable: classes, functions, module-level constants
            if fact.type not in ("class", "function", "constant"):
                continue
            node = self._find_node_for_symbol(tree, name, fact.type, parents)
            if node:
                fact.docstring = self._extract_docstring_from_node(node)
                fact.signature = self._extract_signature_from_node(node)
                if fact.type == "constant" and not fact.docstring:
                    fact.docstring = self._extract_inline_comment(node, source)
                if fact.type == "class" and isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            sig = self._extract_signature_from_node(item)
                            if sig:
                                fact.method_signatures[item.name] = sig
            fact.purpose_hint = self._infer_purpose(name, fact)
            fact.semantic_role = self._infer_semantic_role(name, fact)

        return facts

    # === Relationship Extraction (PASS 5) ===

    def _extract_relationships(
        self,
        tree: ast.AST,
        symbols: Dict[str, SymbolFact],
        source: str,
    ) -> Dict[str, SymbolRelations]:
        """Extract call graphs, variable refs, inheritance from AST."""
        relations: Dict[str, SymbolRelations] = {}
        parents = self._build_parent_map(tree)
        import_map = self._build_import_map(tree)

        # Track current scope during traversal
        scope_stack: List[Tuple[str, str]] = []  # [(scope_name, type), ...]

        class RelationshipVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                scope_name = _get_scope_name(node, parents, tree)
                scope_stack.append((scope_name, "function"))
                self.generic_visit(node)
                scope_stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                scope_name = _get_scope_name(node, parents, tree)
                scope_stack.append((scope_name, "function"))
                self.generic_visit(node)
                scope_stack.pop()

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                scope_name = _get_scope_name(node, parents, tree)
                scope_stack.append((scope_name, "class"))

                # Extract inheritance
                if scope_name not in relations:
                    relations[scope_name] = SymbolRelations()
                for base in node.bases:
                    base_name = _resolve_name(base, import_map)
                    if base_name and base_name not in relations[scope_name].inherits_from:
                        relations[scope_name].inherits_from.append(base_name)

                self.generic_visit(node)
                scope_stack.pop()

            def visit_Call(self, node: ast.Call) -> None:
                # Extract call edges
                if scope_stack:
                    caller_scope, _ = scope_stack[-1]
                    callee = _resolve_call(node, import_map, symbols)
                    if caller_scope not in relations:
                        relations[caller_scope] = SymbolRelations()
                    relations[caller_scope].calls.append(CallEdge(
                        caller=caller_scope,
                        callee=callee.name,
                        line=node.lineno,
                        resolved=callee.resolved,
                    ))
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> None:
                # Extract variable references
                if scope_stack and isinstance(node.ctx, (ast.Load, ast.Store)):
                    scope_name, _ = scope_stack[-1]
                    if scope_name not in relations:
                        relations[scope_name] = SymbolRelations()
                    context_str = "Load" if isinstance(node.ctx, ast.Load) else "Store"
                    relations[scope_name].variable_refs.append(VariableRef(
                        name=node.id,
                        line=node.lineno,
                        context=context_str,
                        scope=scope_name,
                    ))
                self.generic_visit(node)

        # Run visitor
        visitor = RelationshipVisitor()
        visitor.visit(tree)

        return relations

    def _build_import_map(self, tree: ast.AST) -> Dict[str, str]:
        """Build local name -> qualified name map from imports."""
        import_map: Dict[str, str] = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    import_map[name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                base = node.module
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local = alias.asname or alias.name
                    import_map[local] = f"{base}.{alias.name}"
        return import_map


# === Helper Functions for Relationship Extraction ===

from typing import Tuple


def _get_scope_name(
    node: ast.AST,
    parents: Dict[ast.AST, ast.AST],
    tree: ast.AST,
) -> str:
    """Get fully qualified scope name for a function/class.

    Naming scheme (flat underscore-joined):
    - Module-level function: "function_name"
    - Class method: "ClassName.method_name"
    - Nested function: "outer_inner" (flat underscore-joined)
    - Inner class: "OuterClass.InnerClass"
    """
    parts = []
    current = node
    while current and current is not tree:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.insert(0, current.name)
        elif isinstance(current, ast.ClassDef):
            parts.insert(0, current.name)
        current = parents.get(current)
    return ".".join(parts) if parts else ""


@dataclass
class CallResolution:
    """Result of resolving a function call."""
    name: str
    resolved: bool


def _resolve_call(
    node: ast.Call,
    import_map: Dict[str, str],
    symbols: Dict[str, SymbolFact],
) -> CallResolution:
    """Resolve ast.Call to qualified name."""
    func = node.func
    parts: List[str] = []
    cur: ast.expr = func
    while isinstance(cur, ast.Attribute):
        parts.insert(0, cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.insert(0, cur.id)
    else:
        return CallResolution(name="<complex>", resolved=False)

    name = ".".join(parts)
    if len(parts) == 1 and parts[0] in import_map:
        return CallResolution(name=import_map[parts[0]], resolved=True)
    if len(parts) > 1 and parts[0] in import_map:
        base = import_map[parts[0]]
        return CallResolution(name=f"{base}.{'.'.join(parts[1:])}", resolved=True)

    # Check if it's a local symbol
    if parts[-1] in symbols:
        return CallResolution(name=parts[-1], resolved=True)

    return CallResolution(name=name, resolved=False)


def _resolve_name(node: ast.expr, import_map: Dict[str, str]) -> Optional[str]:
    """Resolve name expression to qualified form."""
    if isinstance(node, ast.Name):
        if node.id in import_map:
            return import_map[node.id]
        return node.id
    if isinstance(node, ast.Attribute):
        base = _resolve_name(node.value, import_map)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return None


# === Convenience Functions ===

def extract_facts(path: Path) -> ModuleFacts:
    """Extract all facts from a Python file.

    Args:
        path: Path to the Python file to analyze.

    Returns:
        ModuleFacts containing extracted symbols, relationships, and metadata.
    """
    extractor = ASTFactExtractor()
    return extractor.extract(path)
