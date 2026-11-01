"""
Restricted Structured Text-like syntax to custom PLC-IR transpiler.

This front end is not a full IEC language implementation or a graphical
Ladder Diagram importer. One supported statement is expected per source line.
Supports:
  - Boolean coil assignments: `Y0 := (X0 OR Y0) AND NOT X1;`
  - Latching calls: `IF X0 THEN SET(Y0); END_IF;`, `IF X1 THEN RST(Y0); END_IF;`
  - Timer function blocks: `TON(T0, IN := X0, PT := 5);`
  - Counter function blocks: `CTU(C0, CU := X0, PV := 3);`
  - Register operations: `IF X0 THEN D0 := 10; END_IF;`, `IF X0 THEN D0 := D0 + 2; END_IF;`
  - Inline comparisons: `Y0 := X0 AND (D0 > 10);`, `Y0 := (D0 == 5);`
"""

import ast
import re
from typing import List

from .isa import Opcode
from .numeric import require_integer


class STTranspiler:
    """
    Compiles the documented ST-like statement subset into PLC-IR.
    """

    _IF_THEN_RE = re.compile(
        r"^\s*IF\s+(.+?)\s+THEN\s+(.+?)\s*;?\s*END_IF\s*;?\s*$",
        re.IGNORECASE,
    )
    _TON_RE = re.compile(
        r"^\s*TON\s*\(\s*([TC]\d+)\s*,\s*IN\s*:=\s*(.+?)\s*,\s*PT\s*:=\s*K?(\d+)\s*\)\s*;?\s*$",
        re.IGNORECASE,
    )
    _CTU_RE = re.compile(
        r"^\s*CTU\s*\(\s*(C\d+)\s*,\s*CU\s*:=\s*(.+?)\s*,\s*PV\s*:=\s*K?(\d+)\s*\)\s*;?\s*$",
        re.IGNORECASE,
    )
    _SET_RST_RE = re.compile(
        r"^\s*(SET|RST)\s*\(\s*([A-Za-z0-9_.]+)\s*\)\s*;?\s*$",
        re.IGNORECASE,
    )
    _ADD_ASSIGN_RE = re.compile(
        r"^\s*(D\d+)\s*:=\s*\1\s*\+\s*K?(-?\d+)\s*;?\s*$",
        re.IGNORECASE,
    )
    _MOV_ASSIGN_RE = re.compile(
        r"^\s*(D\d+)\s*:=\s*K?(-?\d+)\s*;?\s*$",
        re.IGNORECASE,
    )
    _ASSIGN_RE = re.compile(r"^\s*([A-Za-z0-9_.]+)\s*:=\s*(.+?)\s*;?\s*$")

    @classmethod
    def transpile_line(cls, st_line: str) -> List[str]:
        """Transpile a single ST statement into a list of IL instructions."""
        line = st_line.strip()
        if not line or line.startswith("//") or line.startswith("(*"):
            return []

        # 1. TON(T0, IN := expr, PT := 5);
        m_ton = cls._TON_RE.match(line)
        if m_ton:
            t_sym = m_ton.group(1).upper()
            in_expr = m_ton.group(2).strip()
            pt_val = m_ton.group(3).strip()
            il_lines = cls._compile_condition_expr(in_expr)
            il_lines.append(f"TON {t_sym}, K{pt_val}")
            return il_lines

        # 2. CTU(C0, CU := expr, PV := 3);
        m_ctu = cls._CTU_RE.match(line)
        if m_ctu:
            c_sym = m_ctu.group(1).upper()
            cu_expr = m_ctu.group(2).strip()
            pv_val = m_ctu.group(3).strip()
            il_lines = cls._compile_condition_expr(cu_expr)
            il_lines.append(f"CTU {c_sym}, K{pv_val}")
            return il_lines

        # 3. IF <cond> THEN <action>; END_IF;
        m_if = cls._IF_THEN_RE.match(line)
        if m_if:
            cond_str = m_if.group(1).strip()
            action_str = m_if.group(2).strip()
            il_lines = cls._compile_condition_expr(cond_str)

            m_sr = cls._SET_RST_RE.match(action_str)
            if m_sr:
                op = m_sr.group(1).upper()
                target = m_sr.group(2).upper()
                il_lines.append(f"{op} {target}")
                return il_lines

            m_add = cls._ADD_ASSIGN_RE.match(action_str)
            if m_add:
                reg = m_add.group(1).upper()
                val = m_add.group(2)
                il_lines.append(f"ADD {reg}, K{val}")
                return il_lines

            m_mov = cls._MOV_ASSIGN_RE.match(action_str)
            if m_mov:
                reg = m_mov.group(1).upper()
                val = m_mov.group(2)
                il_lines.append(f"MOV {reg}, K{val}")
                return il_lines

            raise ValueError(f"Unsupported IF-THEN action in ST: '{action_str}'")

        # 4. Standard boolean coil assignment: Y0 := expr;
        match = cls._ASSIGN_RE.match(line)
        if not match:
            raise ValueError(f"Unsupported Structured Text statement: '{st_line}'")

        target = match.group(1).strip().upper()
        expr_str = match.group(2).strip()
        il_lines = cls._compile_condition_expr(expr_str)
        il_lines.append(f"OUT {target}")
        return il_lines

    @classmethod
    def _compile_condition_expr(cls, expr_str: str) -> List[str]:
        py_expr = re.sub(r"\bAND\b", "and", expr_str, flags=re.IGNORECASE)
        py_expr = re.sub(r"\bOR\b", "or", py_expr, flags=re.IGNORECASE)
        py_expr = re.sub(r"\bNOT\b", "not", py_expr, flags=re.IGNORECASE)
        tree = ast.parse(py_expr, mode="eval")
        return cls._emit_expr(tree.body, is_first=True)

    @classmethod
    def transpile(cls, st_code: str, append_end: bool = True) -> str:
        """Transpile a multi-line Structured Text program into an IL program string."""
        all_lines: List[str] = []
        for raw_line in st_code.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("(*"):
                continue
            all_lines.extend(cls.transpile_line(stripped))
        if append_end and (not all_lines or all_lines[-1] != "END"):
            all_lines.append("END")
        return "\n".join(all_lines)

    @classmethod
    def _is_atomic(cls, node: ast.AST) -> bool:
        """Return True if node is a single variable, NOT(variable), or comparison."""
        if isinstance(node, (ast.Name, ast.Attribute, ast.Compare)):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return isinstance(node.operand, (ast.Name, ast.Attribute))
        return False

    @classmethod
    def _symbol_name(cls, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id.upper()
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return f"{node.value.id.upper()}.{node.attr.upper()}"
        raise ValueError(f"Unsupported AST symbol node: {ast.dump(node)}")

    @classmethod
    def _emit_compare(cls, node: ast.Compare, is_first: bool) -> List[str]:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Chained comparisons are not supported in ST")
        reg = cls._symbol_name(node.left)
        comp = node.comparators[0]
        if isinstance(comp, ast.Constant) and isinstance(comp.value, (int, float)):
            k_val = int(require_integer(comp.value))
        elif (
            isinstance(comp, ast.UnaryOp)
            and isinstance(comp.op, ast.USub)
            and isinstance(comp.operand, ast.Constant)
        ):
            k_val = int(require_integer(-comp.operand.value))
        else:
            raise ValueError("Comparison RHS must be an integer constant")

        op = node.ops[0]
        if isinstance(op, ast.Eq):
            cmp_instr = f"CMP_EQ {reg}, K{k_val}"
        elif isinstance(op, ast.Gt):
            cmp_instr = f"CMP_GT {reg}, K{k_val}"
        elif isinstance(op, ast.Lt):
            cmp_instr = f"CMP_LT {reg}, K{k_val}"
        else:
            raise ValueError(f"Unsupported comparison operator: {ast.dump(op)}")

        if is_first:
            # x OR NOT x constructs True without reserving a mutable relay.
            return ["LD X0", "ORI X0", cmp_instr]
        return [cmp_instr]

    @classmethod
    def _emit_atomic(cls, node: ast.AST, op_pos: str, op_neg: str, is_first: bool = False) -> List[str]:
        if isinstance(node, ast.Compare):
            if op_pos == "OR":
                raise ValueError("Comparison inside OR branch requires intermediate relay M")
            return cls._emit_compare(node, is_first=is_first)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            sym = cls._symbol_name(node.operand)
            return [f"{op_neg} {sym}"]
        sym = cls._symbol_name(node)
        return [f"{op_pos} {sym}"]

    @classmethod
    def _emit_expr(cls, node: ast.AST, is_first: bool = True) -> List[str]:
        if cls._is_atomic(node):
            if is_first:
                return cls._emit_atomic(node, "LD", "LDI", is_first=True)
            raise ValueError("Atomic non-first expression requires operator context")

        if isinstance(node, ast.BoolOp):
            is_and = isinstance(node.op, ast.And)
            op_pos = "AND" if is_and else "OR"
            op_neg = "ANI" if is_and else "ORI"

            lines: List[str] = []
            first_val = node.values[0]
            if cls._is_atomic(first_val):
                lines.extend(cls._emit_atomic(first_val, "LD", "LDI", is_first=True))
            else:
                lines.extend(cls._emit_expr(first_val, is_first=True))

            for child in node.values[1:]:
                if cls._is_atomic(child):
                    lines.extend(cls._emit_atomic(child, op_pos, op_neg, is_first=False))
                else:
                    raise ValueError(
                        "Complex nested sub-expression on RHS of operator requires intermediate relay M"
                    )
            return lines

        raise ValueError(f"Unsupported ST expression node: {ast.dump(node)}")
