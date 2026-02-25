from __future__ import annotations
"""Expression evaluator for batch pipeline variable interpolation."""
import re
from typing import Any, Optional
from scout.batch_context import BatchContext

class ExpressionEvaluator:
    """Evaluate expressions with ${var} interpolation and comparisons."""
    
    # Pattern: ${var.path} or ${comparison expr}
    INTERP_PATTERN = re.compile(r'\$\{([^}]+)\}')
    
    def __init__(self, context: BatchContext):
        self.context = context
    
    def evaluate(self, expr: str) -> Any:
        """
        Evaluate an expression.
        
        Supports:
        - ${var} - direct variable access
        - ${var.nested} - dot-path access
        - ${var > 0} - comparison
        - ${var == "value"} - equality
        - ${var and other} - boolean logic
        - Plain values (strings, numbers, booleans)
        """
        if not expr:
            return None
        
        expr = expr.strip()
        
        # Check if it's a ${...} expression
        if expr.startswith('${'):
            inner = expr[2:]  # Remove ${
            
            # Find the correct closing } using smart detection
            end_pos = self._find_variable_end(inner)
            
            if end_pos == len(inner):
                # No closing brace - treat entire thing as variable or comparison
                has_comp_op = any(op in inner for op in ['>=', '<=', '!=', '==', '>', '<', ' and ', ' or '])
                if has_comp_op:
                    return self._eval_interpolation(inner)
                return self._resolve_var(inner)
            
            # Split at closing brace
            var_part = inner[:end_pos].strip()
            comp_part = inner[end_pos+1:].strip()
            
            # Check if there's a comparison after the }
            has_comp = any(comp_part.startswith(op) for op in ['>=', '<=', '!=', '==', '>', '<', ' and ', ' or '])
            
            if has_comp:
                # Combine var_part + comparison operator + rest
                # e.g. "count} > 3" becomes "count > 3"
                return self._eval_interpolation(var_part + " " + comp_part)
            else:
                # Just a variable, no comparison
                return self._resolve_var(var_part)
        
        # Plain value - try to parse as JSON primitive
        return self._parse_value(expr)
    
    def _find_variable_end(self, expression: str) -> int:
        """
        Find the closing } for a ${ variable.
        
        Strategy:
        - Scan forward from start
        - If we hit } AND next non-space char is operator (<, >, ==, !=, etc) → closing brace
        - If we hit } AND next char is . or alphanumeric → part of variable name
        - If we hit } AND it's end of string → closing brace
        """
        operators = ['<', '>', '=', '!', 'a', 'o', 'n']  # Start of operators
        
        i = 0
        while i < len(expression):
            if expression[i] == '}':
                # Check what comes after
                rest = expression[i+1:].strip()
                if not rest:
                    return i  # End of string, this is closing
                # Check if operator follows
                if rest[0] in operators:
                    return i  # Operator follows, this is closing
                # Otherwise, keep scanning (} is part of variable name)
            i += 1
        
        return len(expression)  # No closing brace found
    
    def _eval_interpolation(self, inner: str) -> Any:
        """Evaluate contents of ${...}."""
        # Check for comparison operators - need to handle spaces
        for op in ['>=', '<=', '!=', '==', '>', '<', ' and ', ' or ']:
            if op in inner:
                # Split on the operator, handling spaces around it
                # Escape special regex chars in operator for split
                import re
                pattern = re.escape(op)
                parts = re.split(pattern, inner, maxsplit=1)
                if len(parts) == 2:
                    left = self._resolve_var(parts[0].strip())
                    right = self._resolve_var(parts[1].strip())
                    
                    if op == '>':
                        return left > right
                    elif op == '<':
                        return left < right
                    elif op == '>=':
                        return left >= right
                    elif op == '<=':
                        return left <= right
                    elif op == '==':
                        return left == right
                    elif op == '!=':
                        return left != right
                    elif op == ' and ':
                        return bool(left) and bool(right)
                    elif op == ' or ':
                        return bool(left) or bool(right)
        
        # Simple variable resolution
        return self._resolve_var(inner)
    
    def _resolve_var(self, name: str) -> Any:
        """Resolve a variable name to its value."""
        name = name.strip()
        
        # Handle negation
        if name.startswith('!'):
            var_name = name[1:].strip()
            return not self.context.get_var(var_name, False)
        
        # Handle .length for lists (must check before generic dot-path)
        if name.endswith('.length'):
            base_name = name[:-7]  # Remove .length
            value = self.context.get_var(base_name)
            if isinstance(value, (list, tuple, str, dict)):
                return len(value)
            return None
        
        # Check for boolean literal
        if name.lower() == 'true':
            return True
        if name.lower() == 'false':
            return False
        if name.lower() == 'none' or name.lower() == 'null':
            return None
        
        # Try to parse as number
        try:
            if '.' in name:
                return float(name)
            return int(name)
        except ValueError:
            pass
        
        # Resolve as variable (supports dot-paths)
        return self.context.get_var(name)
    
    def _parse_value(self, value: str) -> Any:
        """Parse a plain value string."""
        value = value.strip()
        
        # Boolean
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        
        # None
        if value.lower() == 'none' or value.lower() == 'null':
            return None
        
        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # String - remove quotes if present
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        
        return value
    
    def evaluate_condition(self, condition: Optional[str]) -> bool:
        """
        Evaluate a condition for if/skip_if/stop_if.
        
        Returns True if:
        - condition is None or empty
        - condition evaluates to truthy value
        - condition is "true" (case insensitive)
        """
        if not condition:
            return True
        
        result = self.evaluate(condition)
        
        # Handle common truthy patterns
        if result is None:
            return False
        if isinstance(result, bool):
            return result
        if isinstance(result, (int, float)):
            return result > 0
        if isinstance(result, str):
            return result.lower() == 'true' or len(result) > 0
        if isinstance(result, (list, dict)):
            return len(result) > 0
        
        return bool(result)
    
    def interpolate_args(self, args: dict) -> dict:
        """Interpolate ${...} in all arg values."""
        if not args:
            return {}
        
        result = {}
        for key, value in args.items():
            if isinstance(value, str):
                result[key] = self._interpolate_string(value)
            elif isinstance(value, dict):
                result[key] = self.interpolate_args(value)
            elif isinstance(value, list):
                result[key] = [self._interpolate_string(v) if isinstance(v, str) else v for v in value]
            else:
                result[key] = value
        
        return result
    
    def _interpolate_string(self, s: str) -> str:
        """Replace ${var} with actual values in a string."""
        def replacer(match):
            var_path = match.group(1)
            value = self._resolve_var(var_path)
            if value is None:
                return match.group(0)  # Keep original if not found
            return str(value)
        
        return self.INTERP_PATTERN.sub(replacer, s)
