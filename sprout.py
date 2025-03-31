import re
import threading
import queue
import os
import time
import traceback
import sys
from enum import Enum
from typing import List, Dict, Any, Callable, Optional, Union, Tuple

# Token types
TOKEN_TYPES = [
    ("NUMBER", r"\d+(\.\d+)?"),
    ("STRING", r'"([^"\\]|\\.)*"'),
    ("IDENT", r"[a-zA-Z_]\w*"),
    ("ASSIGN", r"="),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("TIMES", r"\*"),
    ("DIVIDE", r"/"),
    ("MOD", r"%"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("GT", r">"),
    ("LT", r"<"),
    ("GTE", r">="),
    ("LTE", r"<="),
    ("EQ", r"=="),
    ("NEQ", r"!="),
    ("AND", r"&&"),
    ("OR", r"\|\|"),
    ("NOT", r"!"),
    ("COMMA", r","),
    ("DOT", r"\."),
    ("SEMICOLON", r";"),
    ("COLON", r":"),
    ("COMMENT", r"#.*"),
    ("NEWLINE", r"\n"),
    ("SKIP", r"[ \t]+"),
    ("MISMATCH", r"."),
]

KEYWORDS = {
    "let", "print", "if", "else", "elif", "while", "for", "in", "break", "continue",
    "fn", "return", "spawn", "join", "lock", "unlock", "channel", "send", "receive",
    "import", "public", "private", "as", "test", "assert", "before", "after",
    "true", "false", "null", "try", "catch", "finally", "class", "new", "this", "super"
}

# Error handling
class SproutError(Exception):
    def __init__(self, message, line=None, column=None, filename=None):
        self.message = message
        self.line = line
        self.column = column
        self.filename = filename
        super().__init__(self.format_error())
    
    def format_error(self):
        location = ""
        if self.filename:
            location += f"in file '{self.filename}' "
        if self.line is not None:
            location += f"at line {self.line}"
            if self.column is not None:
                location += f", column {self.column}"
        
        if location:
            return f"{self.__class__.__name__}: {self.message} ({location})"
        return f"{self.__class__.__name__}: {self.message}"

class SyntaxError(SproutError): pass
class RuntimeError(SproutError): pass
class NameError(SproutError): pass
class TypeError(SproutError): pass
class ValueError(SproutError): pass
class AssertionError(SproutError): pass
class ImportError(SproutError): pass

# AST nodes
class Node:
    def __repr__(self):
        attrs = ', '.join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

class Block(Node):
    def __init__(self, statements):
        self.statements = statements

class Number(Node):
    def __init__(self, value):
        self.value = float(value) if '.' in str(value) else int(value)

class String(Node):
    def __init__(self, value):
        # Remove the quotes and handle escape sequences
        self.value = bytes(value[1:-1], "utf-8").decode("unicode_escape")

class Boolean(Node):
    def __init__(self, value):
        self.value = value == "true"

class Null(Node):
    def __init__(self):
        pass

class List(Node):
    def __init__(self, elements):
        self.elements = elements

class Dictionary(Node):
    def __init__(self, pairs):
        self.pairs = pairs

class Variable(Node):
    def __init__(self, name):
        self.name = name

class VariableDeclaration(Node):
    def __init__(self, name, value=None):
        self.name = name
        self.value = value

class Assignment(Node):
    def __init__(self, target, value):
        self.target = target
        self.value = value

class BinaryOp(Node):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class UnaryOp(Node):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

class If(Node):
    def __init__(self, condition, true_block, elif_blocks=None, else_block=None):
        self.condition = condition
        self.true_block = true_block
        self.elif_blocks = elif_blocks or []  # List of (condition, block) tuples
        self.else_block = else_block

class While(Node):
    def __init__(self, condition, block):
        self.condition = condition
        self.block = block

class For(Node):
    def __init__(self, var_name, iterable, block):
        self.var_name = var_name
        self.iterable = iterable
        self.block = block

class Break(Node):
    pass

class Continue(Node):
    pass

class FunctionDefinition(Node):
    def __init__(self, name, params, body, visibility="private"):
        self.name = name
        self.params = params
        self.body = body
        self.visibility = visibility

class Return(Node):
    def __init__(self, value=None):
        self.value = value

class FunctionCall(Node):
    def __init__(self, name, args):
        self.name = name
        self.args = args

class MemberAccess(Node):
    def __init__(self, object_expr, member):
        self.object_expr = object_expr
        self.member = member

class IndexAccess(Node):
    def __init__(self, object_expr, index):
        self.object_expr = object_expr
        self.index = index

class ClassDefinition(Node):
    def __init__(self, name, methods, visibility="private"):
        self.name = name
        self.methods = methods
        self.visibility = visibility

class NewInstance(Node):
    def __init__(self, class_name, args):
        self.class_name = class_name
        self.args = args

class This(Node):
    pass

class Super(Node):
    def __init__(self, method=None, args=None):
        self.method = method
        self.args = args or []

class Try(Node):
    def __init__(self, try_block, catch_var, catch_block, finally_block=None):
        self.try_block = try_block
        self.catch_var = catch_var
        self.catch_block = catch_block
        self.finally_block = finally_block

class Spawn(Node):
    def __init__(self, function_call):
        self.function_call = function_call

class Join(Node):
    def __init__(self, thread_var):
        self.thread_var = thread_var

class Lock(Node):
    def __init__(self, lock_name):
        self.lock_name = lock_name

class Unlock(Node):
    def __init__(self, lock_name):
        self.lock_name = lock_name

class ChannelCreate(Node):
    def __init__(self, name, capacity=0):
        self.name = name
        self.capacity = capacity

class Send(Node):
    def __init__(self, channel, value):
        self.channel = channel
        self.value = value

class Receive(Node):
    def __init__(self, channel):
        self.channel = channel

class Import(Node):
    def __init__(self, module_name, alias=None):
        self.module_name = module_name
        self.alias = alias or module_name

class Test(Node):
    def __init__(self, name, before, body, after):
        self.name = name
        self.before = before
        self.body = body
        self.after = after

class Assert(Node):
    def __init__(self, expr, message=None):
        self.expr = expr
        self.message = message

class Print(Node):
    def __init__(self, expr):
        self.expr = expr

# Lexer implementation
def tokenize(code, filename=None):
    tokens = []
    line = 1
    column = 1
    
    code = code + '\n'  # Ensure the code ends with a newline
    pos = 0
    
    while pos < len(code):
        match = None
        for token_type, pattern in TOKEN_TYPES:
            regex = re.compile(pattern)
            match = regex.match(code, pos)
            if match:
                value = match.group(0)
                if token_type == "IDENT" and value in KEYWORDS:
                    token_type = value.upper()
                
                if token_type != "SKIP" and token_type != "COMMENT" and token_type != "NEWLINE":
                    tokens.append((token_type, value, line, column))
                
                # Update line and column counters
                if token_type == "NEWLINE":
                    line += 1
                    column = 1
                else:
                    column += len(value)
                
                pos = match.end()
                break
        
        if not match:
            raise SyntaxError(f"Invalid syntax", line, column, filename)
    
    return tokens

# Parser with complete implementation for all token types
class Parser:
    def __init__(self, tokens, filename=None):
        self.tokens = tokens
        self.pos = 0
        self.filename = filename
        self.line = 0
        self.column = 0

    def error(self, message):
        token = self.peek()
        if token[0] != "EOF":
            line, column = token[2], token[3]
        else:
            line, column = self.line, self.column
        raise SyntaxError(message, line, column, self.filename)

    def peek(self):
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            self.line, self.column = token[2], token[3]
            return token
        return ("EOF", None, self.line, self.column)

    def advance(self):
        token = self.peek()
        self.pos += 1
        return token

    def match(self, expected):
        token = self.peek()
        if token[0] == expected:
            return self.advance()
        else:
            self.error(f"Expected {expected}, got {token[0]}")

    def match_optional(self, expected):
        if self.peek()[0] == expected:
            return self.advance()
        return None

    def parse(self):
        statements = []
        while self.pos < len(self.tokens):
            statements.append(self.statement())
            self.match_optional("SEMICOLON")  # Optional semicolon
        return Block(statements)

    def statement(self):
        tok_type = self.peek()[0]
        
        if tok_type == "LET":
            return self.var_declaration()
        elif tok_type == "IF":
            return self.if_statement()
        elif tok_type == "WHILE":
            return self.while_statement()
        elif tok_type == "FOR":
            return self.for_statement()
        elif tok_type == "BREAK":
            self.advance()
            return Break()
        elif tok_type == "CONTINUE":
            self.advance()
            return Continue()
        elif tok_type == "FN":
            return self.function_definition()
        elif tok_type == "RETURN":
            return self.return_statement()
        elif tok_type == "LBRACE":
            return self.block()
        elif tok_type == "PRINT":
            return self.print_statement()
        elif tok_type == "IMPORT":
            return self.import_stmt()
        elif tok_type == "PUBLIC" or tok_type == "PRIVATE":
            visibility = self.advance()[1]
            stmt = self.statement()
            if hasattr(stmt, 'visibility'):
                stmt.visibility = visibility
            else:
                self.error(f"Cannot apply visibility to {type(stmt).__name__}")
            return stmt
        elif tok_type == "TEST":
            return self.test_stmt()
        elif tok_type == "ASSERT":
            return self.assert_stmt()
        elif tok_type == "CLASS":
            return self.class_definition()
        elif tok_type == "TRY":
            return self.try_statement()
        elif tok_type == "SPAWN":
            return self.spawn_statement()
        elif tok_type == "JOIN":
            return self.join_statement()
        elif tok_type == "LOCK":
            return self.lock_statement()
        elif tok_type == "UNLOCK":
            return self.unlock_statement()
        elif tok_type == "CHANNEL":
            return self.channel_statement()
        elif tok_type == "SEND":
            return self.send_statement()
        elif tok_type == "RECEIVE":
            return self.receive_statement()
        else:
            # Could be an assignment or a function call
            expr = self.expression()
            if self.peek()[0] == "ASSIGN":
                self.advance()  # Consume ASSIGN
                value = self.expression()
                return Assignment(expr, value)
            return expr

    def var_declaration(self):
        self.match("LET")
        name = self.match("IDENT")[1]
        value = None
        if self.peek()[0] == "ASSIGN":
            self.advance()  # Consume ASSIGN
            value = self.expression()
        return VariableDeclaration(name, value)

    def if_statement(self):
        self.match("IF")
        condition = self.expression()
        true_block = self.block()
        
        elif_blocks = []
        while self.peek()[0] == "ELIF":
            self.advance()  # Consume ELIF
            elif_condition = self.expression()
            elif_block = self.block()
            elif_blocks.append((elif_condition, elif_block))
        
        else_block = None
        if self.peek()[0] == "ELSE":
            self.advance()  # Consume ELSE
            else_block = self.block()
        
        return If(condition, true_block, elif_blocks, else_block)

    def while_statement(self):
        self.match("WHILE")
        condition = self.expression()
        block = self.block()
        return While(condition, block)

    def for_statement(self):
        self.match("FOR")
        var_name = self.match("IDENT")[1]
        self.match("IN")
        iterable = self.expression()
        block = self.block()
        return For(var_name, iterable, block)

    def function_definition(self):
        self.match("FN")
        name = self.match("IDENT")[1]
        self.match("LPAREN")
        params = []
        
        if self.peek()[0] != "RPAREN":
            params.append(self.match("IDENT")[1])
            while self.peek()[0] == "COMMA":
                self.advance()  # Consume COMMA
                params.append(self.match("IDENT")[1])
        
        self.match("RPAREN")
        body = self.block()
        return FunctionDefinition(name, params, body)

    def return_statement(self):
        self.match("RETURN")
        if self.peek()[0] != "SEMICOLON" and self.peek()[0] != "RBRACE":
            value = self.expression()
            return Return(value)
        return Return()

    def block(self):
        self.match("LBRACE")
        statements = []
        while self.peek()[0] != "RBRACE":
            stmt = self.statement()
            statements.append(stmt)
            self.match_optional("SEMICOLON")  # Optional semicolon
        self.match("RBRACE")
        return Block(statements)

    def print_statement(self):
        self.match("PRINT")
        expr = self.expression()
        return Print(expr)

    def import_stmt(self):
        self.match("IMPORT")
        module_name = self.match("IDENT")[1]
        alias = module_name
        if self.peek()[0] == "AS":
            self.advance()  # Consume AS
            alias = self.match("IDENT")[1]
        return Import(module_name, alias)

    def test_stmt(self):
        self.match("TEST")
        name = self.match("IDENT")[1]
        self.match("LBRACE")
        
        before = Block([])
        body = []
        after = Block([])
        
        while self.peek()[0] != "RBRACE":
            if self.peek()[0] == "BEFORE":
                self.advance()  # Consume BEFORE
                before = self.block()
            elif self.peek()[0] == "AFTER":
                self.advance()  # Consume AFTER
                after = self.block()
            else:
                body.append(self.statement())
                self.match_optional("SEMICOLON")  # Optional semicolon
        
        self.match("RBRACE")
        return Test(name, before, Block(body), after)

    def assert_stmt(self):
        self.match("ASSERT")
        expr = self.expression()
        message = None
        if self.peek()[0] == "COMMA":
            self.advance()  # Consume COMMA
            message = self.expression()
        return Assert(expr, message)

    def class_definition(self):
        self.match("CLASS")
        name = self.match("IDENT")[1]
        self.match("LBRACE")
        methods = []
        
        while self.peek()[0] != "RBRACE":
            if self.peek()[0] == "FN":
                methods.append(self.function_definition())
            else:
                self.error(f"Expected method definition, got {self.peek()[0]}")
            self.match_optional("SEMICOLON")  # Optional semicolon
        
        self.match("RBRACE")
        return ClassDefinition(name, methods)

    def try_statement(self):
        self.match("TRY")
        try_block = self.block()
        self.match("CATCH")
        self.match("LPAREN")
        catch_var = self.match("IDENT")[1]
        self.match("RPAREN")
        catch_block = self.block()
        
        finally_block = None
        if self.peek()[0] == "FINALLY":
            self.advance()  # Consume FINALLY
            finally_block = self.block()
        
        return Try(try_block, catch_var, catch_block, finally_block)

    def spawn_statement(self):
        self.match("SPAWN")
        call = self.function_call()
        return Spawn(call)

    def join_statement(self):
        self.match("JOIN")
        thread_var = self.expression()
        return Join(thread_var)

    def lock_statement(self):
        self.match("LOCK")
        lock_name = self.expression()
        return Lock(lock_name)

    def unlock_statement(self):
        self.match("UNLOCK")
        lock_name = self.expression()
        return Unlock(lock_name)

    def channel_statement(self):
        self.match("CHANNEL")
        name = self.match("IDENT")[1]
        capacity = 0
        if self.peek()[0] == "COMMA":
            self.advance()  # Consume COMMA
            capacity = self.expression()
        return ChannelCreate(name, capacity)

    def send_statement(self):
        self.match("SEND")
        channel = self.expression()
        self.match("COMMA")
        value = self.expression()
        return Send(channel, value)

    def receive_statement(self):
        self.match("RECEIVE")
        channel = self.expression()
        return Receive(channel)

    def expression(self):
        return self.logical_or()

    def logical_or(self):
        expr = self.logical_and()
        while self.peek()[0] == "OR":
            op = self.advance()[0]
            right = self.logical_and()
            expr = BinaryOp(op, expr, right)
        return expr

    def logical_and(self):
        expr = self.equality()
        while self.peek()[0] == "AND":
            op = self.advance()[0]
            right = self.equality()
            expr = BinaryOp(op, expr, right)
        return expr

    def equality(self):
        expr = self.comparison()
        while self.peek()[0] in ["EQ", "NEQ"]:
            op = self.advance()[0]
            right = self.comparison()
            expr = BinaryOp(op, expr, right)
        return expr

    def comparison(self):
        expr = self.term()
        while self.peek()[0] in ["GT", "LT", "GTE", "LTE"]:
            op = self.advance()[0]
            right = self.term()
            expr = BinaryOp(op, expr, right)
        return expr

    def term(self):
        expr = self.factor()
        while self.peek()[0] in ["PLUS", "MINUS"]:
            op = self.advance()[0]
            right = self.factor()
            expr = BinaryOp(op, expr, right)
        return expr

    def factor(self):
        expr = self.unary()
        while self.peek()[0] in ["TIMES", "DIVIDE", "MOD"]:
            op = self.advance()[0]
            right = self.unary()
            expr = BinaryOp(op, expr, right)
        return expr

    def unary(self):
        if self.peek()[0] in ["MINUS", "NOT"]:
            op = self.advance()[0]
            expr = self.unary()
            return UnaryOp(op, expr)
        return self.call()

    def call(self):
        expr = self.primary()
        
        while True:
            if self.peek()[0] == "LPAREN":
                self.advance()  # Consume LPAREN
                args = []
                
                if self.peek()[0] != "RPAREN":
                    args.append(self.expression())
                    while self.peek()[0] == "COMMA":
                        self.advance()  # Consume COMMA
                        args.append(self.expression())
                
                self.match("RPAREN")
                
                if isinstance(expr, Variable):
                    expr = FunctionCall(expr.name, args)
                elif isinstance(expr, MemberAccess):
                    # Handle method calls like obj.method()
                    if isinstance(expr.object_expr, Variable) and isinstance(expr.member, Variable):
                        expr = FunctionCall(f"{expr.object_expr.name}.{expr.member.name}", args)
                    else:
                        self.error("Invalid method call")
                else:
                    self.error("Cannot call a non-function")
            
            elif self.peek()[0] == "DOT":
                self.advance()  # Consume DOT
                member = self.primary()
                expr = MemberAccess(expr, member)
            
            elif self.peek()[0] == "LBRACKET":
                self.advance()  # Consume LBRACKET
                index = self.expression()
                self.match("RBRACKET")
                expr = IndexAccess(expr, index)
            
            else:
                break
        
        return expr

    def primary(self):
        tok_type = self.peek()[0]
        
        if tok_type == "NUMBER":
            return Number(self.advance()[1])
        elif tok_type == "STRING":
            return String(self.advance()[1])
        elif tok_type == "TRUE":
            self.advance()
            return Boolean("true")
        elif tok_type == "FALSE":
            self.advance()
            return Boolean("false")
        elif tok_type == "NULL":
            self.advance()
            return Null()
        elif tok_type == "IDENT":
            return Variable(self.advance()[1])
        elif tok_type == "LPAREN":
            self.advance()  # Consume LPAREN
            expr = self.expression()
            self.match("RPAREN")
            return expr
        elif tok_type == "LBRACKET":
            return self.list_literal()
        elif tok_type == "LBRACE":
            return self.dict_literal()
        elif tok_type == "NEW":
            return self.new_instance()
        elif tok_type == "THIS":
            self.advance()
            return This()
        elif tok_type == "SUPER":
            return self.super_call()
        else:
            self.error(f"Unexpected token {tok_type}")

    def list_literal(self):
        self.match("LBRACKET")
        elements = []
        
        if self.peek()[0] != "RBRACKET":
            elements.append(self.expression())
            while self.peek()[0] == "COMMA":
                self.advance()  # Consume COMMA
                if self.peek()[0] == "RBRACKET":  # Allow trailing comma
                    break
                elements.append(self.expression())
        
        self.match("RBRACKET")
        return List(elements)

    def dict_literal(self):
        self.match("LBRACE")
        pairs = []
        
        if self.peek()[0] != "RBRACE":
            key = self.expression()
            self.match("COLON")
            value = self.expression()
            pairs.append((key, value))
            
            while self.peek()[0] == "COMMA":
                self.advance()  # Consume COMMA
                if self.peek()[0] == "RBRACE":  # Allow trailing comma
                    break
                key = self.expression()
                self.match("COLON")
                value = self.expression()
                pairs.append((key, value))
        
        self.match("RBRACE")
        return Dictionary(pairs)

    def new_instance(self):
        self.match("NEW")
        class_name = self.match("IDENT")[1]
        self.match("LPAREN")
        args = []
        
        if self.peek()[0] != "RPAREN":
            args.append(self.expression())
            while self.peek()[0] == "COMMA":
                self.advance()  # Consume COMMA
                args.append(self.expression())
        
        self.match("RPAREN")
        return NewInstance(class_name, args)

    def super_call(self):
        self.match("SUPER")
        method = None
        args = []
        
        if self.peek()[0] == "DOT":
            self.advance()  # Consume DOT
            method = self.match("IDENT")[1]
            self.match("LPAREN")
            
            if self.peek()[0] != "RPAREN":
                args.append(self.expression())
                while self.peek()[0] == "COMMA":
                    self.advance()  # Consume COMMA
                    args.append(self.expression())
            
            self.match("RPAREN")
        
        return Super(method, args)

    def function_call(self):
        name = self.match("IDENT")[1]
        self.match("LPAREN")
        args = []
        
        if self.peek()[0] != "RPAREN":
            args.append(self.expression())
            while self.peek()[0] == "COMMA":
                self.advance()  # Consume COMMA
                args.append(self.expression())
        
        self.match("RPAREN")
        return FunctionCall(name, args)

# Enhanced interpreter with concurrency and class support
class Interpreter:
    def __init__(self, stdout=sys.stdout, stderr=sys.stderr):
        self.global_env = {}
        self.env_stack = [self.global_env]
        self.functions = {}
        self.classes = {}
        self.return_value = None
        self.break_flag = False
        self.continue_flag = False
        self.current_self = None
        self.current_class = None
        self.threads = {}
        self.locks = {}
        self.channels = {}
        self.modules = {}
        self.stdout = stdout
        self.stderr = stderr
        self.current_file = None
        
        # Initialize standard library
        self.init_stdlib()

    def get_env(self):
        return self.env_stack[-1]

    def push_env(self):
        self.env_stack.append({})

    def pop_env(self):
        if len(self.env_stack) > 1:
            return self.env_stack.pop()
        else:
            raise RuntimeError("Cannot pop global environment")

    def define_var(self, name, value):
        self.get_env()[name] = value
        return value

    def get_var(self, name):
        # Search for variable in all environments, from local to global
        for env in reversed(self.env_stack):
            if name in env:
                return env[name]
        raise NameError(f"Undefined variable '{name}'")

    def assign_var(self, name, value):
        # Search for variable in all environments, from local to global
        for env in reversed(self.env_stack):
            if name in env:
                env[name] = value
                return value
        raise NameError(f"Undefined variable '{name}'")

    def init_stdlib(self):
        # Math functions
        self.define_std_function("abs", abs)
        self.define_std_function("max", max)
        self.define_std_function("min", min)
        self.define_std_function("round", round)
        self.define_std_function("floor", lambda x: int(x) if x >= 0 else int(x) - 1)
        self.define_std_function("ceil", lambda x: int(x) + (1 if x > int(x) else 0))
        self.define_std_function("pow", pow)
        self.define_std_function("sqrt", lambda x: x ** 0.5)
        
        # String functions
        self.define_std_function("len", len)
        self.define_std_function("substr", lambda s, start, end=None: s[start:end])
        self.define_std_function("split", lambda s, delim: s.split(delim))
        self.define_std_function("join", lambda arr, delim: delim.join(arr))
        self.define_std_function("to_string", str)
        self.define_std_function("to_number", lambda s: float(s) if '.' in s else int(s))
        self.define_std_function("trim", lambda s: s.strip())
        
        # List functions
        self.define_std_function("push", lambda lst, item: lst.append(item) or lst)
        self.define_std_function("pop", lambda lst: lst.pop())
        self.define_std_function("shift", lambda lst: lst.pop(0))
        self.define_std_function("unshift", lambda lst, item: lst.insert(0, item) or lst)
        self.define_std_function("slice", lambda lst, start, end=None: lst[start:end])
        self.define_std_function("map", lambda lst, fn: [fn(x) for x in lst])
        self.define_std_function("filter", lambda lst, fn: [x for x in lst if fn(x)])
        self.define_std_function("reduce", lambda lst, fn, initial=0: 
                                 (lambda acc, lst: acc if not lst else 
                                  self.reduce(lst[1:], fn, fn(acc, lst[0])))
                                 (initial, lst))
        
        # Dictionary functions
        self.define_std_function("keys", lambda dct: list(dct.keys()))
        self.define_std_function("values", lambda dct: list(dct.values()))
        self.define_std_function("has_key", lambda dct, key: key in dct)
        
        # Type functions
        self.define_std_function("type", lambda x: type(x).__name__)
        self.define_std_function("is_number", lambda x: isinstance(x, (int, float)))
        self.define_std_function("is_string", lambda x: isinstance(x, str))
        self.define_std_function("is_list", lambda x: isinstance(x, list))
        self.define_std_function("is_dict", lambda x: isinstance(x, dict))
        self.define_std_function("is_null", lambda x: x is None)
        
        # System functions
        self.define_std_function("time", lambda: time.time())
        self.define_std_function("sleep", lambda ms: time.sleep(ms / 1000))
        self.define_std_function("rand", lambda min_val=0, max_val=1: min_val + (max_val - min_val) * __import__('random').random())
        self.define_std_function("rand_int", lambda min_val, max_val: __import__('random').randint(min_val, max_val))
        
        # IO functions
        self.define_std_function("read_file", self._read_file)
        self.define_std_function("write_file", self._write_file)
        self.define_std_function("append_file", self._append_file)
        self.define_std_function("file_exists", lambda path: os.path.exists(path))
        self.define_std_function("list_dir", lambda path=".": os.listdir(path))
        
        # Testing functions
        self.define_std_function("assert_eq", lambda a, b, msg=None: 
                                 self._assert(a == b, msg or f"Expected {a} to equal {b}"))
        self.define_std_function("assert_ne", lambda a, b, msg=None: 
                                 self._assert(a != b, msg or f"Expected {a} to not equal {b}"))
        self.define_std_function("assert_true", lambda cond, msg=None: 
                                 self._assert(cond, msg or "Assertion failed"))
        self.define_std_function("assert_false", lambda cond, msg=None: 
                                 self._assert(not cond, msg or f"Expected {cond} to be false"))
    
    def _assert(self, condition, message=None):
        if not condition:
            raise AssertionError(message or "Assertion failed")
    
    def _read_file(self, path):
        try:
            with open(path, 'r') as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read file '{path}': {str(e)}")
    
    def _write_file(self, path, content):
        try:
            with open(path, 'w') as f:
                f.write(str(content))
        except Exception as e:
            raise RuntimeError(f"Failed to write to file '{path}': {str(e)}")
    
    def _append_file(self, path, content):
        try:
            with open(path, 'a') as f:
                f.write(str(content))
        except Exception as e:
            raise RuntimeError(f"Failed to append to file '{path}': {str(e)}")
    
    def define_std_function(self, name, func):
        """Define a standard library function."""
        self.functions[name] = func
    
    def eval(self, node):
        """Evaluate an AST node."""
        # Handling different node types
        if isinstance(node, Block):
            result = None
            for stmt in node.statements:
                result = self.eval(stmt)
                if self.return_value is not None or self.break_flag or self.continue_flag:
                    break
            return result
        
        elif isinstance(node, Number):
            return node.value
        
        elif isinstance(node, String):
            return node.value
        
        elif isinstance(node, Boolean):
            return node.value
        
        elif isinstance(node, Null):
            return None
        
        elif isinstance(node, List):
            return [self.eval(element) for element in node.elements]
        
        elif isinstance(node, Dictionary):
            return {self.eval(key): self.eval(value) for key, value in node.pairs}
        
        elif isinstance(node, Variable):
            return self.get_var(node.name)
        
        elif isinstance(node, VariableDeclaration):
            value = None
            if node.value:
                value = self.eval(node.value)
            return self.define_var(node.name, value)
        
        elif isinstance(node, Assignment):
            value = self.eval(node.value)
            
            if isinstance(node.target, Variable):
                return self.assign_var(node.target.name, value)
            
            elif isinstance(node.target, IndexAccess):
                obj = self.eval(node.target.object_expr)
                index = self.eval(node.target.index)
                
                if isinstance(obj, list):
                    if not isinstance(index, int):
                        raise TypeError("List index must be an integer")
                    if index < 0 or index >= len(obj):
                        raise IndexError("List index out of range")
                    obj[index] = value
                    return value
                
                elif isinstance(obj, dict):
                    obj[index] = value
                    return value
                
                else:
                    raise TypeError(f"Cannot index into {type(obj).__name__}")
            
            elif isinstance(node.target, MemberAccess):
                obj = self.eval(node.target.object_expr)
                
                if isinstance(obj, dict):
                    if isinstance(node.target.member, Variable):
                        member_name = node.target.member.name
                        obj[member_name] = value
                        return value
                
                raise TypeError(f"Cannot access member of {type(obj).__name__}")
            
            else:
                raise TypeError(f"Invalid assignment target: {type(node.target).__name__}")
        
        elif isinstance(node, BinaryOp):
            left = self.eval(node.left)
            right = self.eval(node.right)
            
            if node.op == "PLUS":
                return left + right
            elif node.op == "MINUS":
                return left - right
            elif node.op == "TIMES":
                return left * right
            elif node.op == "DIVIDE":
                if right == 0:
                    raise ValueError("Division by zero")
                return left / right
            elif node.op == "MOD":
                return left % right
            elif node.op == "GT":
                return left > right
            elif node.op == "LT":
                return left < right
            elif node.op == "GTE":
                return left >= right
            elif node.op == "LTE":
                return left <= right
            elif node.op == "EQ":
                return left == right
            elif node.op == "NEQ":
                return left != right
            elif node.op == "AND":
                return bool(left and right)
            elif node.op == "OR":
                return bool(left or right)
            else:
                raise ValueError(f"Unknown operator: {node.op}")
        
        elif isinstance(node, UnaryOp):
            expr = self.eval(node.expr)
            
            if node.op == "MINUS":
                return -expr
            elif node.op == "NOT":
                return not expr
            else:
                raise ValueError(f"Unknown operator: {node.op}")
        
        elif isinstance(node, If):
            if self.eval(node.condition):
                return self.eval(node.true_block)
            
            for condition, block in node.elif_blocks:
                if self.eval(condition):
                    return self.eval(block)
            
            if node.else_block:
                return self.eval(node.else_block)
        
        elif isinstance(node, While):
            result = None
            while self.eval(node.condition):
                result = self.eval(node.block)
                
                if self.return_value is not None:
                    return self.return_value
                
                if self.break_flag:
                    self.break_flag = False
                    break
                
                if self.continue_flag:
                    self.continue_flag = False
                    continue
            
            return result
        
        elif isinstance(node, For):
            result = None
            iterable = self.eval(node.iterable)
            
            if not hasattr(iterable, '__iter__'):
                raise TypeError(f"'{type(iterable).__name__}' object is not iterable")
            
            for item in iterable:
                self.define_var(node.var_name, item)
                result = self.eval(node.block)
                
                if self.return_value is not None:
                    return self.return_value
                
                if self.break_flag:
                    self.break_flag = False
                    break
                
                if self.continue_flag:
                    self.continue_flag = False
                    continue
            
            return result
        
        elif isinstance(node, Break):
            self.break_flag = True
            return None
        
        elif isinstance(node, Continue):
            self.continue_flag = True
            return None
        
        elif isinstance(node, FunctionDefinition):
            self.functions[node.name] = node
            return node
        
        elif isinstance(node, Return):
            if node.value:
                self.return_value = self.eval(node.value)
            else:
                self.return_value = None
            return self.return_value
        
        elif isinstance(node, FunctionCall):
            if '.' in node.name:
                # Method call
                parts = node.name.split('.')
                obj_name, method_name = parts[0], parts[1]
                obj = self.get_var(obj_name)
                
                if obj_name in self.modules and method_name in self.modules[obj_name]:
                    func = self.modules[obj_name][method_name]
                    args = [self.eval(arg) for arg in node.args]
                    return self.call_function(func, args)
                
                raise NameError(f"Undefined method '{method_name}' on object '{obj_name}'")
            
            # Regular function call
            if node.name in self.functions:
                func = self.functions[node.name]
                args = [self.eval(arg) for arg in node.args]
                return self.call_function(func, args)
            
            raise NameError(f"Undefined function '{node.name}'")
        
        elif isinstance(node, MemberAccess):
            obj = self.eval(node.object_expr)
            
            if isinstance(obj, dict) and isinstance(node.member, Variable):
                member_name = node.member.name
                if member_name in obj:
                    return obj[member_name]
                raise NameError(f"Undefined member '{member_name}'")
            
            raise TypeError(f"Cannot access member of {type(obj).__name__}")
        
        elif isinstance(node, IndexAccess):
            obj = self.eval(node.object_expr)
            index = self.eval(node.index)
            
            if isinstance(obj, (list, str)):
                if not isinstance(index, int):
                    raise TypeError("Index must be an integer")
                if index < 0 or index >= len(obj):
                    raise IndexError("Index out of range")
                return obj[index]
            
            elif isinstance(obj, dict):
                if index not in obj:
                    raise KeyError(f"Key '{index}' not found in dictionary")
                return obj[index]
            
            raise TypeError(f"Cannot index into {type(obj).__name__}")
        
        elif isinstance(node, ClassDefinition):
            self.classes[node.name] = node
            return node
        
        elif isinstance(node, NewInstance):
            if node.class_name not in self.classes:
                raise NameError(f"Undefined class '{node.class_name}'")
            
            class_def = self.classes[node.class_name]
            instance = {}
            
            # Save current context
            saved_self = self.current_self
            saved_class = self.current_class
            
            # Set new context
            self.current_self = instance
            self.current_class = class_def
            
            # Initialize instance
            for method in class_def.methods:
                if method.name == "init":
                    args = [self.eval(arg) for arg in node.args]
                    self.push_env()
                    self.define_var("this", instance)
                    
                    for i, param in enumerate(method.params):
                        if i < len(args):
                            self.define_var(param, args[i])
                    
                    self.eval(method.body)
                    self.pop_env()
                    break
            
            # Restore context
            self.current_self = saved_self
            self.current_class = saved_class
            
            return instance
        
        elif isinstance(node, This):
            if self.current_self is None:
                raise RuntimeError("'this' used outside of class method")
            return self.current_self
        
        elif isinstance(node, Super):
            if self.current_class is None:
                raise RuntimeError("'super' used outside of class method")
            
            # TODO: Add class inheritance and super method calls
            raise NotImplementedError("Class inheritance not implemented yet")
        
        elif isinstance(node, Try):
            try:
                return self.eval(node.try_block)
            except Exception as e:
                # Save exception info
                error = {
                    "message": str(e),
                    "type": type(e).__name__
                }
                
                # Execute catch block with error variable
                self.push_env()
                self.define_var(node.catch_var, error)
                result = self.eval(node.catch_block)
                self.pop_env()
                
                return result
            finally:
                if node.finally_block:
                    self.eval(node.finally_block)
        
        elif isinstance(node, Spawn):
            func_call = node.function_call
            func_name = func_call.name
            
            if func_name not in self.functions:
                raise NameError(f"Undefined function '{func_name}'")
            
            func = self.functions[func_name]
            args = [self.eval(arg) for arg in func_call.args]
            
            # Create thread ID
            thread_id = f"thread_{len(self.threads) + 1}"
            
            # Start thread
            thread = threading.Thread(
                target=self.run_in_thread,
                args=(thread_id, func, args)
            )
            thread.daemon = True
            thread.start()
            
            self.threads[thread_id] = thread
            return thread_id
        
        elif isinstance(node, Join):
            thread_id = self.eval(node.thread_var)
            
            if thread_id not in self.threads:
                raise RuntimeError(f"Invalid thread ID: {thread_id}")
            
            # Wait for thread to complete
            self.threads[thread_id].join()
            return None
        
        elif isinstance(node, Lock):
            lock_name = self.eval(node.lock_name)
            
            if lock_name not in self.locks:
                self.locks[lock_name] = threading.RLock()
            
            self.locks[lock_name].acquire()
            return None
        
        elif isinstance(node, Unlock):
            lock_name = self.eval(node.lock_name)
            
            if lock_name not in self.locks:
                raise RuntimeError(f"Cannot unlock non-existent lock: {lock_name}")
            
            self.locks[lock_name].release()
            return None
        
        elif isinstance(node, ChannelCreate):
            capacity = self.eval(node.capacity) if isinstance(node.capacity, Node) else node.capacity
            self.channels[node.name] = queue.Queue(maxsize=capacity)
            return node.name
        
        elif isinstance(node, Send):
            channel_name = self.eval(node.channel)
            value = self.eval(node.value)
            
            if channel_name not in self.channels:
                raise RuntimeError(f"Channel not found: {channel_name}")
            
            self.channels[channel_name].put(value)
            return None
        
        elif isinstance(node, Receive):
            channel_name = self.eval(node.channel)
            
            if channel_name not in self.channels:
                raise RuntimeError(f"Channel not found: {channel_name}")
            
            return self.channels[channel_name].get()
        
        elif isinstance(node, Import):
            filename = f"{node.module_name}.sprout"
            if not os.path.exists(filename):
                raise ImportError(f"Module '{node.module_name}' not found")
            
            try:
                with open(filename) as f:
                    code = f.read()
                
                tokens = tokenize(code, filename)
                parser = Parser(tokens, filename)
                block = parser.parse()
                
                module_interpreter = Interpreter()
                module_interpreter.eval(block)
                
                module_env = {}
                for name, value in module_interpreter.functions.items():
                    if isinstance(value, FunctionDefinition) and value.visibility == "public":
                        module_env[name] = value
                
                self.modules[node.alias] = module_env
                return node.alias
            
            except Exception as e:
                raise ImportError(f"Error importing module '{node.module_name}': {str(e)}")
        
        elif isinstance(node, Test):
            print(f"Running test: {node.name}")
            start_time = time.time()
            
            try:
                # Create a new interpreter with the same environment
                test_interpreter = Interpreter()
                
                # Copy functions and classes
                test_interpreter.functions = self.functions.copy()
                test_interpreter.classes = self.classes.copy()
                
                # Run before setup
                test_interpreter.eval(node.before)
                
                # Run test body
                test_interpreter.eval(node.body)
                
                # Run after cleanup
                test_interpreter.eval(node.after)
                
                duration = (time.time() - start_time) * 1000
                print(f"✅ Passed: {node.name} ({duration:.2f}ms)")
            
            except Exception as e:
                print(f"❌ Failed: {node.name} — {str(e)}")
                traceback.print_exc()
            
            return None
        
        elif isinstance(node, Assert):
            result = self.eval(node.expr)
            
            if not result:
                message = self.eval(node.message) if node.message else "Assertion failed"
                raise AssertionError(message)
            
            return True
        
        elif isinstance(node, Print):
            value = self.eval(node.expr)
            print(value, file=self.stdout)
            return value
        
        else:
            raise RuntimeError(f"Unknown node type: {type(node).__name__}")
    
    def call_function(self, func, args):
        """Call a function with the given arguments."""
        if callable(func):
            # Python function (standard library)
            return func(*args)
        
        elif isinstance(func, FunctionDefinition):
            # Sprout function
            if len(args) != len(func.params):
                raise TypeError(f"Expected {len(func.params)} arguments, got {len(args)}")
            
            # Push a new environment for function execution
            self.push_env()
            
            # Bind parameters to arguments
            for param, arg in zip(func.params, args):
                self.define_var(param, arg)
            
            # Execute function body
            result = self.eval(func.body)
            
            # Check for return value
            if self.return_value is not None:
                result = self.return_value
                self.return_value = None
            
            # Pop the function environment
            self.pop_env()
            
            return result
        
        else:
            raise TypeError(f"Not a callable function: {func}")
    
    def run_in_thread(self, thread_id, func, args):
        """Run a function in a separate thread."""
        try:
            # Create a new interpreter for this thread
            thread_interpreter = Interpreter()
            
            # Copy functions and classes
            thread_interpreter.functions = self.functions.copy()
            thread_interpreter.classes = self.classes.copy()
            
            # Call the function
            thread_interpreter.call_function(func, args)
        
        except Exception as e:
            print(f"Error in thread {thread_id}: {str(e)}")
            traceback.print_exc()

# Main execution function with improved error handling
def run_file(filename):
    try:
        with open(filename, 'r') as f:
            code = f.read()
        
        tokens = tokenize(code, filename)
        parser = Parser(tokens, filename)
        block = parser.parse()
        
        interpreter = Interpreter()
        interpreter.current_file = filename
        interpreter.eval(block)
    
    except SproutError as e:
        print(str(e))
        return 1
    except Exception as e:
        print(f"Internal error: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

# Interactive REPL
def repl():
    interpreter = Interpreter()
    
    print("Sprout Programming Language REPL")
    print("Type 'exit' or 'quit' to exit, 'help' for help")
    
    while True:
        try:
            line = input("sprout> ")
            
            if line.strip().lower() in ('exit', 'quit'):
                break
            
            if line.strip().lower() == 'help':
                print("\nAvailable commands:")
                print("  exit, quit - Exit the REPL")
                print("  help       - Show this help message")
                print("\nAvailable standard library functions:")
                for name in sorted(interpreter.functions.keys()):
                    print(f"  {name}")
                print()
                continue
            
            tokens = tokenize(line)
            parser = Parser(tokens)
            block = parser.parse()
            
            result = interpreter.eval(block)
            if result is not None:
                print(result)
        
        except SproutError as e:
            print(str(e))
        except Exception as e:
            print(f"Internal error: {str(e)}")
            traceback.print_exc()

# Add command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sprout Programming Language")
    parser.add_argument('file', nargs='?', help="Source file to execute")
    parser.add_argument('-i', '--interactive', action='store_true', help="Start interactive REPL after executing file")
    parser.add_argument('-t', '--test', action='store_true', help="Run tests in the file")
    
    args = parser.parse_args()
    
    exit_code = 0
    
    if args.file:
        exit_code = run_file(args.file)
    
    if args.interactive or not args.file:
        repl()
    
    sys.exit(exit_code)