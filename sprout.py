import re
import threading
import queue
import os
import time

# Token types
TOKEN_TYPES = [
    ("NUMBER", r"\d+"),
    ("IDENT", r"[a-zA-Z_]\w*"),
    ("ASSIGN", r"="),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("TIMES", r"\*"),
    ("DIVIDE", r"/"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("GT", r">") ,
    ("LT", r"<"),
    ("EQ", r"=="),
    ("NEQ", r"!="),
    ("COMMA", r","),
    ("DOT", r"\."),
    ("NEWLINE", r"\n"),
    ("SKIP", r"[ \t]+"),
    ("MISMATCH", r".")
]

KEYWORDS = {"let", "print", "if", "else", "while", "fn", "return", "spawn", "join", "lock", "channel", "send", "receive", "import", "public", "private", "as", "test", "assert", "before", "after"}

# AST Node for Import
class Import:
    def __init__(self, module_name, alias=None):
        self.module_name = module_name
        self.alias = alias or module_name

# AST Node for Test
class Test:
    def __init__(self, name, before, body, after):
        self.name = name
        self.before = before
        self.body = body
        self.after = after

# AST Node for Assert
class Assert:
    def __init__(self, expr):
        self.expr = expr

# Parser with before/after hooks and test support
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)

    def advance(self):
        self.pos += 1

    def match(self, expected):
        token = self.peek()
        if token[0] == expected:
            self.advance()
            return token
        else:
            raise SyntaxError(f"Expected {expected}, got {token[0]}")

    def parse(self):
        statements = []
        while self.pos < len(self.tokens):
            statements.append(self.statement())
        return Block(statements)

    def statement(self):
        tok = self.peek()
        if tok[0] == "IMPORT":
            return self.import_stmt()
        elif tok[0] == "PUBLIC" or tok[0] == "PRIVATE":
            visibility = self.match(tok[0])[1]
            fn = self.statement()
            fn.visibility = visibility
            return fn
        elif tok[0] == "TEST":
            return self.test_stmt()
        elif tok[0] == "ASSERT":
            return self.assert_stmt()
        else:
            raise SyntaxError(f"Unexpected token {tok}")

    def import_stmt(self):
        self.match("IMPORT")
        module_name = self.match("IDENT")[1]
        alias = module_name
        if self.peek()[0] == "AS":
            self.match("AS")
            alias = self.match("IDENT")[1]
        return Import(module_name, alias)

    def test_stmt(self):
        self.match("TEST")
        name = self.match("IDENT")[1]
        self.match("LBRACE")
        before = []
        body = []
        after = []
        target = body
        while self.peek()[0] != "RBRACE":
            if self.peek()[0] == "BEFORE":
                self.match("BEFORE")
                self.match("LBRACE")
                before = self.block_contents()
            elif self.peek()[0] == "AFTER":
                self.match("AFTER")
                self.match("LBRACE")
                after = self.block_contents()
            else:
                target.append(self.statement())
        self.match("RBRACE")
        return Test(name, Block(before), Block(body), Block(after))

    def block_contents(self):
        body = []
        while self.peek()[0] != "RBRACE":
            body.append(self.statement())
        self.match("RBRACE")
        return body

    def assert_stmt(self):
        self.match("ASSERT")
        expr = self.expression()
        return Assert(expr)

    def expression(self):
        tok = self.peek()
        if tok[0] == "NUMBER":
            return Number(self.match("NUMBER")[1])
        elif tok[0] == "IDENT":
            return Variable(self.match("IDENT")[1])
        else:
            raise SyntaxError("Invalid expression")

# Interpreter with before/after hooks
class Interpreter:
    def __init__(self):
        self.env = {}
        self.functions = {}
        self.return_value = None
        self.threads = []
        self.locks = {}
        self.channels = {}
        self.modules = {}

    def eval(self, node):
        if isinstance(node, Import):
            filename = f"{node.module_name}.sprout"
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Module '{node.module_name}' not found")
            with open(filename) as f:
                code = f.read()
            tokens = tokenize(code)
            parser = Parser(tokens)
            block = parser.parse()
            module_env = {}
            interpreter = Interpreter()
            interpreter.eval(block)
            for name, fn in interpreter.functions.items():
                if getattr(fn, 'visibility', 'private') == 'public':
                    module_env[name] = fn
            self.modules[node.alias] = module_env
        elif isinstance(node, FunctionCall):
            parts = node.name.split(".")
            if len(parts) == 2:
                mod, fn = parts
                func = self.modules.get(mod, {}).get(fn)
            else:
                func = self.functions.get(node.name)
            if not func:
                raise NameError(f"Undefined function '{node.name}'")
            saved_env = self.env.copy()
            self.env = {}
            for param, arg in zip(func.params, node.args):
                self.env[param] = self.eval(arg)
            self.eval(func.body)
            result = self.return_value
            self.return_value = None
            self.env = saved_env
            return result
        elif isinstance(node, Test):
            print(f"Running test: {node.name}")
            start = time.time()
            try:
                self.eval(node.before)
                self.eval(node.body)
                self.eval(node.after)
                duration = (time.time() - start) * 1000
                print(f"✅ Passed: {node.name} ({duration:.2f}ms)")
            except Exception as e:
                print(f"❌ Failed: {node.name} — {e}")
        elif isinstance(node, Assert):
            if not self.eval(node.expr):
                raise AssertionError("Assertion failed")
        elif isinstance(node, Variable):
            return self.env.get(node.name, 0)
        elif isinstance(node, Number):
            return node.value