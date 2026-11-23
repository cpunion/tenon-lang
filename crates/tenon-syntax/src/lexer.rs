use crate::span::{ParseError, Span};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    // Literals & Identifiers
    Ident(String),
    Int(i64),
    Str(String),

    // Keywords
    Import,
    Pub,
    Type,
    Distinct,
    Record,
    Enum,
    Fn,
    Pure,
    Async,
    Await,
    Test,
    Inout,
    Comptime,
    Where,
    Requires,
    Ensures,
    Invariant,
    Decreases,
    Let,
    Var,
    If,
    Else,
    While,
    Break,
    Continue,
    Match,
    Return,
    Assert,
    Mutate,
    Old,
    Forall,
    True,
    False,
    Move,
    Evolution,
    Policy,
    Archive,
    Retain,
    AllowLoss,
    LockLattice,
    Protocol,
    Satisfies,
    Laws,
    FaultModel,
    Stream,
    Reactive,
    View,
    From,
    Bounds,
    Subscriber,
    Bind,
    Pacing,
    LocalStorage,
    Output,

    // Punctuation & Operators
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Colon,
    Dot,
    At,
    Eq,
    EqEq,
    BangEq,
    Bang,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    PlusPlus,
    PlusEq,
    Minus,
    MinusEq,
    Star,
    Slash,
    Percent,
    Amp,
    AmpAmp,
    Pipe,
    PipePipe,
    FatArrow,
    Bidirectional, // <->
    Newline,
    Eof,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

pub fn lex(src: &str) -> Result<Vec<Token>, ParseError> {
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut line = 1;
    let mut col = 1;
    let mut tokens = Vec::new();

    while i < bytes.len() {
        let ch = bytes[i] as char;
        match ch {
            ' ' | '\t' | '\r' => {
                i += 1;
                col += 1;
            }
            '\n' => {
                let span = Span::new(i, i + 1, line, col);
                tokens.push(Token {
                    kind: TokenKind::Newline,
                    span,
                });
                i += 1;
                line += 1;
                col = 1;
            }
            '/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                    col += 1;
                }
            }
            '/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                let start = i;
                let sline = line;
                let scol = col;
                i += 2;
                col += 2;
                let mut closed = false;
                while i + 1 < bytes.len() {
                    if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        i += 2;
                        col += 2;
                        closed = true;
                        break;
                    }
                    if bytes[i] == b'\n' {
                        line += 1;
                        col = 1;
                    } else {
                        col += 1;
                    }
                    i += 1;
                }
                if !closed {
                    return Err(ParseError {
                        message: "unterminated block comment".into(),
                        span: Span::new(start, i, sline, scol),
                    });
                }
            }
            '"' => {
                let start = i;
                let sline = line;
                let scol = col;
                i += 1;
                col += 1;
                let mut raw: Vec<u8> = Vec::new();
                let mut closed = false;
                while i < bytes.len() {
                    let b = bytes[i];
                    if b == b'"' {
                        i += 1;
                        col += 1;
                        closed = true;
                        break;
                    }
                    if b == b'\\' && i + 1 < bytes.len() {
                        i += 1;
                        col += 1;
                        match bytes[i] {
                            b'n' => raw.push(b'\n'),
                            b't' => raw.push(b'\t'),
                            b'r' => raw.push(b'\r'),
                            b'\\' => raw.push(b'\\'),
                            b'"' => raw.push(b'"'),
                            other => raw.push(other),
                        }
                        i += 1;
                        col += 1;
                    } else {
                        if b == b'\n' {
                            line += 1;
                            col = 1;
                        } else {
                            col += 1;
                        }
                        raw.push(b);
                        i += 1;
                    }
                }
                if !closed {
                    return Err(ParseError {
                        message: "unterminated string literal".into(),
                        span: Span::new(start, i, sline, scol),
                    });
                }
                let s = String::from_utf8_lossy(&raw).into_owned();
                tokens.push(Token {
                    kind: TokenKind::Str(s),
                    span: Span::new(start, i, sline, scol),
                });
            }
            '0'..='9' => {
                let start = i;
                let scol = col;
                let mut num_str = String::new();
                while i < bytes.len() {
                    let c = bytes[i] as char;
                    if c.is_ascii_digit() {
                        num_str.push(c);
                        i += 1;
                        col += 1;
                    } else if c == '_' {
                        i += 1;
                        col += 1;
                    } else {
                        break;
                    }
                }
                let val: i64 = num_str.parse().map_err(|_| ParseError {
                    message: format!("invalid integer literal `{}`", num_str),
                    span: Span::new(start, i, line, scol),
                })?;
                tokens.push(Token {
                    kind: TokenKind::Int(val),
                    span: Span::new(start, i, line, scol),
                });
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let start = i;
                let scol = col;
                while i < bytes.len() {
                    let c = bytes[i] as char;
                    if c.is_ascii_alphanumeric() || c == '_' {
                        i += 1;
                        col += 1;
                    } else {
                        break;
                    }
                }
                let word = &src[start..i];
                let kind = match word {
                    "import" => TokenKind::Import,
                    "pub" => TokenKind::Pub,
                    "type" => TokenKind::Type,
                    "distinct" => TokenKind::Distinct,
                    "record" => TokenKind::Record,
                    "enum" => TokenKind::Enum,
                    "fn" => TokenKind::Fn,
                    "pure" => TokenKind::Pure,
                    "async" => TokenKind::Async,
                    "await" => TokenKind::Await,
                    "test" => TokenKind::Test,
                    "inout" => TokenKind::Inout,
                    "comptime" => TokenKind::Comptime,
                    "where" => TokenKind::Where,
                    "requires" => TokenKind::Requires,
                    "ensures" => TokenKind::Ensures,
                    "invariant" => TokenKind::Invariant,
                    "decreases" => TokenKind::Decreases,
                    "let" => TokenKind::Let,
                    "var" => TokenKind::Var,
                    "if" => TokenKind::If,
                    "else" => TokenKind::Else,
                    "while" => TokenKind::While,
                    "break" => TokenKind::Break,
                    "continue" => TokenKind::Continue,
                    "match" => TokenKind::Match,
                    "return" => TokenKind::Return,
                    "assert" => TokenKind::Assert,
                    "mutate" => TokenKind::Mutate,
                    "old" => TokenKind::Old,
                    "forall" => TokenKind::Forall,
                    "true" => TokenKind::True,
                    "false" => TokenKind::False,
                    "move" => TokenKind::Move,
                    "evolution" => TokenKind::Evolution,
                    "policy" => TokenKind::Policy,
                    "archive" => TokenKind::Archive,
                    "retain" => TokenKind::Retain,
                    "allow_loss" => TokenKind::AllowLoss,
                    "lock_lattice" => TokenKind::LockLattice,
                    "protocol" => TokenKind::Protocol,
                    "satisfies" => TokenKind::Satisfies,
                    "laws" => TokenKind::Laws,
                    "fault_model" => TokenKind::FaultModel,
                    "stream" => TokenKind::Stream,
                    "reactive" => TokenKind::Reactive,
                    "view" => TokenKind::View,
                    "from" => TokenKind::From,
                    "bounds" => TokenKind::Bounds,
                    "subscriber" => TokenKind::Subscriber,
                    "bind" => TokenKind::Bind,
                    "pacing" => TokenKind::Pacing,
                    "local_storage" => TokenKind::LocalStorage,
                    "output" => TokenKind::Output,
                    _ => TokenKind::Ident(word.to_string()),
                };
                tokens.push(Token {
                    kind,
                    span: Span::new(start, i, line, scol),
                });
            }
            _ => {
                let start = i;
                let scol = col;
                let rem = &src[i..];
                let (kind, len) = if rem.starts_with("<->") {
                    (TokenKind::Bidirectional, 3)
                } else if rem.starts_with("==") {
                    (TokenKind::EqEq, 2)
                } else if rem.starts_with("!=") {
                    (TokenKind::BangEq, 2)
                } else if rem.starts_with("<=") {
                    (TokenKind::Le, 2)
                } else if rem.starts_with(">=") {
                    (TokenKind::Ge, 2)
                } else if rem.starts_with("&&") {
                    (TokenKind::AmpAmp, 2)
                } else if rem.starts_with("||") {
                    (TokenKind::PipePipe, 2)
                } else if rem.starts_with("=>") {
                    (TokenKind::FatArrow, 2)
                } else if rem.starts_with("++") {
                    (TokenKind::PlusPlus, 2)
                } else if rem.starts_with("+=") {
                    (TokenKind::PlusEq, 2)
                } else if rem.starts_with("-=") {
                    (TokenKind::MinusEq, 2)
                } else {
                    match ch {
                        '(' => (TokenKind::LParen, 1),
                        ')' => (TokenKind::RParen, 1),
                        '{' => (TokenKind::LBrace, 1),
                        '}' => (TokenKind::RBrace, 1),
                        '[' => (TokenKind::LBracket, 1),
                        ']' => (TokenKind::RBracket, 1),
                        ',' => (TokenKind::Comma, 1),
                        ';' => (TokenKind::Semi, 1),
                        ':' => (TokenKind::Colon, 1),
                        '.' => (TokenKind::Dot, 1),
                        '@' => (TokenKind::At, 1),
                        '=' => (TokenKind::Eq, 1),
                        '!' => (TokenKind::Bang, 1),
                        '<' => (TokenKind::Lt, 1),
                        '>' => (TokenKind::Gt, 1),
                        '+' => (TokenKind::Plus, 1),
                        '-' => (TokenKind::Minus, 1),
                        '*' => (TokenKind::Star, 1),
                        '/' => (TokenKind::Slash, 1),
                        '%' => (TokenKind::Percent, 1),
                        '&' => (TokenKind::Amp, 1),
                        '|' => (TokenKind::Pipe, 1),
                        other => {
                            return Err(ParseError {
                                message: format!("unexpected character `{}`", other),
                                span: Span::new(start, start + 1, line, scol),
                            });
                        }
                    }
                };
                i += len;
                col += len;
                tokens.push(Token {
                    kind,
                    span: Span::new(start, i, line, scol),
                });
            }
        }
    }

    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span::new(i, i, line, col),
    });
    Ok(tokens)
}
