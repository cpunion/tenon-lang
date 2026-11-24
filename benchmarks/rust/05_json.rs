// Benchmark 05 (Rust std-only): 5,000x Nested JSON Parse + Field Access + Rebuild + Stringify

enum JVal {
    Null,
    Bool(bool),
    Int(i64),
    Str(String),
    Arr(Vec<JVal>),
    Obj(Vec<(String, JVal)>),
}

impl JVal {
    fn get(&self, key: &str) -> Option<&JVal> {
        if let JVal::Obj(entries) = self {
            for (k, v) in entries {
                if k == key {
                    return Some(v);
                }
            }
        }
        None
    }

    fn write_to(&self, out: &mut String) {
        match self {
            JVal::Null => out.push_str("null"),
            JVal::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            JVal::Int(n) => out.push_str(&n.to_string()),
            JVal::Str(s) => {
                out.push('"');
                for ch in s.bytes() {
                    match ch {
                        b'"' => out.push_str("\\\""),
                        b'\\' => out.push_str("\\\\"),
                        b'\n' => out.push_str("\\n"),
                        b'\r' => out.push_str("\\r"),
                        b'\t' => out.push_str("\\t"),
                        _ => out.push(ch as char),
                    }
                }
                out.push('"');
            }
            JVal::Arr(items) => {
                out.push('[');
                for (i, it) in items.iter().enumerate() {
                    if i > 0 { out.push(','); }
                    it.write_to(out);
                }
                out.push(']');
            }
            JVal::Obj(entries) => {
                out.push('{');
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 { out.push(','); }
                    out.push('"');
                    out.push_str(k);
                    out.push_str("\":");
                    v.write_to(out);
                }
                out.push('}');
            }
        }
    }
}

fn skip_ws(b: &[u8], pos: &mut usize) {
    while *pos < b.len() && matches!(b[*pos], b' ' | b'\t' | b'\n' | b'\r') {
        *pos += 1;
    }
}

fn hex_val(c: u8) -> u32 {
    match c {
        b'0'..=b'9' => (c - b'0') as u32,
        b'a'..=b'f' => 10 + (c - b'a') as u32,
        b'A'..=b'F' => 10 + (c - b'A') as u32,
        _ => 0,
    }
}

fn parse_str(b: &[u8], pos: &mut usize) -> String {
    *pos += 1;
    let mut out = String::new();
    while *pos < b.len() {
        let c = b[*pos];
        *pos += 1;
        if c == b'"' {
            return out;
        }
        if c == b'\\' && *pos < b.len() {
            let e = b[*pos];
            *pos += 1;
            match e {
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'"' | b'\\' | b'/' => out.push(e as char),
                b'u' if *pos + 4 <= b.len() => {
                    let cp = (hex_val(b[*pos]) << 12)
                        | (hex_val(b[*pos + 1]) << 8)
                        | (hex_val(b[*pos + 2]) << 4)
                        | hex_val(b[*pos + 3]);
                    *pos += 4;
                    if let Some(ch) = char::from_u32(cp) {
                        out.push(ch);
                    }
                }
                _ => out.push(e as char),
            }
        } else {
            out.push(c as char);
        }
    }
    out
}

fn parse_val(b: &[u8], pos: &mut usize) -> JVal {
    skip_ws(b, pos);
    if *pos >= b.len() {
        return JVal::Null;
    }
    match b[*pos] {
        b'"' => JVal::Str(parse_str(b, pos)),
        b'[' => {
            *pos += 1;
            let mut items = Vec::new();
            skip_ws(b, pos);
            if *pos < b.len() && b[*pos] == b']' {
                *pos += 1;
                return JVal::Arr(items);
            }
            while *pos < b.len() {
                items.push(parse_val(b, pos));
                skip_ws(b, pos);
                if *pos < b.len() && b[*pos] == b',' {
                    *pos += 1;
                    continue;
                }
                if *pos < b.len() && b[*pos] == b']' {
                    *pos += 1;
                }
                break;
            }
            JVal::Arr(items)
        }
        b'{' => {
            *pos += 1;
            let mut entries = Vec::new();
            skip_ws(b, pos);
            if *pos < b.len() && b[*pos] == b'}' {
                *pos += 1;
                return JVal::Obj(entries);
            }
            while *pos < b.len() {
                skip_ws(b, pos);
                let k = parse_str(b, pos);
                skip_ws(b, pos);
                if *pos < b.len() && b[*pos] == b':' {
                    *pos += 1;
                }
                let v = parse_val(b, pos);
                entries.push((k, v));
                skip_ws(b, pos);
                if *pos < b.len() && b[*pos] == b',' {
                    *pos += 1;
                    continue;
                }
                if *pos < b.len() && b[*pos] == b'}' {
                    *pos += 1;
                }
                break;
            }
            JVal::Obj(entries)
        }
        b't' => { *pos += 4; JVal::Bool(true) }
        b'f' => { *pos += 5; JVal::Bool(false) }
        b'n' => { *pos += 4; JVal::Null }
        _ => {
            let neg = if b[*pos] == b'-' { *pos += 1; true } else { false };
            let mut acc: i64 = 0;
            while *pos < b.len() && b[*pos].is_ascii_digit() {
                acc = acc * 10 + (b[*pos] - b'0') as i64;
                *pos += 1;
            }
            JVal::Int(if neg { -acc } else { acc })
        }
    }
}

fn main() {
    let sample = br#"{"type":"response.output_item.done","ok":true,"seq":42,"nums":[10,20,30,40,50],"item":{"id":"fc_123","name":"write_file","arguments":"{\"path\":\"src/main.tn\",\"content\":\"line1\\nline2 \\u4f60\\u597d\"}"}}"#;
    let mut checksum: i64 = 0;
    for i in 0..5000i64 {
        let mut pos = 0;
        let ev = parse_val(sample, &mut pos);
        let t = match ev.get("type") {
            Some(JVal::Str(s)) => s.clone(),
            _ => String::new(),
        };
        let seq = match ev.get("seq") {
            Some(JVal::Int(n)) => *n,
            _ => 0,
        };
        let num2 = match ev.get("nums") {
            Some(JVal::Arr(arr)) => match arr.get(2) {
                Some(JVal::Int(n)) => *n,
                _ => 0,
            },
            _ => 0,
        };
        let built = JVal::Obj(vec![
            ("type".to_string(), JVal::Str(t)),
            ("seq".to_string(), JVal::Int(seq + i)),
            ("pick".to_string(), JVal::Int(num2)),
            ("ok".to_string(), JVal::Bool(true)),
        ]);
        let mut out = String::with_capacity(96);
        built.write_to(&mut out);
        checksum += out.len() as i64 + num2;
    }
    println!("{}", checksum);
}
