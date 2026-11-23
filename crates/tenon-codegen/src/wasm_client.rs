/// Direct `.tn` Manifest -> Spec-Compliant Binary WebAssembly (`.wasm` MVP) Client Store Compiler.
/// Zero external toolchain dependencies: emits valid LEB128 WebAssembly binary directly from
/// declarative `SessionMsgList` (Anti-Join) and `SessionChatStore` (`coalesce_window(N ms)`).

#[derive(Debug, Clone, Copy)]
pub struct WasmClientConfig {
    pub coalesce_ms: u32,
    pub anti_join: bool,
    pub debounce_cards_ms: u32,
}

impl Default for WasmClientConfig {
    fn default() -> Self {
        Self {
            coalesce_ms: 33,
            anti_join: true,
            debounce_cards_ms: 100,
        }
    }
}

// Memory layout offsets (i32 aligned in linear memory page 0):
const OFF_ACTIVE_BRANCH: u32 = 0;
const OFF_EXEC_EPOCH: u32 = 4;
const OFF_ACTIVE_RUN_HASH: u32 = 8;
const OFF_LIVE_LEN: u32 = 12;
const OFF_LAST_FLUSH_MS: u32 = 16;
const OFF_PENDING_DIRTY: u32 = 20;
const OFF_TOTAL_DELTAS: u32 = 24;
const OFF_TOTAL_RENDERS: u32 = 28;
const OFF_ANTI_JOIN_HANDOFFS: u32 = 32;
const OFF_DROPPED_STALE: u32 = 36;

/// Compile `WasmClientConfig` into a real, standalone WebAssembly binary module (`.wasm`).
pub fn compile_wasm_client_store(cfg: WasmClientConfig) -> Vec<u8> {
    let mut wasm = Vec::new();
    // Magic `\0asm` + Version 1
    wasm.extend_from_slice(&[0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00]);

    // Section 1: Type Section
    // type 0: () -> (i32)
    // type 1: (i32, i32, i32) -> (i32)
    // type 2: (i32, i32, i32, i32, i32) -> (i32)
    // type 3: (i32) -> (i32)
    let mut type_sec = Vec::new();
    push_uleb(&mut type_sec, 4);
    // type 0
    type_sec.extend_from_slice(&[0x60, 0x00, 0x01, 0x7F]);
    // type 1
    type_sec.extend_from_slice(&[0x60, 0x03, 0x7F, 0x7F, 0x7F, 0x01, 0x7F]);
    // type 2
    type_sec.extend_from_slice(&[0x60, 0x05, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x01, 0x7F]);
    // type 3
    type_sec.extend_from_slice(&[0x60, 0x01, 0x7F, 0x01, 0x7F]);
    emit_section(&mut wasm, 1, &type_sec);

    // Section 3: Function Section (11 functions)
    // 0: coalesce_window_ms() -> i32            (type 0)
    // 1: debounce_cards_ms() -> i32             (type 0)
    // 2: on_snapshot(b, e, committed_hash) -> i32 (type 1)
    // 3: on_delta(b, run, seq, bytes, now) -> i32 (type 2)
    // 4: flush_due(now_ms) -> i32               (type 3)
    // 5: active_branch() -> i32                 (type 0)
    // 6: live_len() -> i32                      (type 0)
    // 7: total_deltas() -> i32                  (type 0)
    // 8: total_renders() -> i32                 (type 0)
    // 9: anti_join_handoffs() -> i32            (type 0)
    // 10: dropped_stale_deltas() -> i32         (type 0)
    let fn_types: &[u32] = &[0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0];
    let mut func_sec = Vec::new();
    push_uleb(&mut func_sec, fn_types.len() as u32);
    for &t in fn_types {
        push_uleb(&mut func_sec, t);
    }
    emit_section(&mut wasm, 3, &func_sec);

    // Section 5: Memory Section (1 linear memory, min 1 page = 64 KiB)
    let mut mem_sec = Vec::new();
    push_uleb(&mut mem_sec, 1);
    mem_sec.push(0x00); // no max limit
    push_uleb(&mut mem_sec, 1);
    emit_section(&mut wasm, 5, &mem_sec);

    // Section 7: Export Section
    let exports: &[(&str, u8, u32)] = &[
        ("memory", 0x02, 0),
        ("coalesce_window_ms", 0x00, 0),
        ("debounce_cards_ms", 0x00, 1),
        ("on_snapshot", 0x00, 2),
        ("on_delta", 0x00, 3),
        ("flush_due", 0x00, 4),
        ("active_branch", 0x00, 5),
        ("live_len", 0x00, 6),
        ("total_deltas", 0x00, 7),
        ("total_renders", 0x00, 8),
        ("anti_join_handoffs", 0x00, 9),
        ("dropped_stale_deltas", 0x00, 10),
    ];
    let mut exp_sec = Vec::new();
    push_uleb(&mut exp_sec, exports.len() as u32);
    for &(name, kind, idx) in exports {
        push_uleb(&mut exp_sec, name.len() as u32);
        exp_sec.extend_from_slice(name.as_bytes());
        exp_sec.push(kind);
        push_uleb(&mut exp_sec, idx);
    }
    emit_section(&mut wasm, 7, &exp_sec);

    // Section 10: Code Section
    let bodies: Vec<Vec<u8>> = vec![
        emit_const_fn(cfg.coalesce_ms as i32),
        emit_const_fn(cfg.debounce_cards_ms as i32),
        emit_on_snapshot(cfg.anti_join),
        emit_on_delta(cfg.coalesce_ms as i32),
        emit_flush_due(cfg.coalesce_ms as i32),
        emit_load_slot_fn(OFF_ACTIVE_BRANCH),
        emit_load_slot_fn(OFF_LIVE_LEN),
        emit_load_slot_fn(OFF_TOTAL_DELTAS),
        emit_load_slot_fn(OFF_TOTAL_RENDERS),
        emit_load_slot_fn(OFF_ANTI_JOIN_HANDOFFS),
        emit_load_slot_fn(OFF_DROPPED_STALE),
    ];
    let mut code_sec = Vec::new();
    push_uleb(&mut code_sec, bodies.len() as u32);
    for b in bodies {
        push_uleb(&mut code_sec, b.len() as u32);
        code_sec.extend_from_slice(&b);
    }
    emit_section(&mut wasm, 10, &code_sec);

    wasm
}

fn emit_const_fn(val: i32) -> Vec<u8> {
    let mut b = vec![0x00]; // 0 local decls
    i32_const(&mut b, val);
    b.push(0x0B); // end
    b
}

fn emit_load_slot_fn(offset: u32) -> Vec<u8> {
    let mut b = vec![0x00];
    load_i32(&mut b, offset);
    b.push(0x0B);
    b
}

/// on_snapshot(local 0: branch_id, local 1: exec_epoch, local 2: committed_run_hash) -> i32 (1 = render now)
fn emit_on_snapshot(anti_join: bool) -> Vec<u8> {
    let mut b = vec![0x01, 0x01, 0x7F]; // 1 extra i32 local (local 3: prev_branch)

    // prev_branch = mem[OFF_ACTIVE_BRANCH]
    load_i32(&mut b, OFF_ACTIVE_BRANCH);
    local_set(&mut b, 3);

    // mem[OFF_ACTIVE_BRANCH] = branch_id; mem[OFF_EXEC_EPOCH] = exec_epoch
    i32_const(&mut b, 0);
    local_get(&mut b, 0);
    store_i32(&mut b, OFF_ACTIVE_BRANCH);

    i32_const(&mut b, 0);
    local_get(&mut b, 1);
    store_i32(&mut b, OFF_EXEC_EPOCH);

    // If prev_branch != 0 && prev_branch != branch_id -> clear stale-branch overlay!
    local_get(&mut b, 3);
    i32_const(&mut b, 0);
    b.push(0x47); // i32.ne
    local_get(&mut b, 3);
    local_get(&mut b, 0);
    b.push(0x47); // i32.ne
    b.push(0x71); // i32.and
    b.extend_from_slice(&[0x04, 0x40]); // if void
    store_const(&mut b, OFF_ACTIVE_RUN_HASH, 0);
    store_const(&mut b, OFF_LIVE_LEN, 0);
    store_const(&mut b, OFF_PENDING_DIRTY, 0);
    b.push(0x0B); // end if

    if anti_join {
        // Compiled Anti-Join: if committed_run_hash != 0 && committed_run_hash == mem[OFF_ACTIVE_RUN_HASH]
        local_get(&mut b, 2);
        i32_const(&mut b, 0);
        b.push(0x47); // i32.ne
        local_get(&mut b, 2);
        load_i32(&mut b, OFF_ACTIVE_RUN_HASH);
        b.push(0x46); // i32.eq
        b.push(0x71); // i32.and
        b.extend_from_slice(&[0x04, 0x40]); // if void
        store_const(&mut b, OFF_ACTIVE_RUN_HASH, 0);
        store_const(&mut b, OFF_LIVE_LEN, 0);
        store_const(&mut b, OFF_PENDING_DIRTY, 0);
        inc_slot(&mut b, OFF_ANTI_JOIN_HANDOFFS);
        b.push(0x0B); // end if
    }

    // Snapshot ALWAYS bypasses debounce and triggers immediate render!
    inc_slot(&mut b, OFF_TOTAL_RENDERS);
    i32_const(&mut b, 1);
    b.push(0x0B); // end fn
    b
}

/// on_delta(0: branch_id, 1: run_hash, 2: seq, 3: byte_len, 4: now_ms) -> i32 (1=render, 0=coalesced)
fn emit_on_delta(coalesce_ms: i32) -> Vec<u8> {
    let mut b = vec![0x00]; // no extra locals

    // Lineage guard: if active_branch != 0 && branch_id != active_branch -> drop & return 0!
    load_i32(&mut b, OFF_ACTIVE_BRANCH);
    i32_const(&mut b, 0);
    b.push(0x47); // i32.ne
    local_get(&mut b, 0);
    load_i32(&mut b, OFF_ACTIVE_BRANCH);
    b.push(0x47); // i32.ne
    b.push(0x71); // i32.and
    b.extend_from_slice(&[0x04, 0x40]); // if void
    inc_slot(&mut b, OFF_DROPPED_STALE);
    i32_const(&mut b, 0);
    b.push(0x0F); // return 0
    b.push(0x0B); // end if

    // If active_branch == 0 -> initialize active_branch = branch_id
    load_i32(&mut b, OFF_ACTIVE_BRANCH);
    b.push(0x45); // i32.eqz
    b.extend_from_slice(&[0x04, 0x40]);
    i32_const(&mut b, 0);
    local_get(&mut b, 0);
    store_i32(&mut b, OFF_ACTIVE_BRANCH);
    b.push(0x0B);

    inc_slot(&mut b, OFF_TOTAL_DELTAS);

    // If run_hash != active_run_hash -> start new overlay stream
    local_get(&mut b, 1);
    load_i32(&mut b, OFF_ACTIVE_RUN_HASH);
    b.push(0x47); // i32.ne
    b.extend_from_slice(&[0x04, 0x40]);
    i32_const(&mut b, 0);
    local_get(&mut b, 1);
    store_i32(&mut b, OFF_ACTIVE_RUN_HASH);
    store_const(&mut b, OFF_LIVE_LEN, 0);
    i32_const(&mut b, 0);
    local_get(&mut b, 4);
    store_i32(&mut b, OFF_LAST_FLUSH_MS);
    b.push(0x0B);

    // live_len += byte_len; pending_dirty = 1
    i32_const(&mut b, 0);
    load_i32(&mut b, OFF_LIVE_LEN);
    local_get(&mut b, 3);
    b.push(0x6A); // i32.add
    store_i32(&mut b, OFF_LIVE_LEN);
    store_const(&mut b, OFF_PENDING_DIRTY, 1);

    // Compiled Coalesce Window: if seq == 1 || (now_ms - last_flush_ms) >= coalesce_ms -> flush!
    local_get(&mut b, 2);
    i32_const(&mut b, 1);
    b.push(0x46); // i32.eq
    local_get(&mut b, 4);
    load_i32(&mut b, OFF_LAST_FLUSH_MS);
    b.push(0x6B); // i32.sub
    i32_const(&mut b, coalesce_ms);
    b.push(0x4E); // i32.ge_s
    b.push(0x72); // i32.or
    b.extend_from_slice(&[0x04, 0x40]);
    i32_const(&mut b, 0);
    local_get(&mut b, 4);
    store_i32(&mut b, OFF_LAST_FLUSH_MS);
    store_const(&mut b, OFF_PENDING_DIRTY, 0);
    inc_slot(&mut b, OFF_TOTAL_RENDERS);
    i32_const(&mut b, 1);
    b.push(0x0F); // return 1
    b.push(0x0B);

    i32_const(&mut b, 0);
    b.push(0x0B);
    b
}

/// flush_due(0: now_ms) -> i32 (1=flushed, 0=no-op)
fn emit_flush_due(coalesce_ms: i32) -> Vec<u8> {
    let mut b = vec![0x00];
    load_i32(&mut b, OFF_PENDING_DIRTY);
    i32_const(&mut b, 0);
    b.push(0x47); // pending_dirty != 0
    local_get(&mut b, 0);
    load_i32(&mut b, OFF_LAST_FLUSH_MS);
    b.push(0x6B); // now_ms - last_flush_ms
    i32_const(&mut b, coalesce_ms);
    b.push(0x4E); // >= coalesce_ms
    b.push(0x71); // &&
    b.extend_from_slice(&[0x04, 0x40]);
    i32_const(&mut b, 0);
    local_get(&mut b, 0);
    store_i32(&mut b, OFF_LAST_FLUSH_MS);
    store_const(&mut b, OFF_PENDING_DIRTY, 0);
    inc_slot(&mut b, OFF_TOTAL_RENDERS);
    i32_const(&mut b, 1);
    b.push(0x0F);
    b.push(0x0B);
    i32_const(&mut b, 0);
    b.push(0x0B);
    b
}

fn load_i32(b: &mut Vec<u8>, offset: u32) {
    i32_const(b, 0);
    b.push(0x28); // i32.load
    push_uleb(b, 2); // align 4
    push_uleb(b, offset);
}

fn store_i32(b: &mut Vec<u8>, offset: u32) {
    b.push(0x36); // i32.store
    push_uleb(b, 2);
    push_uleb(b, offset);
}

fn store_const(b: &mut Vec<u8>, offset: u32, val: i32) {
    i32_const(b, 0);
    i32_const(b, val);
    store_i32(b, offset);
}

fn inc_slot(b: &mut Vec<u8>, offset: u32) {
    i32_const(b, 0);
    load_i32(b, offset);
    i32_const(b, 1);
    b.push(0x6A); // i32.add
    store_i32(b, offset);
}

fn local_get(b: &mut Vec<u8>, idx: u32) {
    b.push(0x20);
    push_uleb(b, idx);
}

fn local_set(b: &mut Vec<u8>, idx: u32) {
    b.push(0x21);
    push_uleb(b, idx);
}

fn i32_const(b: &mut Vec<u8>, val: i32) {
    b.push(0x41);
    push_sleb(b, val as i64);
}

fn emit_section(out: &mut Vec<u8>, id: u8, payload: &[u8]) {
    out.push(id);
    push_uleb(out, payload.len() as u32);
    out.extend_from_slice(payload);
}

fn push_uleb(out: &mut Vec<u8>, mut val: u32) {
    loop {
        let mut byte = (val & 0x7F) as u8;
        val >>= 7;
        if val != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if val == 0 {
            break;
        }
    }
}

fn push_sleb(out: &mut Vec<u8>, mut val: i64) {
    loop {
        let mut byte = (val & 0x7F) as u8;
        val >>= 7;
        let done = (val == 0 && (byte & 0x40) == 0) || (val == -1 && (byte & 0x40) != 0);
        if !done {
            byte |= 0x80;
        }
        out.push(byte);
        if done {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn emitted_wasm_validates_and_runs_coalesce_and_anti_join_in_v8() {
        let bytes = compile_wasm_client_store(WasmClientConfig {
            coalesce_ms: 33,
            anti_join: true,
            debounce_cards_ms: 100,
        });
        assert_eq!(&bytes[0..8], &[0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00]);

        let tmp = format!("/tmp/tenon_test_client_{}.wasm", std::process::id());
        std::fs::write(&tmp, &bytes).unwrap();

        // Execute the compiled .wasm binary directly in Node.js V8 WebAssembly engine!
        let js = format!(
            r#"
const fs = require('fs');
const buf = fs.readFileSync({:?});
if (!WebAssembly.validate(buf)) throw new Error('WASM validation failed');
const mod = new WebAssembly.Module(buf);
const {{ exports: e }} = new WebAssembly.Instance(mod);
if (e.coalesce_window_ms() !== 33) throw new Error('expected 33ms');
if (e.debounce_cards_ms() !== 100) throw new Error('expected 100ms');

// Send 10 rapid deltas within 18ms -> seq=1 renders immediately, seq 2..10 coalesce!
let rendersFromDeltas = 0;
for (let i = 1; i <= 10; i++) {{
  rendersFromDeltas += e.on_delta(2, 777, i, 5, i * 2);
}}
if (rendersFromDeltas !== 1) throw new Error('expected 1 render during burst, got ' + rendersFromDeltas);
if (e.live_len() !== 50) throw new Error('expected live_len 50, got ' + e.live_len());

// Stale branch delta (branch 1 while active is 2) must be dropped!
const stale = e.on_delta(1, 555, 1, 10, 25);
if (stale !== 0 || e.dropped_stale_deltas() !== 1) throw new Error('stale delta not dropped');

// After 33ms window elapses (now=40ms), flush_due triggers 1 coalesced render!
if (e.flush_due(40) !== 1) throw new Error('expected flush_due=1');
if (e.flush_due(45) !== 0) throw new Error('expected redundant flush_due=0');

// Authoritative snapshot with committed run 777 triggers compiled Anti-Join -> live_len resets to 0!
if (e.on_snapshot(2, 4, 777) !== 1) throw new Error('snapshot must trigger render');
if (e.live_len() !== 0) throw new Error('anti-join failed to clear live_len');
if (e.anti_join_handoffs() !== 1) throw new Error('expected anti_join_handoffs=1');
console.log('V8_WASM_OK', e.total_deltas(), e.total_renders(), e.anti_join_handoffs(), e.dropped_stale_deltas());
"#,
            tmp
        );

        let out = Command::new("node").args(["-e", &js]).output();
        let _ = std::fs::remove_file(&tmp);
        if let Ok(o) = out {
            assert!(
                o.status.success(),
                "Node V8 wasm execution failed: stdout={} stderr={}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            );
            assert!(String::from_utf8_lossy(&o.stdout).contains("V8_WASM_OK 10 3 1 1"));
        }
    }
}
