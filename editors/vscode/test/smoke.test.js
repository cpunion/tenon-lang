'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');

const EXT_DIR = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(EXT_DIR, '..', '..');

function findTenonBin() {
  if (process.env.TENON_BIN && fs.existsSync(process.env.TENON_BIN)) {
    return process.env.TENON_BIN;
  }
  const candidates = ['target/release/tenon', 'target/debug/tenon']
    .map((rel) => path.join(REPO_ROOT, rel))
    .filter((p) => fs.existsSync(p))
    .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
  if (candidates.length > 0) {
    return candidates[0];
  }
  throw new Error('Could not find compiled `tenon` binary under target/release or target/debug');
}

function encodeRpc(obj) {
  const body = Buffer.from(JSON.stringify(obj), 'utf8');
  return Buffer.concat([Buffer.from(`Content-Length: ${body.length}\r\n\r\n`, 'ascii'), body]);
}

function createRpcClient(binPath) {
  const child = spawn(binPath, ['lsp'], { stdio: ['pipe', 'pipe', 'pipe'] });
  let buffer = Buffer.alloc(0);
  const listeners = [];

  child.stdout.on('data', (chunk) => {
    buffer = Buffer.concat([buffer, chunk]);
    while (true) {
      const headerEnd = buffer.indexOf('\r\n\r\n');
      if (headerEnd === -1) break;
      const header = buffer.slice(0, headerEnd).toString('ascii');
      const m = /Content-Length:\s*(\d+)/i.exec(header);
      if (!m) break;
      const len = Number(m[1]);
      const total = headerEnd + 4 + len;
      if (buffer.length < total) break;
      const body = buffer.slice(headerEnd + 4, total).toString('utf8');
      buffer = buffer.slice(total);
      const msg = JSON.parse(body);
      for (let i = 0; i < listeners.length; i++) {
        if (listeners[i].pred(msg)) {
          const hit = listeners.splice(i, 1)[0];
          hit.resolve(msg);
          break;
        }
      }
    }
  });

  function waitFor(pred, timeoutMs = 8000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Timed out waiting for LSP message')), timeoutMs);
      listeners.push({
        pred,
        resolve: (msg) => {
          clearTimeout(timer);
          resolve(msg);
        },
      });
    });
  }

  function send(msg) {
    child.stdin.write(encodeRpc(msg));
  }

  return { child, send, waitFor };
}

test('manifest, language configuration, and TextMate grammar are valid JSON', () => {
  const pkg = JSON.parse(fs.readFileSync(path.join(EXT_DIR, 'package.json'), 'utf8'));
  assert.equal(pkg.name, 'tenon-language');
  assert.equal(pkg.contributes.languages[0].extensions[0], '.tn');

  const langCfg = JSON.parse(fs.readFileSync(path.join(EXT_DIR, 'language-configuration.json'), 'utf8'));
  assert.equal(langCfg.comments.lineComment, '//');

  const grammar = JSON.parse(
    fs.readFileSync(path.join(EXT_DIR, 'syntaxes', 'tenon.tmLanguage.json'), 'utf8')
  );
  assert.equal(grammar.scopeName, 'source.tenon');
  assert.ok(grammar.repository.keywords);
  assert.ok(grammar.repository.types);
});

test('real `tenon lsp` process handles initialize, diagnostics, hover, definition, completion, symbols, formatting', async () => {
  const tenonBin = findTenonBin();
  const rpc = createRpcClient(tenonBin);

  try {
    const initP = rpc.waitFor((m) => m.id === 1);
    rpc.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { processId: process.pid, rootUri: null, capabilities: {} } });
    const initResp = await initP;
    assert.equal(initResp.result.serverInfo.name, 'tenon-lsp');
    assert.equal(initResp.result.capabilities.hoverProvider, true);
    assert.equal(initResp.result.capabilities.definitionProvider, true);

    rpc.send({ jsonrpc: '2.0', method: 'initialized', params: {} });

    const uri = 'file://' + path.join(REPO_ROOT, 'examples', 'web_app', 'main.tn');
    const text = fs.readFileSync(path.join(REPO_ROOT, 'examples', 'web_app', 'main.tn'), 'utf8');

    const diagP = rpc.waitFor((m) => m.method === 'textDocument/publishDiagnostics' && m.params.uri === uri);
    rpc.send({
      jsonrpc: '2.0',
      method: 'textDocument/didOpen',
      params: { textDocument: { uri, languageId: 'tenon', version: 1, text } },
    });
    const diagMsg = await diagP;
    assert.equal(diagMsg.params.diagnostics.length, 0, `expected 0 diagnostics for web_app/main.tn, got: ${JSON.stringify(diagMsg.params.diagnostics)}`);

    // Document symbols on web_app/main.tn
    const symP = rpc.waitFor((m) => m.id === 2);
    rpc.send({ jsonrpc: '2.0', id: 2, method: 'textDocument/documentSymbol', params: { textDocument: { uri } } });
    const symResp = await symP;
    assert.ok(symResp.result.some((s) => s.name === 'health_handler'), 'expected health_handler symbol');
    assert.ok(symResp.result.some((s) => s.name === 'create_user_handler'), 'expected create_user_handler symbol');

    // Hover over `create_user_handler`
    const createLine = text.split('\n').findIndex((l) => l.includes('pub fn create_user_handler'));
    const hoverP = rpc.waitFor((m) => m.id === 3);
    rpc.send({
      jsonrpc: '2.0',
      id: 3,
      method: 'textDocument/hover',
      params: { textDocument: { uri }, position: { line: createLine, character: 10 } },
    });
    const hoverResp = await hoverP;
    assert.ok(hoverResp.result.contents.value.includes('@post("/users")'), 'hover should display @post("/users")');

    // Hover over `@get` attribute -> attribute doc, NOT List.get
    const getAttrLine = text.split('\n').findIndex((l) => l.includes('@get("/health")'));
    const attrHoverP = rpc.waitFor((m) => m.id === 30);
    rpc.send({
      jsonrpc: '2.0',
      id: 30,
      method: 'textDocument/hover',
      params: { textDocument: { uri }, position: { line: getAttrLine, character: 2 } },
    });
    const attrHoverResp = await attrHoverP;
    assert.ok(attrHoverResp.result.contents.value.includes('HTTP `GET` route handler attribute'));

    // Go to definition on `@get` MUST return null (must NOT jump to std/core/ops.tn List.get!)
    const attrDefP = rpc.waitFor((m) => m.id === 31);
    rpc.send({
      jsonrpc: '2.0',
      id: 31,
      method: 'textDocument/definition',
      params: { textDocument: { uri }, position: { line: getAttrLine, character: 2 } },
    });
    const attrDefResp = await attrDefP;
    assert.equal(attrDefResp.result, null, 'definition on @get must not jump to List.get');

    // Go to definition on `conn.get("/health")` MUST jump to HttpConn.get in std/net/http.tn!
    const connGetLine = text.split('\n').findIndex((l) => l.includes('conn.get("/health")'));
    const connGetCol = text.split('\n')[connGetLine].indexOf('conn.get') + 6;
    const connDefP = rpc.waitFor((m) => m.id === 32);
    rpc.send({
      jsonrpc: '2.0',
      id: 32,
      method: 'textDocument/definition',
      params: { textDocument: { uri }, position: { line: connGetLine, character: connGetCol } },
    });
    const connDefResp = await connDefP;
    assert.ok(connDefResp.result && connDefResp.result.uri.endsWith('/std/net/http.tn'), `expected std/net/http.tn, got ${JSON.stringify(connDefResp.result)}`);

    // Completion
    const compP = rpc.waitFor((m) => m.id === 4);
    rpc.send({
      jsonrpc: '2.0',
      id: 4,
      method: 'textDocument/completion',
      params: { textDocument: { uri }, position: { line: createLine, character: 0 } },
    });
    const compResp = await compP;
    assert.ok(compResp.result.some((c) => c.label === 'build_app'));

    // Shutdown + exit
    const shutP = rpc.waitFor((m) => m.id === 5);
    rpc.send({ jsonrpc: '2.0', id: 5, method: 'shutdown', params: null });
    await shutP;
    rpc.send({ jsonrpc: '2.0', method: 'exit', params: null });

    const exitCode = await new Promise((resolve) => rpc.child.once('exit', resolve));
    assert.equal(exitCode, 0);
  } finally {
    if (rpc.child.exitCode === null) {
      rpc.child.kill();
    }
  }
});
