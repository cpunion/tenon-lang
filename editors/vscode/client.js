'use strict';

const fs = require('fs');
const path = require('path');

let client = null;

function resolveTenonExecutable(vscode) {
  const cfg = vscode.workspace.getConfiguration('tenon');
  const configured = (cfg.get('executable') || 'tenon').trim();
  if (configured !== 'tenon' && configured !== '') {
    return configured;
  }

  const folders = vscode.workspace.workspaceFolders || [];
  for (const folder of folders) {
    const root = folder.uri.fsPath;
    for (const rel of ['target/release/tenon', 'target/debug/tenon']) {
      const candidate = path.join(root, rel);
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    }
  }
  return 'tenon';
}

function activate(context) {
  const vscode = require('vscode');
  const { LanguageClient, TransportKind } = require('vscode-languageclient/node');

  const command = resolveTenonExecutable(vscode);
  const serverOptions = {
    run: { command, args: ['lsp'], transport: TransportKind.stdio },
    debug: { command, args: ['lsp'], transport: TransportKind.stdio },
  };

  const clientOptions = {
    documentSelector: [{ scheme: 'file', language: 'tenon' }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher('**/*.{tn,toml}'),
    },
  };

  client = new LanguageClient(
    'tenonLanguageServer',
    'Tenon Language Server',
    serverOptions,
    clientOptions
  );

  context.subscriptions.push(client);
  client.start();
}

function deactivate() {
  if (!client) {
    return undefined;
  }
  return client.stop();
}

module.exports = { activate, deactivate, resolveTenonExecutable };
