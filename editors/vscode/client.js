'use strict';

const fs = require('fs');
const os = require('os');
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

  const home = os.homedir();
  for (const candidate of [
    path.join(home, '.local', 'bin', 'tenon'),
    path.join(home, '.cargo', 'bin', 'tenon'),
    '/opt/homebrew/bin/tenon',
    '/usr/local/bin/tenon',
  ]) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return 'tenon';
}

async function startClient(vscode, outputChannel) {
  const { LanguageClient } = require('vscode-languageclient/node');
  const command = resolveTenonExecutable(vscode);
  outputChannel.appendLine(`[tenon] Starting language server: ${command} lsp`);

  const serverOptions = {
    run: { command, args: ['lsp'] },
    debug: { command, args: ['lsp'] },
  };

  const watcher = vscode.workspace.createFileSystemWatcher('**/*.{tn,toml}');
  const clientOptions = {
    documentSelector: [{ scheme: 'file', language: 'tenon' }],
    outputChannel,
    synchronize: {
      configurationSection: 'tenon',
      fileEvents: watcher,
    },
  };

  client = new LanguageClient(
    'tenonLanguageServer',
    'Tenon Language Server',
    serverOptions,
    clientOptions
  );

  await client.start();
  return watcher;
}

async function activate(context) {
  const vscode = require('vscode');
  const outputChannel = vscode.window.createOutputChannel('Tenon Language Server');
  context.subscriptions.push(outputChannel);

  const watcher = await startClient(vscode, outputChannel);
  context.subscriptions.push(watcher);

  const restartCmd = vscode.commands.registerCommand('tenon.restartServer', async () => {
    outputChannel.appendLine('[tenon] Restarting Tenon Language Server...');
    if (client) {
      await client.stop();
      client = null;
    }
    const newWatcher = await startClient(vscode, outputChannel);
    context.subscriptions.push(newWatcher);
    vscode.window.showInformationMessage('Tenon Language Server restarted.');
  });
  context.subscriptions.push(restartCmd);
}

async function deactivate() {
  if (!client) {
    return undefined;
  }
  const c = client;
  client = null;
  return c.stop();
}

module.exports = { activate, deactivate, resolveTenonExecutable };
