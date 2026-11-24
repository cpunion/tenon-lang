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

  const debugCmd = vscode.commands.registerCommand('tenon.debugFile', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'tenon') {
      vscode.window.showWarningMessage('Open a .tn file to build & debug with Tenon.');
      return;
    }
    await editor.document.save();
    const filePath = editor.document.uri.fsPath;
    const cmd = resolveTenonExecutable(vscode);
    const outBin = path.join(os.tmpdir(), `tenon_dbg_${path.basename(filePath, '.tn')}`);
    const { execFile } = require('child_process');
    outputChannel.appendLine(`[tenon] Building DWARF debug binary: ${cmd} build ${filePath} -g --opt O0 -o ${outBin}`);
    execFile(cmd, ['build', filePath, '-g', '--opt', 'O0', '-o', outBin], async (err, stdout, stderr) => {
      if (err) {
        outputChannel.show(true);
        outputChannel.appendLine(stderr || stdout || String(err));
        vscode.window.showErrorMessage(`Tenon debug build failed: ${stderr || err.message}`);
        return;
      }
      outputChannel.appendLine(stdout);
      const debugType = vscode.extensions.getExtension('vadimcn.vscode-lldb') ? 'lldb' : 'cppdbg';
      await vscode.debug.startDebugging(vscode.workspace.workspaceFolders?.[0], {
        name: `Tenon Debug (${path.basename(filePath)})`,
        type: debugType,
        request: 'launch',
        program: outBin,
        cwd: path.dirname(filePath),
        MIMode: 'lldb',
      });
    });
  });
  context.subscriptions.push(debugCmd);
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
