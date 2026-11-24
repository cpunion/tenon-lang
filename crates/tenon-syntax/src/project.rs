use crate::ast::*;
use crate::parser::parse_with_file_id;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Compilation mode controlling test file (`*_test.tn`) & `test fn` inclusion (Loom-aligned).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadMode {
    /// Production build/run/check: excludes `*_test.tn` files and strips `test fn` items.
    Production,
    /// Test execution: includes `*_test.tn` and `test fn` items for the root package only;
    /// all imported dependency packages are still loaded strictly in `Production` mode.
    Test,
}

#[derive(Debug, Clone, Default)]
pub struct DepSpec {
    pub path: String,
    pub configuration: BTreeMap<String, String>,
}

/// Parsed `tenon.toml` module manifest (Loom-style minimal TOML without external crates).
#[derive(Debug, Clone, Default)]
pub struct TenonManifest {
    pub name: String,
    pub version: String,
    pub root_dir: PathBuf,
    pub dependencies: BTreeMap<String, DepSpec>,
    pub configuration: BTreeMap<String, String>,
}

impl TenonManifest {
    pub fn parse(src: &str, root_dir: PathBuf) -> Result<Self, String> {
        let mut manifest = TenonManifest {
            root_dir,
            ..Default::default()
        };
        enum Section {
            Root,
            Module,
            Configuration,
            Dep(String),
            DepConfig(String),
        }
        let mut sec = Section::Root;
        for (idx, raw_line) in src.lines().enumerate() {
            let line = strip_toml_comment(raw_line).trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                let header = line[1..line.len() - 1].trim();
                if header == "module" || header == "package" {
                    sec = Section::Module;
                } else if header == "configuration" || header == "config" {
                    sec = Section::Configuration;
                } else if let Some(rest) = header.strip_prefix("dependencies.") {
                    if let Some(dep_name) = rest.strip_suffix(".configuration") {
                        sec = Section::DepConfig(dep_name.trim().to_string());
                    } else if let Some(dep_name) = rest.strip_suffix(".config") {
                        sec = Section::DepConfig(dep_name.trim().to_string());
                    } else {
                        sec = Section::Dep(rest.trim().to_string());
                    }
                } else {
                    sec = Section::Root;
                }
                continue;
            }
            let Some((k, v_raw)) = line.split_once('=') else {
                return Err(format!("tenon.toml:{}: expected `key = value`", idx + 1));
            };
            let key = k.trim();
            let val = unquote_toml_str(v_raw.trim());
            match &sec {
                Section::Module => match key {
                    "name" => manifest.name = val,
                    "version" => manifest.version = val,
                    _ => {}
                },
                Section::Configuration => {
                    manifest.configuration.insert(key.to_string(), val);
                }
                Section::Dep(dep) => {
                    let entry = manifest.dependencies.entry(dep.clone()).or_default();
                    if key == "path" {
                        entry.path = val;
                    }
                }
                Section::DepConfig(dep) => {
                    let entry = manifest.dependencies.entry(dep.clone()).or_default();
                    entry.configuration.insert(key.to_string(), val);
                }
                Section::Root => {
                    if key == "name" && manifest.name.is_empty() {
                        manifest.name = val;
                    }
                }
            }
        }
        Ok(manifest)
    }

    pub fn load_from_dir(dir: &Path) -> Result<Option<Self>, String> {
        let toml_path = dir.join("tenon.toml");
        if !toml_path.is_file() {
            return Ok(None);
        }
        let src = fs::read_to_string(&toml_path)
            .map_err(|e| format!("failed to read `{}`: {}", toml_path.display(), e))?;
        let canon_dir = fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
        Ok(Some(Self::parse(&src, canon_dir)?))
    }
}

fn strip_toml_comment(line: &str) -> &str {
    let mut in_str = false;
    let mut escaped = false;
    for (i, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_str {
            escaped = true;
            continue;
        }
        if ch == '"' {
            in_str = !in_str;
            continue;
        }
        if ch == '#' && !in_str {
            return &line[..i];
        }
    }
    line
}

fn unquote_toml_str(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        s[1..s.len() - 1]
            .replace("\\\"", "\"")
            .replace("\\\\", "\\")
            .replace("\\n", "\n")
    } else if s.len() >= 2 && s.starts_with('\'') && s.ends_with('\'') {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

/// Walk upward from `start` to find the nearest enclosing `tenon.toml` module manifest.
pub fn find_enclosing_manifest(start: &Path) -> Result<Option<TenonManifest>, String> {
    let base = if start.is_file() {
        start.parent().unwrap_or_else(|| Path::new("."))
    } else {
        start
    };
    let mut cur = fs::canonicalize(base).unwrap_or_else(|_| base.to_path_buf());
    for _ in 0..16 {
        if let Some(m) = TenonManifest::load_from_dir(&cur)? {
            return Ok(Some(m));
        }
        if !cur.pop() {
            break;
        }
    }
    Ok(None)
}

pub fn is_test_source_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.ends_with("_test.tn"))
}

/// Discover all Tenon package directories recursively under `root`, stopping at nested
/// `tenon.toml` module boundaries (and ignoring `.git`, `target`, `node_modules`, hidden dirs),
/// matching Loom's `test_packages` discovery model.
pub fn discover_test_packages(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    discover_packages_rec(root, true, &mut out)?;
    out.sort();
    Ok(out)
}

fn discover_packages_rec(dir: &Path, is_root: bool, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if !dir.is_dir() {
        if dir.is_file() && dir.extension().and_then(|e| e.to_str()) == Some("tn") {
            out.push(dir.to_path_buf());
        }
        return Ok(());
    }
    if !is_root && dir.join("tenon.toml").is_file() {
        // Nested module boundary: still discover packages inside it only when explicitly requested
        // or treat it as its own root boundary.
        return Ok(());
    }
    let mut has_tn = false;
    let mut subdirs = Vec::new();
    let entries = fs::read_dir(dir)
        .map_err(|e| format!("failed to list `{}`: {}", dir.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }
        if path.is_file() && name.ends_with(".tn") {
            has_tn = true;
        } else if path.is_dir() {
            subdirs.push(path);
        }
    }
    if has_tn {
        out.push(dir.to_path_buf());
    }
    subdirs.sort();
    for sub in subdirs {
        discover_packages_rec(&sub, false, out)?;
    }
    Ok(())
}

/// Export table for a loaded package/module.
#[derive(Debug, Clone, Default)]
struct PackageExports {
    #[allow(dead_code)]
    package_id: u32,
    stem: String,
    /// Publicly exported symbol name -> actual lowered/resolved symbol name in merged `Program`.
    pub_symbols: BTreeMap<String, String>,
    /// All top-level symbol names defined in this package (pub + private) for clearer diagnostics.
    all_symbols: BTreeSet<String>,
}

/// Per-file namespace bindings created by its `import` statements.
#[derive(Debug, Clone, Default)]
struct LocalBindings {
    /// Namespace alias (e.g. `math` from `import demo.math` or `m` from `import demo.math as m`)
    /// -> `(stem, pub_symbol_map, all_symbols)`
    namespaces: BTreeMap<String, PackageExports>,
    /// Direct unqualified symbol aliases (from `import pkg.{A as B}` or collision-mangled imports):
    /// local_name -> target_name
    direct_aliases: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum PackageKey {
    Dir(PathBuf),
    File(PathBuf),
}

struct ProjectLoader {
    visited: BTreeMap<PackageKey, PackageExports>,
    loading: BTreeSet<PackageKey>,
    loaded_files: BTreeSet<PathBuf>,
    source_files: Vec<String>,
    file_packages: Vec<u32>,
    test_file_ids: Vec<u32>,
    next_package_id: u32,
    /// Tracks already-claimed global item names in `out_items` so two distinct packages defining
    /// the same top-level `fn`/`const` name don't collide at codegen time.
    global_fn_const_names: BTreeSet<String>,
    out_items: Vec<Item>,
    build_config: BTreeMap<String, String>,
    file_overlays: BTreeMap<PathBuf, String>,
}

/// Load a `.tn` file OR a Tenon package directory in `Production` or `Test` mode, resolving all
/// legacy string imports, Go-style dotted imports (`import pkg.subpkg [as alias]`),
/// wildcard imports (`import pkg.subpkg.*`), and selective imports (`import pkg.subpkg.{A, B as C}`),
/// then running `@cfg`, `comptime` + macros, and `@derive(...)`.
pub fn load_package(entry: &Path, mode: LoadMode) -> Result<Program, String> {
    load_package_with_config(entry, mode, &BTreeMap::new())
}

pub fn load_package_with_config(
    entry: &Path,
    mode: LoadMode,
    extra_config: &BTreeMap<String, String>,
) -> Result<Program, String> {
    load_package_with_overlays(entry, mode, extra_config, &BTreeMap::new())
}

pub fn load_package_with_overlays(
    entry: &Path,
    mode: LoadMode,
    extra_config: &BTreeMap<String, String>,
    overlays: &BTreeMap<PathBuf, String>,
) -> Result<Program, String> {
    let manifest = find_enclosing_manifest(entry)?;
    let mut build_config = manifest
        .as_ref()
        .map(|m| m.configuration.clone())
        .unwrap_or_default();
    for (k, v) in extra_config {
        build_config.insert(k.clone(), v.clone());
    }
    let mut canon_overlays = BTreeMap::new();
    for (p, text) in overlays {
        let cp = fs::canonicalize(p).unwrap_or_else(|_| p.clone());
        canon_overlays.insert(cp, text.clone());
    }

    let mut loader = ProjectLoader {
        visited: BTreeMap::new(),
        loading: BTreeSet::new(),
        loaded_files: BTreeSet::new(),
        source_files: vec!["<inline>".to_string()],
        file_packages: vec![0],
        test_file_ids: Vec::new(),
        next_package_id: 1,
        global_fn_const_names: BTreeSet::new(),
        out_items: Vec::new(),
        build_config,
        file_overlays: canon_overlays,
    };

    if entry.is_dir() {
        let canon = fs::canonicalize(entry).unwrap_or_else(|_| entry.to_path_buf());
        loader.load_dir_package(&canon, mode, manifest.as_ref(), true)?;
    } else {
        let canon = fs::canonicalize(entry).unwrap_or_else(|_| entry.to_path_buf());
        let parent = canon.parent().unwrap_or_else(|| Path::new("."));
        // If this file sits inside a `tenon.toml`-governed package directory (or is a `*_test.tn`
        // file in a directory with sibling production `.tn` files), load the whole package directory
        // so sibling files share private declarations (Loom directory-as-package semantics!).
        let has_manifest_in_parent = manifest.is_some();
        let is_test_entry = mode == LoadMode::Test && is_test_source_file(&canon);
        if has_manifest_in_parent || is_test_entry {
            let parent_canon = fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf());
            loader.load_dir_package(&parent_canon, mode, manifest.as_ref(), true)?;
        } else {
            loader.load_single_file(&canon, mode, manifest.as_ref(), true)?;
        }
    }

    let mut prog = Program {
        items: loader.out_items,
        source_files: loader.source_files,
        file_packages: loader.file_packages,
        test_file_ids: loader.test_file_ids,
        build_config: loader.build_config,
        comptime_items: Vec::new(),
    };
    crate::eval_cfg_attrs(&mut prog);
    crate::comptime::expand_comptime_and_macros(&mut prog)?;
    crate::expand_derives(&mut prog);
    Ok(prog)
}

enum ResolvedImportTarget {
    Dir(PathBuf),
    File(PathBuf),
    /// Dotted import where the last segment is an exported item inside the resolved package
    /// (e.g. `import std.text.contains` or `import codec.parser.answer`).
    ItemInDir { dir: PathBuf, item: String },
    ItemInFile { file: PathBuf, item: String },
}

impl ProjectLoader {
    fn alloc_package_id(&mut self) -> u32 {
        let id = self.next_package_id;
        self.next_package_id += 1;
        id
    }

    fn load_dir_package(
        &mut self,
        dir: &Path,
        mode: LoadMode,
        manifest: Option<&TenonManifest>,
        is_root: bool,
    ) -> Result<PackageExports, String> {
        let key = PackageKey::Dir(dir.to_path_buf());
        if let Some(exp) = self.visited.get(&key) {
            return Ok(exp.clone());
        }
        if !self.loading.insert(key.clone()) {
            return Ok(PackageExports::default());
        }

        // Check if this directory has its own manifest (e.g. a dependency module root)
        let own_manifest = find_enclosing_manifest(dir)?;
        let active_manifest = own_manifest.as_ref().or(manifest);

        let mut files: Vec<PathBuf> = Vec::new();
        let entries = fs::read_dir(dir)
            .map_err(|e| format!("failed to list package `{}`: {}", dir.display(), e))?;
        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let p = entry.path();
            if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("tn") {
                let is_test_file = is_test_source_file(&p);
                if is_test_file && mode != LoadMode::Test {
                    continue;
                }
                files.push(p);
            }
        }
        // Sort production files before `*_test.tn` files for deterministic order!
        files.sort_by(|a, b| {
            let at = is_test_source_file(a);
            let bt = is_test_source_file(b);
            at.cmp(&bt).then_with(|| a.cmp(b))
        });

        if files.is_empty() {
            self.loading.remove(&key);
            return Err(format!("no `.tn` files found in package `{}`", dir.display()));
        }

        let pkg_id = self.alloc_package_id();
        let stem = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let exp = self.load_package_files(dir, &files, pkg_id, &stem, mode, active_manifest, is_root)?;
        self.loading.remove(&key);
        self.visited.insert(key, exp.clone());
        Ok(exp)
    }

    fn load_single_file(
        &mut self,
        file: &Path,
        mode: LoadMode,
        manifest: Option<&TenonManifest>,
        is_root: bool,
    ) -> Result<PackageExports, String> {
        let key = PackageKey::File(file.to_path_buf());
        if let Some(exp) = self.visited.get(&key) {
            return Ok(exp.clone());
        }
        if !self.loading.insert(key.clone()) {
            return Ok(PackageExports::default());
        }
        let own_manifest = find_enclosing_manifest(file)?;
        let active_manifest = own_manifest.as_ref().or(manifest);
        let parent = file.parent().unwrap_or_else(|| Path::new("."));
        let pkg_id = self.alloc_package_id();
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let exp = self.load_package_files(
            parent,
            &[file.to_path_buf()],
            pkg_id,
            &stem,
            mode,
            active_manifest,
            is_root,
        )?;
        self.loading.remove(&key);
        self.visited.insert(key, exp.clone());
        Ok(exp)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_package_files(
        &mut self,
        pkg_dir: &Path,
        files: &[PathBuf],
        pkg_id: u32,
        stem: &str,
        mode: LoadMode,
        manifest: Option<&TenonManifest>,
        is_root: bool,
    ) -> Result<PackageExports, String> {
        // Phase 1: parse all files in this package and collect imports + local items per file.
        struct ParsedFile {
            path: PathBuf,
            is_test: bool,
            imports: Vec<(String, Option<String>, ImportKind, crate::span::Span)>,
            items: Vec<Item>,
        }
        let mut parsed_files = Vec::new();
        for fpath in files {
            let canon = fs::canonicalize(fpath).unwrap_or_else(|_| fpath.to_path_buf());
            if !self.loaded_files.insert(canon.clone()) {
                continue;
            }
            let file_id = self.source_files.len() as u32;
            self.source_files.push(canon.to_string_lossy().into_owned());
            self.file_packages.push(pkg_id);
            let is_test = is_test_source_file(&canon);
            if is_test {
                self.test_file_ids.push(file_id);
            }
            let src = if let Some(ov) = self.file_overlays.get(&canon) {
                ov.clone()
            } else {
                fs::read_to_string(&canon)
                    .map_err(|e| format!("failed to read `{}`: {}", canon.display(), e))?
            };
            let prog = parse_with_file_id(&src, file_id)
                .map_err(|e| format!("{}: {}", canon.display(), e))?;

            let mut imports = Vec::new();
            let mut local_items = Vec::new();
            for item in prog.items {
                match item {
                    Item::Import {
                        path,
                        alias,
                        kind,
                        span,
                    } => {
                        let k = kind.unwrap_or_else(|| ImportKind::FilePath(path.clone()));
                        imports.push((path, alias, k, span));
                    }
                    Item::Fn(ref f) if f.is_test && mode != LoadMode::Test => {
                        // Production build strips inline `test fn` declarations!
                    }
                    other => local_items.push(other),
                }
            }
            parsed_files.push(ParsedFile {
                path: canon,
                is_test,
                imports,
                items: local_items,
            });
        }

        // Phase 2: resolve & load all imported dependency packages (ALWAYS in Production mode!)
        let mut file_bindings: Vec<LocalBindings> = Vec::with_capacity(parsed_files.len());
        for pf in &parsed_files {
            let mut bindings = LocalBindings::default();
            for (raw_path, alias, kind, _span) in &pf.imports {
                match kind {
                    ImportKind::FilePath(rel) => {
                        let target = resolve_legacy_import_path(&pf.path, rel)?;
                        let target_canon =
                            fs::canonicalize(&target).unwrap_or_else(|_| target.clone());
                        let dep_exp = self.load_single_file(
                            &target_canon,
                            LoadMode::Production,
                            manifest,
                            false,
                        )?;
                        if let Some(a) = alias {
                            bindings.namespaces.insert(a.clone(), dep_exp);
                        }
                    }
                    ImportKind::Package(segments) => {
                        let resolved = resolve_dotted_import(
                            &pf.path,
                            pkg_dir,
                            segments,
                            manifest,
                            true,
                        )?;
                        match resolved {
                            ResolvedImportTarget::Dir(d) => {
                                let dep_exp = self.load_dir_package(
                                    &d,
                                    LoadMode::Production,
                                    manifest,
                                    false,
                                )?;
                                let ns = alias
                                    .clone()
                                    .unwrap_or_else(|| segments.last().cloned().unwrap_or_default());
                                bindings.namespaces.insert(ns, dep_exp.clone());
                                if segments.len() > 1 {
                                    bindings.namespaces.insert(segments.join("."), dep_exp);
                                }
                            }
                            ResolvedImportTarget::File(f) => {
                                let dep_exp = self.load_single_file(
                                    &f,
                                    LoadMode::Production,
                                    manifest,
                                    false,
                                )?;
                                let ns = alias
                                    .clone()
                                    .unwrap_or_else(|| segments.last().cloned().unwrap_or_default());
                                bindings.namespaces.insert(ns, dep_exp.clone());
                                if segments.len() > 1 {
                                    bindings.namespaces.insert(segments.join("."), dep_exp);
                                }
                            }
                            ResolvedImportTarget::ItemInDir { dir: d, item } => {
                                let dep_exp = self.load_dir_package(
                                    &d,
                                    LoadMode::Production,
                                    manifest,
                                    false,
                                )?;
                                let target_sym = resolve_imported_item(&dep_exp, &item, raw_path)?;
                                let local_name = alias.clone().unwrap_or_else(|| item.clone());
                                if local_name != target_sym {
                                    bindings.direct_aliases.insert(local_name, target_sym);
                                }
                                let pkg_ns = segments[..segments.len() - 1].join(".");
                                if !pkg_ns.is_empty() {
                                    bindings.namespaces.insert(pkg_ns, dep_exp.clone());
                                }
                                if segments.len() >= 2 {
                                    bindings
                                        .namespaces
                                        .insert(segments[segments.len() - 2].clone(), dep_exp);
                                }
                            }
                            ResolvedImportTarget::ItemInFile { file: f, item } => {
                                let dep_exp = self.load_single_file(
                                    &f,
                                    LoadMode::Production,
                                    manifest,
                                    false,
                                )?;
                                let target_sym = resolve_imported_item(&dep_exp, &item, raw_path)?;
                                let local_name = alias.clone().unwrap_or_else(|| item.clone());
                                if local_name != target_sym {
                                    bindings.direct_aliases.insert(local_name, target_sym);
                                }
                                let pkg_ns = segments[..segments.len() - 1].join(".");
                                if !pkg_ns.is_empty() {
                                    bindings.namespaces.insert(pkg_ns, dep_exp.clone());
                                }
                                if segments.len() >= 2 {
                                    bindings
                                        .namespaces
                                        .insert(segments[segments.len() - 2].clone(), dep_exp);
                                }
                            }
                        }
                    }
                    ImportKind::Glob(segments) => {
                        let resolved = resolve_dotted_import(
                            &pf.path,
                            pkg_dir,
                            segments,
                            manifest,
                            false,
                        )?;
                        let dep_exp = match resolved {
                            ResolvedImportTarget::Dir(d) => self.load_dir_package(
                                &d,
                                LoadMode::Production,
                                manifest,
                                false,
                            )?,
                            ResolvedImportTarget::File(f) => self.load_single_file(
                                &f,
                                LoadMode::Production,
                                manifest,
                                false,
                            )?,
                            _ => {
                                return Err(format!(
                                    "wildcard import `{}.*` must target a package",
                                    segments.join(".")
                                ));
                            }
                        };
                        for (orig, actual) in &dep_exp.pub_symbols {
                            if orig != actual {
                                bindings.direct_aliases.insert(orig.clone(), actual.clone());
                            }
                        }
                    }
                    ImportKind::Selective { segments, names } => {
                        let resolved = resolve_dotted_import(
                            &pf.path,
                            pkg_dir,
                            segments,
                            manifest,
                            false,
                        )?;
                        let dep_exp = match resolved {
                            ResolvedImportTarget::Dir(d) => self.load_dir_package(
                                &d,
                                LoadMode::Production,
                                manifest,
                                false,
                            )?,
                            ResolvedImportTarget::File(f) => self.load_single_file(
                                &f,
                                LoadMode::Production,
                                manifest,
                                false,
                            )?,
                            _ => {
                                return Err(format!(
                                    "selective import `{}.{{...}}` must target a package",
                                    segments.join(".")
                                ));
                            }
                        };
                        for named in names {
                            let target_sym =
                                resolve_imported_item(&dep_exp, &named.name, raw_path)?;
                            let local_name = named
                                .alias
                                .clone()
                                .unwrap_or_else(|| named.name.clone());
                            if local_name != target_sym {
                                bindings.direct_aliases.insert(local_name, target_sym);
                            }
                        }
                    }
                }
            }
            file_bindings.push(bindings);
        }

        // Phase 3: check for function/const name collisions against already-loaded packages.
        // If a non-root dependency defines a `fn` or `const` whose name is already claimed in
        // `global_fn_const_names` (e.g. `codec.answer` vs `other.answer`, or two private `helper`s),
        // cleanly alpha-rename it inside this package to `__p{pkg_id}__{name}`!
        let mut pkg_renames: BTreeMap<String, String> = BTreeMap::new();
        let mut exports = PackageExports {
            package_id: pkg_id,
            stem: stem.to_string(),
            pub_symbols: BTreeMap::new(),
            all_symbols: BTreeSet::new(),
        };

        for pf in &parsed_files {
            for item in &pf.items {
                match item {
                    Item::Fn(f) => {
                        exports.all_symbols.insert(f.name.clone());
                        let actual = if !is_root
                            && f.name != "main"
                            && !f.is_test
                            && self.global_fn_const_names.contains(&f.name)
                        {
                            let mangled = format!("__p{}__{}", pkg_id, f.name);
                            pkg_renames.insert(f.name.clone(), mangled.clone());
                            mangled
                        } else {
                            f.name.clone()
                        };
                        self.global_fn_const_names.insert(actual.clone());
                        if f.is_pub && !pf.is_test {
                            exports.pub_symbols.insert(f.name.clone(), actual);
                        }
                    }
                    Item::Const(c) => {
                        exports.all_symbols.insert(c.name.clone());
                        let actual = if !is_root && self.global_fn_const_names.contains(&c.name) {
                            let mangled = format!("__p{}__{}", pkg_id, c.name);
                            pkg_renames.insert(c.name.clone(), mangled.clone());
                            mangled
                        } else {
                            c.name.clone()
                        };
                        self.global_fn_const_names.insert(actual.clone());
                        if c.is_pub && !pf.is_test {
                            exports.pub_symbols.insert(c.name.clone(), actual);
                        }
                    }
                    Item::ExternFn(ef) => {
                        exports.all_symbols.insert(ef.name.clone());
                        if ef.is_pub && !pf.is_test {
                            exports.pub_symbols.insert(ef.name.clone(), ef.name.clone());
                        }
                    }
                    Item::Record(r) => {
                        exports.all_symbols.insert(r.name.clone());
                        if r.is_pub && !pf.is_test {
                            exports.pub_symbols.insert(r.name.clone(), r.name.clone());
                        }
                    }
                    Item::Enum(e) => {
                        exports.all_symbols.insert(e.name.clone());
                        if e.is_pub && !pf.is_test {
                            exports.pub_symbols.insert(e.name.clone(), e.name.clone());
                        }
                    }
                    Item::Bitfield(b) => {
                        exports.all_symbols.insert(b.name.clone());
                        if b.is_pub && !pf.is_test {
                            exports.pub_symbols.insert(b.name.clone(), b.name.clone());
                        }
                    }
                    Item::Union(u) => {
                        exports.all_symbols.insert(u.name.clone());
                        if u.is_pub && !pf.is_test {
                            exports.pub_symbols.insert(u.name.clone(), u.name.clone());
                        }
                    }
                    Item::TypeAlias(t) => {
                        exports.all_symbols.insert(t.name.clone());
                        if t.is_pub && !pf.is_test {
                            exports.pub_symbols.insert(t.name.clone(), t.name.clone());
                        }
                    }
                    Item::Interface(i) => {
                        exports.all_symbols.insert(i.name.clone());
                        if i.is_pub && !pf.is_test {
                            exports.pub_symbols.insert(i.name.clone(), i.name.clone());
                        }
                    }
                    Item::Macro(m) => {
                        exports.all_symbols.insert(m.name.clone());
                        if m.is_pub && !pf.is_test {
                            exports.pub_symbols.insert(m.name.clone(), m.name.clone());
                        }
                    }
                    Item::AttrDecl(a) => {
                        exports.all_symbols.insert(a.name.clone());
                        if a.is_pub && !pf.is_test {
                            exports.pub_symbols.insert(a.name.clone(), a.name.clone());
                        }
                    }
                    _ => {}
                }
            }
        }

        // Phase 4: rewrite namespace qualifiers & local/collision aliases inside each file's items.
        for (mut pf, mut bindings) in parsed_files.into_iter().zip(file_bindings.into_iter()) {
            for (orig, mangled) in &pkg_renames {
                bindings
                    .direct_aliases
                    .entry(orig.clone())
                    .or_insert_with(|| mangled.clone());
            }
            for item in &mut pf.items {
                if let Item::Fn(f) = item {
                    if let Some(mangled) = pkg_renames.get(&f.name) {
                        f.name = mangled.clone();
                    }
                } else if let Item::Const(c) = item {
                    if let Some(mangled) = pkg_renames.get(&c.name) {
                        c.name = mangled.clone();
                    }
                }
                if !bindings.namespaces.is_empty() || !bindings.direct_aliases.is_empty() {
                    rewrite_item_bindings(item, &bindings);
                }
            }
            self.out_items.extend(pf.items);
        }

        Ok(exports)
    }
}

fn resolve_imported_item(
    dep_exp: &PackageExports,
    item_name: &str,
    import_path: &str,
) -> Result<String, String> {
    if let Some(actual) = dep_exp.pub_symbols.get(item_name) {
        return Ok(actual.clone());
    }
    if dep_exp.all_symbols.contains(item_name) {
        return Err(format!(
            "error[E0105]: cannot import private item `{}` from `{}` (mark it `pub` to export)",
            item_name, import_path
        ));
    }
    Err(format!(
        "error[E0106]: item `{}` not found in package `{}`",
        item_name, import_path
    ))
}

fn resolve_legacy_import_path(current_file: &Path, import_rel: &str) -> Result<PathBuf, String> {
    let rel = Path::new(import_rel);
    if rel.is_absolute() && rel.exists() {
        return Ok(rel.to_path_buf());
    }
    let parent = current_file.parent().unwrap_or_else(|| Path::new("."));
    let direct = parent.join(rel);
    if direct.exists() {
        return Ok(direct);
    }
    let mut cur = fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf());
    for _ in 0..10 {
        let cand = cur.join(rel);
        if cand.exists() {
            return Ok(cand);
        }
        if !cur.pop() {
            break;
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let cand = cwd.join(rel);
        if cand.exists() {
            return Ok(cand);
        }
    }
    Err(format!(
        "cannot resolve import `{}` from `{}`",
        import_rel,
        current_file.display()
    ))
}

fn find_std_root(start: &Path) -> Option<PathBuf> {
    let base = if start.is_file() {
        start.parent().unwrap_or_else(|| Path::new("."))
    } else {
        start
    };
    let mut cur = fs::canonicalize(base).unwrap_or_else(|_| base.to_path_buf());
    for _ in 0..12 {
        let cand = cur.join("std");
        if cand.is_dir() && cand.join("core").is_dir() {
            return Some(cand);
        }
        if !cur.pop() {
            break;
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let cand = cwd.join("std");
        if cand.is_dir() && cand.join("core").is_dir() {
            return Some(cand);
        }
    }
    // Compile-time fallback to workspace `std/` directory
    let repo_std = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|r| r.join("std"));
    if let Some(s) = repo_std {
        if s.is_dir() {
            return Some(s);
        }
    }
    None
}

fn resolve_under_base(
    base: &Path,
    tail: &[String],
    allow_item_leaf: bool,
) -> Option<ResolvedImportTarget> {
    let mut p = base.to_path_buf();
    for seg in tail {
        p.push(seg);
    }
    // 1. Exact `.tn` file match (e.g. `std/collections/map.tn` or `demo/math.tn`)
    let as_file = p.with_extension("tn");
    if as_file.is_file() {
        return Some(ResolvedImportTarget::File(
            fs::canonicalize(&as_file).unwrap_or(as_file),
        ));
    }
    // 2. Directory package match (containing at least one `.tn` file!)
    if p.is_dir() && dir_has_tn_files(&p) {
        return Some(ResolvedImportTarget::Dir(
            fs::canonicalize(&p).unwrap_or(p),
        ));
    }
    // 3. Item-leaf match (e.g. `import std.text.contains` or `import codec.parser.answer`)
    if allow_item_leaf && !tail.is_empty() {
        let item = tail.last()?.clone();
        let parent_tail = &tail[..tail.len() - 1];
        let mut parent_p = base.to_path_buf();
        for seg in parent_tail {
            parent_p.push(seg);
        }
        let parent_file = parent_p.with_extension("tn");
        if parent_file.is_file() {
            return Some(ResolvedImportTarget::ItemInFile {
                file: fs::canonicalize(&parent_file).unwrap_or(parent_file),
                item,
            });
        }
        if parent_p.is_dir() && dir_has_tn_files(&parent_p) {
            return Some(ResolvedImportTarget::ItemInDir {
                dir: fs::canonicalize(&parent_p).unwrap_or(parent_p),
                item,
            });
        }
    }
    None
}

fn dir_has_tn_files(dir: &Path) -> bool {
    let Ok(rd) = fs::read_dir(dir) else {
        return false;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("tn") {
            return true;
        }
    }
    false
}

fn resolve_dotted_import(
    current_file: &Path,
    pkg_dir: &Path,
    segments: &[String],
    manifest: Option<&TenonManifest>,
    allow_item_leaf: bool,
) -> Result<ResolvedImportTarget, String> {
    if segments.is_empty() {
        return Err("empty import path".into());
    }
    let root_seg = &segments[0];
    let tail = &segments[1..];

    // 1. Standard library: `std.<rest>`
    if root_seg == "std" {
        if let Some(std_root) = find_std_root(current_file) {
            if let Some(res) = resolve_under_base(&std_root, tail, allow_item_leaf) {
                return Ok(res);
            }
        }
    }

    // 2. Current module root from `tenon.toml`: `<module_name>.<rest>`
    if let Some(m) = manifest {
        if !m.name.is_empty() && root_seg == &m.name {
            if let Some(res) = resolve_under_base(&m.root_dir, tail, allow_item_leaf) {
                return Ok(res);
            }
        }
        // 3. Declared dependency in `tenon.toml`: `<dep_name>.<rest>`
        if let Some(dep) = m.dependencies.get(root_seg) {
            let dep_base = m.root_dir.join(&dep.path);
            let dep_canon = fs::canonicalize(&dep_base).unwrap_or(dep_base);
            if let Some(res) = resolve_under_base(&dep_canon, tail, allow_item_leaf) {
                return Ok(res);
            }
        }
    }

    // 4. Relative to current package directory (or parent workspace directory)
    if let Some(res) = resolve_under_base(pkg_dir, segments, allow_item_leaf) {
        return Ok(res);
    }
    if let Some(parent) = pkg_dir.parent() {
        if let Some(res) = resolve_under_base(parent, segments, allow_item_leaf) {
            return Ok(res);
        }
    }

    Err(format!(
        "cannot resolve package import `{}` from `{}`",
        segments.join("."),
        current_file.display()
    ))
}

// =========================================================================
// AST Namespace & Selective-Import Rewriter
// =========================================================================

fn resolve_qualified_symbol(full: &str, bindings: &LocalBindings) -> Option<String> {
    if let Some(direct) = bindings.direct_aliases.get(full) {
        return Some(direct.clone());
    }
    // Try longest matching namespace prefix (supports both `subpkg.Foo` and `pkg.subpkg.Foo`)
    if let Some(dot) = full.rfind('.') {
        let prefix = &full[..dot];
        let leaf = &full[dot + 1..];
        if let Some(exp) = bindings.namespaces.get(prefix) {
            if let Some(actual) = exp.pub_symbols.get(leaf) {
                return Some(actual.clone());
            }
            let prefixed = format!("{}_{}", exp.stem, leaf);
            if let Some(actual) = exp.pub_symbols.get(&prefixed) {
                return Some(actual.clone());
            }
            if exp.all_symbols.contains(leaf) {
                // Keep leaf so verifier emits E0105 on cross-package private access!
                return Some(leaf.to_string());
            }
        }
    }
    // Check enum variant prefix like `subpkg.Color.Red` -> `Color.Red`
    let parts: Vec<&str> = full.split('.').collect();
    if parts.len() >= 3 {
        for split in (1..parts.len() - 1).rev() {
            let ns = parts[..split].join(".");
            let ty_name = parts[split];
            if let Some(exp) = bindings.namespaces.get(&ns) {
                if let Some(actual_ty) = exp.pub_symbols.get(ty_name) {
                    let rest = parts[split + 1..].join(".");
                    return Some(format!("{}.{}", actual_ty, rest));
                }
            }
        }
    }
    None
}

fn rewrite_type_bindings(ty: &mut TypeExpr, bindings: &LocalBindings) {
    match ty {
        TypeExpr::Named { name, args, .. } => {
            if let Some(resolved) = resolve_qualified_symbol(name, bindings) {
                *name = resolved;
            }
            for a in args {
                rewrite_type_bindings(a, bindings);
            }
        }
        TypeExpr::Dyn { interface, args, .. } => {
            if let Some(resolved) = resolve_qualified_symbol(interface, bindings) {
                *interface = resolved;
            }
            for a in args {
                rewrite_type_bindings(a, bindings);
            }
        }
        TypeExpr::Tuple(es, _) => {
            for e in es {
                rewrite_type_bindings(e, bindings);
            }
        }
        TypeExpr::Fn { params, ret, .. } => {
            for p in params {
                rewrite_type_bindings(p, bindings);
            }
            if let Some(r) = ret {
                rewrite_type_bindings(r, bindings);
            }
        }
        TypeExpr::Array { elem, .. } => rewrite_type_bindings(elem, bindings),
    }
}

fn rewrite_pattern_bindings(pat: &mut Pattern, bindings: &LocalBindings) {
    match pat {
        Pattern::Variant { name, .. } => {
            if let Some(resolved) = resolve_qualified_symbol(name, bindings) {
                *name = resolved;
            }
        }
        Pattern::VariantMulti { name, patterns, .. } => {
            if let Some(resolved) = resolve_qualified_symbol(name, bindings) {
                *name = resolved;
            }
            for p in patterns {
                rewrite_pattern_bindings(p, bindings);
            }
        }
        Pattern::Tuple(ps, _) => {
            for p in ps {
                rewrite_pattern_bindings(p, bindings);
            }
        }
        _ => {}
    }
}

fn rewrite_item_bindings(item: &mut Item, bindings: &LocalBindings) {
    match item {
        Item::Fn(f) => rewrite_fn_bindings(f, bindings),
        Item::ExternFn(ef) => {
            for p in &mut ef.params {
                rewrite_type_bindings(&mut p.ty, bindings);
            }
            if let Some(rt) = &mut ef.ret_ty {
                rewrite_type_bindings(rt, bindings);
            }
        }
        Item::TypeAlias(ta) => {
            rewrite_type_bindings(&mut ta.base, bindings);
            if let Some(p) = &mut ta.predicate {
                rewrite_expr_bindings(p, bindings);
            }
        }
        Item::Record(r) => {
            for f in &mut r.fields {
                rewrite_type_bindings(&mut f.ty, bindings);
                if let Some(p) = &mut f.predicate {
                    rewrite_expr_bindings(p, bindings);
                }
                if let Some(d) = &mut f.default {
                    rewrite_expr_bindings(d, bindings);
                }
            }
            for w in &mut r.where_clauses {
                rewrite_expr_bindings(w, bindings);
            }
            for m in &mut r.methods {
                rewrite_fn_bindings(m, bindings);
            }
        }
        Item::Enum(e) => {
            for v in &mut e.variants {
                for p in &mut v.fields {
                    rewrite_type_bindings(&mut p.ty, bindings);
                }
            }
        }
        Item::Union(u) => {
            for f in &mut u.fields {
                rewrite_type_bindings(&mut f.ty, bindings);
            }
        }
        Item::Interface(iface) => {
            for m in &mut iface.methods {
                for p in &mut m.params {
                    rewrite_type_bindings(&mut p.ty, bindings);
                }
                if let Some(rt) = &mut m.ret_ty {
                    rewrite_type_bindings(rt, bindings);
                }
                if let Some(b) = &mut m.default_body {
                    rewrite_block_bindings(b, bindings);
                }
            }
        }
        Item::Impl(imp) => {
            if let Some(i) = &mut imp.interface {
                rewrite_type_bindings(i, bindings);
            }
            rewrite_type_bindings(&mut imp.target_ty, bindings);
            for (_, ty) in &mut imp.assoc_bindings {
                rewrite_type_bindings(ty, bindings);
            }
            for m in &mut imp.methods {
                rewrite_fn_bindings(m, bindings);
            }
        }
        Item::Const(c) => {
            rewrite_type_bindings(&mut c.ty, bindings);
            rewrite_expr_bindings(&mut c.value, bindings);
        }
        Item::ComptimeBlock(b) => rewrite_block_bindings(b, bindings),
        Item::Macro(m) => rewrite_expr_bindings(&mut m.body, bindings),
        Item::MacroCall { args, .. } => {
            for a in args {
                rewrite_expr_bindings(a, bindings);
            }
        }
        _ => {}
    }
}

fn rewrite_fn_bindings(f: &mut FnDecl, bindings: &LocalBindings) {
    for p in &mut f.params {
        rewrite_type_bindings(&mut p.ty, bindings);
        if let Some(pr) = &mut p.predicate {
            rewrite_expr_bindings(pr, bindings);
        }
    }
    if let Some(rt) = &mut f.ret_ty {
        rewrite_type_bindings(rt, bindings);
    }
    for r in &mut f.requires {
        rewrite_expr_bindings(r, bindings);
    }
    for e in &mut f.ensures {
        rewrite_expr_bindings(e, bindings);
    }
    rewrite_block_bindings(&mut f.body, bindings);
}

fn rewrite_block_bindings(blk: &mut Block, bindings: &LocalBindings) {
    for s in &mut blk.stmts {
        match s {
            Stmt::Let { ty, value, .. } | Stmt::LetTuple { ty, value, .. } => {
                if let Some(t) = ty {
                    rewrite_type_bindings(t, bindings);
                }
                rewrite_expr_bindings(value, bindings);
            }
            Stmt::Assign { target, value, .. } => {
                rewrite_expr_bindings(target, bindings);
                rewrite_expr_bindings(value, bindings);
            }
            Stmt::Return { value: Some(e), .. }
            | Stmt::Assert { cond: e, .. }
            | Stmt::Expr { expr: e, .. } => rewrite_expr_bindings(e, bindings),
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } => {
                rewrite_block_bindings(body, bindings)
            }
            Stmt::While { cond, body, .. } => {
                rewrite_expr_bindings(cond, bindings);
                rewrite_block_bindings(body, bindings);
            }
            Stmt::For { iter, body, .. } => {
                rewrite_expr_bindings(iter, bindings);
                rewrite_block_bindings(body, bindings);
            }
            _ => {}
        }
    }
    if let Some(t) = &mut blk.tail {
        rewrite_expr_bindings(t, bindings);
    }
}

fn flatten_field_chain(expr: &Expr) -> Option<String> {
    match &expr.kind {
        ExprKind::Ident(s) => Some(s.clone()),
        ExprKind::Field { target, field } => {
            let mut p = flatten_field_chain(target)?;
            p.push('.');
            p.push_str(field);
            Some(p)
        }
        _ => None,
    }
}

fn rewrite_expr_bindings(expr: &mut Expr, bindings: &LocalBindings) {
    // First try full dotted-chain namespace resolution (e.g. `subpkg.add` or `pkg.subpkg.add`)
    if matches!(&expr.kind, ExprKind::Field { .. }) {
        if let Some(full) = flatten_field_chain(expr) {
            if let Some(resolved) = resolve_qualified_symbol(&full, bindings) {
                if let Some((head, tail)) = resolved.split_once('.') {
                    expr.kind = ExprKind::Field {
                        target: Box::new(Expr {
                            kind: ExprKind::Ident(head.to_string()),
                            span: expr.span,
                        }),
                        field: tail.to_string(),
                    };
                } else {
                    expr.kind = ExprKind::Ident(resolved);
                }
                return;
            }
        }
    }
    match &mut expr.kind {
        ExprKind::Ident(s) => {
            if let Some(mapped) = bindings.direct_aliases.get(s.as_str()) {
                *s = mapped.clone();
            }
        }
        ExprKind::Field { target, .. } => {
            rewrite_expr_bindings(target, bindings);
        }
        ExprKind::GenericInst { expr: inner, type_args } => {
            rewrite_expr_bindings(inner, bindings);
            for ta in type_args {
                rewrite_type_bindings(ta, bindings);
            }
        }
        ExprKind::Call { callee, args } => {
            rewrite_expr_bindings(callee, bindings);
            for a in args {
                rewrite_expr_bindings(&mut a.value, bindings);
            }
        }
        ExprKind::MacroCall { name, args } => {
            if let Some(resolved) = resolve_qualified_symbol(name, bindings) {
                *name = resolved;
            }
            for a in args {
                rewrite_expr_bindings(a, bindings);
            }
        }
        ExprKind::Binary { left, right, .. } => {
            rewrite_expr_bindings(left, bindings);
            rewrite_expr_bindings(right, bindings);
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Await(inner)
        | ExprKind::Old(inner)
        | ExprKind::Try(inner)
        | ExprKind::ArrayRepeat { value: inner, .. }
        | ExprKind::TupleIndex { target: inner, .. } => {
            rewrite_expr_bindings(inner, bindings);
        }
        ExprKind::Cast { expr: inner, ty } => {
            rewrite_expr_bindings(inner, bindings);
            rewrite_type_bindings(ty, bindings);
        }
        ExprKind::TypeImplements { ty, interface } => {
            rewrite_type_bindings(ty, bindings);
            if let Some(resolved) = resolve_qualified_symbol(interface, bindings) {
                *interface = resolved;
            }
        }
        ExprKind::Range { start, end } => {
            rewrite_expr_bindings(start, bindings);
            rewrite_expr_bindings(end, bindings);
        }
        ExprKind::Lambda { params, ret_ty, body } => {
            for p in params {
                if let Some(t) = &mut p.ty {
                    rewrite_type_bindings(t, bindings);
                }
            }
            if let Some(rt) = ret_ty {
                rewrite_type_bindings(rt, bindings);
            }
            rewrite_expr_bindings(body, bindings);
        }
        ExprKind::Index { target, index } => {
            rewrite_expr_bindings(target, bindings);
            rewrite_expr_bindings(index, bindings);
        }
        ExprKind::RecordLit { name, fields, .. } => {
            if let Some(resolved) = resolve_qualified_symbol(name, bindings) {
                *name = resolved;
            }
            for (_, v) in fields {
                rewrite_expr_bindings(v, bindings);
            }
        }
        ExprKind::Tuple(es) | ExprKind::List(es) => {
            for e in es {
                rewrite_expr_bindings(e, bindings);
            }
        }
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        }
        | ExprKind::ComptimeIf {
            cond,
            then_block,
            else_branch,
        } => {
            rewrite_expr_bindings(cond, bindings);
            rewrite_block_bindings(then_block, bindings);
            if let Some(eb) = else_branch {
                rewrite_expr_bindings(eb, bindings);
            }
        }
        ExprKind::ComptimeFor { iter, body, .. } => {
            rewrite_expr_bindings(iter, bindings);
            rewrite_block_bindings(body, bindings);
        }
        ExprKind::Match { scrutinee, arms } => {
            rewrite_expr_bindings(scrutinee, bindings);
            for a in arms {
                for p in &mut a.patterns {
                    rewrite_pattern_bindings(p, bindings);
                }
                if let Some(g) = &mut a.guard {
                    rewrite_expr_bindings(g, bindings);
                }
                rewrite_expr_bindings(&mut a.body, bindings);
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) | ExprKind::Comptime(b) => {
            rewrite_block_bindings(b, bindings)
        }
        _ => {}
    }
}
