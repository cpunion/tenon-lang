//! Exact scalar width metadata, SIMD vector descriptors, and C-ABI struct/union/bitfield layout engine.
//!
//! Shared across `tenon-verify`, `tenon-codegen` (Cranelift JIT & LLVM 22 AOT), and C header generation
//! so `size_of[T]()`, `align_of[T]()`, `offset_of(T, f)`, `@repr(C)`, `@packed`, `@align(N)`, `union`,
//! and `bitfield` agree down to the exact bit and byte.

use crate::ast::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
}

impl ScalarKind {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Int8" | "CChar" => Some(Self::I8),
            "Int16" => Some(Self::I16),
            "Int32" | "CInt" => Some(Self::I32),
            "Int" | "Int64" | "ISize" => Some(Self::I64),
            "UInt8" | "Byte" => Some(Self::U8),
            "UInt16" => Some(Self::U16),
            "UInt32" | "CUInt" => Some(Self::U32),
            "UInt64" | "USize" | "CSize" => Some(Self::U64),
            "Float32" => Some(Self::F32),
            "Float" | "Float64" => Some(Self::F64),
            _ => None,
        }
    }

    pub fn is_int(self) -> bool {
        !self.is_float()
    }

    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    pub fn is_signed(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    pub fn is_unsigned(self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    pub fn bit_width(self) -> u32 {
        match self {
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 | Self::F64 => 64,
        }
    }

    pub fn byte_size(self) -> usize {
        (self.bit_width() / 8) as usize
    }

    pub fn align(self) -> usize {
        self.byte_size()
    }

    pub fn min_i128(self) -> Option<i128> {
        match self {
            Self::I8 => Some(i8::MIN as i128),
            Self::I16 => Some(i16::MIN as i128),
            Self::I32 => Some(i32::MIN as i128),
            Self::I64 => Some(i64::MIN as i128),
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => Some(0),
            Self::F32 | Self::F64 => None,
        }
    }

    pub fn max_i128(self) -> Option<i128> {
        match self {
            Self::I8 => Some(i8::MAX as i128),
            Self::I16 => Some(i16::MAX as i128),
            Self::I32 => Some(i32::MAX as i128),
            Self::I64 => Some(i64::MAX as i128),
            Self::U8 => Some(u8::MAX as i128),
            Self::U16 => Some(u16::MAX as i128),
            Self::U32 => Some(u32::MAX as i128),
            Self::U64 => Some(u64::MAX as i128),
            Self::F32 | Self::F64 => None,
        }
    }

    pub fn mask_u64(self) -> u64 {
        match self {
            Self::I8 | Self::U8 => 0xFF,
            Self::I16 | Self::U16 => 0xFFFF,
            Self::I32 | Self::U32 | Self::F32 => 0xFFFF_FFFF,
            Self::I64 | Self::U64 | Self::F64 => u64::MAX,
        }
    }

    pub fn llvm_ty(self) -> &'static str {
        match self {
            Self::I8 | Self::U8 => "i8",
            Self::I16 | Self::U16 => "i16",
            Self::I32 | Self::U32 => "i32",
            Self::I64 | Self::U64 => "i64",
            Self::F32 => "float",
            Self::F64 => "double",
        }
    }

    pub fn c_ty(self) -> &'static str {
        match self {
            Self::I8 => "int8_t",
            Self::I16 => "int16_t",
            Self::I32 => "int32_t",
            Self::I64 => "int64_t",
            Self::U8 => "uint8_t",
            Self::U16 => "uint16_t",
            Self::U32 => "uint32_t",
            Self::U64 => "uint64_t",
            Self::F32 => "float",
            Self::F64 => "double",
        }
    }

    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::I8 => "Int8",
            Self::I16 => "Int16",
            Self::I32 => "Int32",
            Self::I64 => "Int",
            Self::U8 => "UInt8",
            Self::U16 => "UInt16",
            Self::U32 => "UInt32",
            Self::U64 => "UInt64",
            Self::F32 => "Float32",
            Self::F64 => "Float",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimdKind {
    F32x4,
    F64x4,
    I32x4,
    I64x4,
    U8x16,
}

impl SimdKind {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Float32x4" | "F32x4" => Some(Self::F32x4),
            "Float64x4" | "F64x4" => Some(Self::F64x4),
            "Int32x4" | "I32x4" => Some(Self::I32x4),
            "Int64x4" | "I64x4" => Some(Self::I64x4),
            "UInt8x16" | "U8x16" => Some(Self::U8x16),
            _ => None,
        }
    }

    pub fn from_type_expr(ty: &TypeExpr) -> Option<Self> {
        if let TypeExpr::Named { name, args, .. } = ty {
            if let Some(k) = Self::from_name(name) {
                return Some(k);
            }
            if name == "Simd" && args.len() == 2 {
                let elem = args[0].display_name();
                let lanes = args[1].display_name().parse::<usize>().ok()?;
                return match (ScalarKind::from_name(&elem)?, lanes) {
                    (ScalarKind::F32, 4) => Some(Self::F32x4),
                    (ScalarKind::F64, 4) => Some(Self::F64x4),
                    (ScalarKind::I32, 4) => Some(Self::I32x4),
                    (ScalarKind::I64, 4) => Some(Self::I64x4),
                    (ScalarKind::U8, 16) => Some(Self::U8x16),
                    _ => None,
                };
            }
        }
        None
    }

    pub fn lane_kind(self) -> ScalarKind {
        match self {
            Self::F32x4 => ScalarKind::F32,
            Self::F64x4 => ScalarKind::F64,
            Self::I32x4 => ScalarKind::I32,
            Self::I64x4 => ScalarKind::I64,
            Self::U8x16 => ScalarKind::U8,
        }
    }

    pub fn lanes(self) -> usize {
        match self {
            Self::U8x16 => 16,
            _ => 4,
        }
    }

    pub fn byte_size(self) -> usize {
        self.lanes() * self.lane_kind().byte_size()
    }

    pub fn align(self) -> usize {
        self.byte_size()
    }

    pub fn llvm_vec_ty(self) -> &'static str {
        match self {
            Self::F32x4 => "<4 x float>",
            Self::F64x4 => "<4 x double>",
            Self::I32x4 => "<4 x i32>",
            Self::I64x4 => "<4 x i64>",
            Self::U8x16 => "<16 x i8>",
        }
    }

    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::F32x4 => "Float32x4",
            Self::F64x4 => "Float64x4",
            Self::I32x4 => "Int32x4",
            Self::I64x4 => "Int64x4",
            Self::U8x16 => "UInt8x16",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldByteLayout {
    pub name: String,
    pub ty: TypeExpr,
    pub byte_offset: usize,
    pub byte_size: usize,
    pub align: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructByteLayout {
    pub name: String,
    pub is_c_repr: bool,
    pub is_packed: bool,
    pub is_union: bool,
    pub size: usize,
    pub align: usize,
    pub fields: Vec<FieldByteLayout>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitfieldFieldLayout {
    pub name: String,
    pub bit_offset: u32,
    pub bit_width: u32,
    pub mask: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitfieldLayout {
    pub name: String,
    pub backing: ScalarKind,
    pub total_bits: u32,
    pub fields: Vec<BitfieldFieldLayout>,
}

pub fn has_repr_c(attrs: &[Attr]) -> bool {
    attrs.iter().any(|a| {
        if a.name == "repr" {
            a.args.iter().any(|arg| {
                matches!(&arg.value.kind, ExprKind::Ident(s) | ExprKind::Str(s) if s == "C" || s == "c")
            })
        } else {
            false
        }
    })
}

pub fn has_packed_attr(attrs: &[Attr]) -> bool {
    attrs.iter().any(|a| {
        a.name == "packed"
            || (a.name == "repr"
                && a.args.iter().any(|arg| {
                    matches!(&arg.value.kind, ExprKind::Ident(s) | ExprKind::Str(s) if s == "packed")
                }))
    })
}

pub fn explicit_align_attr(attrs: &[Attr]) -> Option<usize> {
    for a in attrs {
        if a.name == "align" {
            if let Some(arg) = a.args.first() {
                if let ExprKind::Int(n) = arg.value.kind {
                    if n > 0 && (n as usize).is_power_of_two() {
                        return Some(n as usize);
                    }
                }
            }
        }
    }
    None
}

pub fn has_gpu_struct(attrs: &[Attr]) -> bool {
    attrs.iter().any(|a| a.name == "gpu_struct")
}

pub fn gpu_struct_mode(attrs: &[Attr]) -> Option<String> {
    for a in attrs {
        if a.name == "gpu_struct" {
            if let Some(arg) = a.args.first() {
                if let ExprKind::Str(s) | ExprKind::Ident(s) = &arg.value.kind {
                    return Some(s.clone());
                }
            }
            return Some("std430".to_string());
        }
    }
    None
}

pub fn is_c_layout_attrs(attrs: &[Attr]) -> bool {
    has_repr_c(attrs)
        || has_packed_attr(attrs)
        || explicit_align_attr(attrs).is_some()
        || has_gpu_struct(attrs)
}

pub fn extract_export_name(attrs: &[Attr], default_name: &str) -> Option<String> {
    for a in attrs {
        if a.name == "export" {
            if let Some(arg) = a.args.first() {
                if let ExprKind::Str(s) = &arg.value.kind {
                    return Some(s.clone());
                }
            }
            return Some(default_name.to_string());
        }
    }
    None
}

pub fn extract_link_libs(attrs: &[Attr]) -> Vec<String> {
    let mut libs = Vec::new();
    for a in attrs {
        if a.name == "link" {
            for arg in &a.args {
                if arg.key.as_deref() == Some("framework") {
                    continue;
                }
                if let ExprKind::Str(s) = &arg.value.kind {
                    libs.push(s.clone());
                }
            }
        }
    }
    libs
}

pub fn extract_link_frameworks(attrs: &[Attr]) -> Vec<String> {
    let mut fws = Vec::new();
    for a in attrs {
        if a.name == "link" {
            for arg in &a.args {
                if arg.key.as_deref() == Some("framework") {
                    if let ExprKind::Str(s) = &arg.value.kind {
                        fws.push(s.clone());
                    }
                }
            }
        }
    }
    fws
}

pub fn align_up(offset: usize, align: usize) -> usize {
    if align <= 1 {
        return offset;
    }
    (offset + align - 1) & !(align - 1)
}

#[derive(Debug, Clone, Default)]
pub struct LayoutEngine {
    pub aliases: BTreeMap<String, TypeExpr>,
    pub records: BTreeMap<String, RecordDecl>,
    pub unions: BTreeMap<String, UnionDecl>,
    pub bitfields: BTreeMap<String, BitfieldDecl>,
}

impl LayoutEngine {
    pub fn from_program(program: &Program) -> Self {
        let mut eng = Self::default();
        for item in &program.items {
            match item {
                Item::TypeAlias(ta) => {
                    eng.aliases.insert(ta.name.clone(), ta.base.clone());
                }
                Item::Record(r) if r.type_params.is_empty() => {
                    eng.records.insert(r.name.clone(), r.clone());
                }
                Item::Union(u) => {
                    eng.unions.insert(u.name.clone(), u.clone());
                }
                Item::Bitfield(b) => {
                    eng.bitfields.insert(b.name.clone(), b.clone());
                }
                _ => {}
            }
        }
        eng
    }

    pub fn resolve_alias(&self, ty: &TypeExpr) -> TypeExpr {
        let mut cur = ty.clone();
        for _ in 0..8 {
            if let TypeExpr::Named { name, args, .. } = &cur {
                if args.is_empty() {
                    if let Some(base) = self.aliases.get(name) {
                        cur = base.clone();
                        continue;
                    }
                }
            }
            break;
        }
        cur
    }

    pub fn is_c_layout_record(&self, name: &str) -> bool {
        self.records
            .get(name)
            .is_some_and(|r| is_c_layout_attrs(&r.attrs))
    }

    pub fn bitfield_layout(&self, name: &str) -> Option<BitfieldLayout> {
        let bf = self.bitfields.get(name)?;
        let resolved = self.resolve_alias(&bf.backing_ty);
        let backing = match &resolved {
            TypeExpr::Named { name: bn, .. } => ScalarKind::from_name(bn)?,
            _ => return None,
        };
        let mut offset = 0u32;
        let mut fields = Vec::with_capacity(bf.fields.len());
        for f in &bf.fields {
            let mask = if f.bits >= 64 {
                u64::MAX
            } else {
                (1u64 << f.bits) - 1
            };
            fields.push(BitfieldFieldLayout {
                name: f.name.clone(),
                bit_offset: offset,
                bit_width: f.bits,
                mask,
            });
            offset += f.bits;
        }
        Some(BitfieldLayout {
            name: bf.name.clone(),
            backing,
            total_bits: offset,
            fields,
        })
    }

    pub fn size_and_align(&self, ty: &TypeExpr) -> (usize, usize) {
        self.size_and_align_depth(ty, 0)
    }

    fn size_and_align_depth(&self, ty: &TypeExpr, depth: usize) -> (usize, usize) {
        if depth > 16 {
            return (8, 8);
        }
        let resolved = self.resolve_alias(ty);
        match &resolved {
            TypeExpr::Array { elem, len, .. } => {
                let (es, ea) = self.size_and_align_depth(elem, depth + 1);
                let stride = align_up(es, ea);
                (stride * (*len), ea.max(1))
            }
            TypeExpr::Named { name, args, .. } => {
                if let Some(sk) = ScalarKind::from_name(name) {
                    return (sk.byte_size(), sk.align());
                }
                if let Some(simd) = SimdKind::from_type_expr(&resolved) {
                    return (simd.byte_size(), simd.align());
                }
                if name == "Bool" {
                    return (1, 1);
                }
                if name == "CVoid" {
                    return (0, 1);
                }
                if matches!(name.as_str(), "Ptr" | "MutPtr" | "CStr") {
                    return (8, 8);
                }
                if name == "Array" && args.len() == 2 {
                    if let Ok(n) = args[1].display_name().parse::<usize>() {
                        let (es, ea) = self.size_and_align_depth(&args[0], depth + 1);
                        let stride = align_up(es, ea);
                        return (stride * n, ea.max(1));
                    }
                }
                if let Some(bl) = self.bitfield_layout(name) {
                    return (bl.backing.byte_size(), bl.backing.align());
                }
                if let Some(ul) = self.union_layout_depth(name, depth + 1) {
                    return (ul.size, ul.align);
                }
                if let Some(sl) = self.struct_layout_depth(name, depth + 1) {
                    return (sl.size, sl.align);
                }
                (8, 8)
            }
            TypeExpr::Tuple(_, _) | TypeExpr::Dyn { .. } | TypeExpr::Fn { .. } => (8, 8),
        }
    }

    pub fn struct_layout(&self, name: &str) -> Option<StructByteLayout> {
        self.struct_layout_depth(name, 0)
    }

    fn struct_layout_depth(&self, name: &str, depth: usize) -> Option<StructByteLayout> {
        let rec = self.records.get(name)?;
        let is_packed = has_packed_attr(&rec.attrs);
        let gpu_mode = gpu_struct_mode(&rec.attrs);
        let is_c_repr = has_repr_c(&rec.attrs)
            || is_packed
            || explicit_align_attr(&rec.attrs).is_some()
            || gpu_mode.is_some();
        let explicit_align = explicit_align_attr(&rec.attrs);

        let mut offset = 0usize;
        let mut max_align = if gpu_mode.as_deref() == Some("std140") { 16 } else { 1 };
        let mut fields = Vec::with_capacity(rec.fields.len());

        for f in &rec.fields {
            let (fsize, falign) = if is_c_repr {
                self.size_and_align_depth(&f.ty, depth + 1)
            } else {
                (8, 8)
            };
            let eff_align = if is_packed { 1 } else { falign.max(1) };
            offset = align_up(offset, eff_align);
            fields.push(FieldByteLayout {
                name: f.name.clone(),
                ty: f.ty.clone(),
                byte_offset: offset,
                byte_size: fsize,
                align: eff_align,
            });
            offset += fsize;
            if eff_align > max_align {
                max_align = eff_align;
            }
        }

        if let Some(ea) = explicit_align {
            if ea > max_align {
                max_align = ea;
            }
        }
        let total_size = if is_packed && explicit_align.is_none() {
            offset
        } else {
            align_up(offset, max_align)
        };

        Some(StructByteLayout {
            name: rec.name.clone(),
            is_c_repr,
            is_packed,
            is_union: false,
            size: total_size,
            align: max_align,
            fields,
        })
    }

    pub fn union_layout(&self, name: &str) -> Option<StructByteLayout> {
        self.union_layout_depth(name, 0)
    }

    fn union_layout_depth(&self, name: &str, depth: usize) -> Option<StructByteLayout> {
        let u = self.unions.get(name)?;
        let is_packed = has_packed_attr(&u.attrs);
        let explicit_align = explicit_align_attr(&u.attrs);
        let mut max_size = 0usize;
        let mut max_align = 1usize;
        let mut fields = Vec::with_capacity(u.fields.len());

        for f in &u.fields {
            let (fsize, falign) = self.size_and_align_depth(&f.ty, depth + 1);
            let eff_align = if is_packed { 1 } else { falign.max(1) };
            fields.push(FieldByteLayout {
                name: f.name.clone(),
                ty: f.ty.clone(),
                byte_offset: 0,
                byte_size: fsize,
                align: eff_align,
            });
            if fsize > max_size {
                max_size = fsize;
            }
            if eff_align > max_align {
                max_align = eff_align;
            }
        }
        if let Some(ea) = explicit_align {
            if ea > max_align {
                max_align = ea;
            }
        }
        let total_size = if is_packed && explicit_align.is_none() {
            max_size
        } else {
            align_up(max_size, max_align)
        };

        Some(StructByteLayout {
            name: u.name.clone(),
            is_c_repr: true,
            is_packed,
            is_union: true,
            size: total_size,
            align: max_align,
            fields,
        })
    }

    pub fn offset_of(&self, type_name: &str, field_name: &str) -> Option<usize> {
        if let Some(sl) = self.struct_layout(type_name) {
            return sl
                .fields
                .iter()
                .find(|f| f.name == field_name)
                .map(|f| f.byte_offset);
        }
        if let Some(ul) = self.union_layout(type_name) {
            return ul
                .fields
                .iter()
                .find(|f| f.name == field_name)
                .map(|f| f.byte_offset);
        }
        None
    }
}
