use base::table_data;
use js_sys::{Int8Array, Object};
use smallvec::SmallVec;

use crate::*;

pub struct Project {
    name: InterString,

    data: SmallVec<[table_data::Table; 1]>,
    files: SmallVec<[File; 1]>,
}

pub struct File {
    name: InterString,

    /// The file's contents. This is `None` if the file is not loaded into memory.
    bytes: Option<Vec<u8>>,

    /// Whether the file can be loaded into memory.
    can_load: bool,

    protect: Protect,
}

impl File {
    pub fn name(&self) -> &InterString {
        &self.name
    }

    pub fn bytes(&self) -> Option<&[u8]> {
        self.bytes.as_deref()
    }

    pub fn can_load(&self) -> bool {
        self.can_load || self.bytes.is_some()
    }

    pub fn protect(&self) -> Protect {
        self.protect
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = File)]
pub struct JsFile {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub name: JsString,

    #[wasm_bindgen(readonly)]
    pub id: i32,

    is_loaded: bool,
    can_load: bool,

    #[wasm_bindgen(readonly)]
    pub protect: JsProtect,
}

#[wasm_bindgen]
impl JsFile {
    #[wasm_bindgen(getter = canLoad)]
    pub fn can_load(&self) -> bool {
        self.can_load
    }

    #[wasm_bindgen(getter = isLoaded)]
    pub fn is_loaded(&self) -> bool {
        self.is_loaded
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Protect {
    read: ProtectLevel,

    /// Whether the resource can be written to.
    /// Transitively, this implies that the resource can be deleted.
    write: ProtectLevel,
}

impl Protect {
    pub fn new(read: ProtectLevel, write: ProtectLevel) -> Self {
        Self { read, write }
    }

    pub fn read(&self) -> ProtectLevel {
        self.read
    }

    pub fn write(&self) -> ProtectLevel {
        self.write
    }

    pub fn delete(&self) -> ProtectLevel {
        self.write
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[wasm_bindgen]
pub struct JsProtect {
    read: ProtectLevel,
    write: ProtectLevel,
}

#[wasm_bindgen]
impl JsProtect {
    pub fn read(&self) -> ProtectLevel {
        self.read
    }

    pub fn write(&self) -> ProtectLevel {
        self.write
    }

    pub fn delete(&self) -> ProtectLevel {
        self.write
    }
}

impl From<Protect> for JsProtect {
    fn from(protect: Protect) -> Self {
        Self {
            read: protect.read,
            write: protect.write,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[wasm_bindgen]
pub enum ProtectLevel {
    /// Owner and higher ranked users are permitted (employers, trusts, admins...).
    HigherRank,

    /// No restrictions.
    ///
    /// More restrictions can still apply transitively e.g.
    /// if project itself is protected
    /// then child resource is protected. Such transitivety does not make sense
    /// for normal front-end usecase (user won't see files in project, as they cannot open it),
    /// but this describes back-end security logic, which should forbid direct requests for files
    /// as well as project.
    Relax,
}

#[derive(Debug)]
pub struct FileBuilder {
    name: InterString,
    protect_read: ProtectLevel,
    protect_delete: ProtectLevel,
    bytes: Int8Array,
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = FileBuilder)]
    pub type JsFileBuilder;
}

macro_rules! get {
    ($val:ident, $name:expr) => {{
        wasm_bindgen::intern($name);
        let js_str = &JsString::from($name);
        if let Some(val) = Object::try_from(&$val) {
            let val = Object::get_own_property_descriptor(&val, js_str);
            if val.is_undefined() {
                Err(TypeReconstructionError::MissingField($name))
            } else {
                Ok(val)
            }
        } else {
            Err(TypeReconstructionError::InvalidType($name))
        }
    }};
}

macro_rules! intern_str {
    ($name:expr) => {{
        wasm_bindgen::intern($name);
        JsString::from($name)
    }};
}

impl FileBuilder {
    pub fn from_js(obj: JsFileBuilder) -> Result<Self, TypeReconstructionError> {
        let name = TypeReconstructionError::ensure_string(get!(obj, "name")?)?;
        let protect_read = TypeReconstructionError::ensure_string(get!(obj, "protectRead")?)?;
        let protect_delete = TypeReconstructionError::ensure_string(get!(obj, "protectDelete")?)?;
        let bytes = get!(obj, "bytes")?;

        let relax = intern_str!("relax");
        let higher_rank = intern_str!("higherRank");

        let protect_read = if protect_read == relax {
            ProtectLevel::Relax
        } else if protect_read == higher_rank {
            ProtectLevel::HigherRank
        } else {
            return Err(TypeReconstructionError::InvalidType("protectRead"));
        };

        let protect_delete = if protect_delete == relax {
            ProtectLevel::Relax
        } else if protect_delete == higher_rank {
            ProtectLevel::HigherRank
        } else {
            return Err(TypeReconstructionError::InvalidType("protectDelete"));
        };

        if name.length() > 255 {
            return Err(TypeReconstructionError::InvalidType("name"));
        }

        Ok(Self {
            name: InterString::new_js(name),
            protect_read,
            protect_delete,
            bytes: bytes
                .dyn_into()
                .map_err(|_| TypeReconstructionError::InvalidType("ArrayBuffer"))?,
        })
    }
}

impl From<FileBuilder> for File {
    fn from(builder: FileBuilder) -> Self {
        Self {
            name: builder.name,
            bytes: Some(vec_i8_into_u8(builder.bytes.to_vec())),
            can_load: true,
            protect: Protect::new(builder.protect_read, builder.protect_delete),
        }
    }
}
