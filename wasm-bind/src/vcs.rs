use compact_str::CompactString;
use project::JsProtect;
use uuid::Uuid;

use crate::*;

/// Commit builder. Allows to create a commit, verify validity of the project and
/// dependencies locally, if possible. And then commit the changes to the server.
///
/// On deploy commits,
/// the server will run full validations of the project and all other projects that depend on it.
/// Even those, that cannot be accessed by the user.
///
/// The API of WorkSession
/// allows to retrieve dependencies that are accessible by the user and validate them
/// locally. This is optional, but can provide some insights into potential issues with the project
/// before attempting to deploy. Inaccessible dependencies will be omitted from the list and
/// will be validated only on the server.
///
/// All project that are using this one as their dependency and are configured to automatically
/// update to the latest version, will be updated to the new version of this project.
/// Update is not performed if the validation of the updated resulting project fails,
/// then such dependencies will be held back until either the issues are resolved
/// or some user later updates those projects manually to be compatible
/// with the new version of this deployed project's version. If this happens, it is possible
/// to retrieve a list of such dependencies and their errors from the server. Protected
/// unaccessible dependencies will be omitted from the list and will be mentioned
/// in an obscured form without sensitive information.
///
/// Server will save previous validation runs stored in cache, so even though effectively this is a
/// "full validation" run, in reality only changed things are incrementally re-validated,
/// which should make the process fast.
#[derive(Debug)]
#[wasm_bindgen(js_name = CommitBuilder)]
pub struct JsCommitBuilder {
    #[wasm_bindgen(getter_with_clone)]
    pub message: JsString,

    /// Additional metadata to be associated with the commit.
    /// This, for example, may contain the user id or username.
    #[wasm_bindgen(getter_with_clone)]
    pub meta: JsValue,
}

#[wasm_bindgen(js_class = CommitBuilder)]
impl JsCommitBuilder {
    /// Create a new commit builder object with the given commit/deploy message.
    #[wasm_bindgen(constructor)]
    pub fn new(message: JsString) -> Self {
        Self {
            message,
            meta: JsValue::NULL,
        }
    }

    /// Commit the project changes. If commit succeeds, returns the commit object, and makes
    /// this builder unusable (further operations will throw exceptions).
    /// If new commit is to be made, new builder instance should be created.
    pub async fn commit(&self) -> Result<JsCommitOutcome, JsCommitError> {
        todo!()
    }

    /// Deploy the project with the given version. The version should be unique per project
    /// and should increment with each deploy.
    /// Can throw exception if the version is already used or if there are issues
    /// with the project, commit, server connectivity. This is rejected if
    /// tag violates version precedence rules - it is not allowed to deploy a version
    /// that is not greater than the latest deployed version.
    ///
    /// The builder will be unusable after a successful deploy
    /// (further operations will throw exceptions).
    #[wasm_bindgen(js_name = commitWithTag)]
    pub async fn deploy(&self, version: JsVersion) -> Result<JsCommitOutcome, JsCommitError> {
        todo!()
    }

    /// Check if the builder was invalidated. It is invalidated after a successful commit or deploy.
    /// Further operations on the invalidated builder will throw exceptions.
    #[wasm_bindgen(getter, js_name = isInvalidated)]
    pub fn is_invalidated(&self) -> bool {
        todo!()
    }

    /// This will do the same server-side validation as those that are made when the deploy
    /// is submitted. No actual changes are made, but this will generate the lists of
    /// errors for reverse and direct dependencies.
    #[wasm_bindgen(js_name = validateOnly)]
    pub fn validate_only(&self) -> Result<JsValidationOutcome, JsCommitError> {
        todo!()
    }
}

/// A commit copy from the version control system.
/// Note that if the commit was changed (e.g. yanked), this will not reflect the changes.
/// The commit should be reloaded for the latest information.
#[derive(Debug)]
#[wasm_bindgen(js_name = Commit)]
pub struct JsCommit {
    /// The commit hash.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub hash: JsCommitHash,

    /// The version of the commit, if it is a deploy.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub version: Option<JsVersion>,

    /// For deploy, it can be yanked and
    /// if it was, this will return a hash of the commit on which this operation was carried out.
    /// Will return `undefined` if the deploy was not yanked or if the commit is not a deploy.
    #[wasm_bindgen(readonly, getter_with_clone, js_name = yankedWith)]
    pub yanked_with: Option<JsCommitHash>,

    /// Version of the deploy which was yanked by this commit.
    /// Will return `undefined` if the commit is not a yank of a deploy.
    #[wasm_bindgen(readonly, getter_with_clone, js_name = yankOfTag)]
    pub yank_of_version: Option<JsVersion>,

    /// The commit message.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub message: JsString,

    /// The commit timestamp.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub timestamp: u64,

    /// The additional metadata of the commit that was associated
    /// by the client.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub meta: JsValue,
}

#[wasm_bindgen(js_class = Commit)]
impl JsCommit {
    /// Whether the commit is a deploy.
    #[wasm_bindgen(getter, js_name = isDeploy)]
    pub fn is_deploy(&self) -> bool {
        self.version.is_some()
    }

    /// Whether the deploy was yanked.
    #[wasm_bindgen(getter, js_name = isYanked)]
    pub fn is_yanked(&self) -> bool {
        self.yanked_with.is_some()
    }

    /// Whether the deploy is not a yank of deploy and not a deploy.
    #[wasm_bindgen(getter, js_name = isRegular)]
    pub fn is_regular(&self) -> bool {
        !self.is_deploy() && !self.is_yanked()
    }

    /// For deploy commits, start a commit builder for a yank operation.
    /// This will return a builder object configured to yank this deploy on
    /// commit (or deploy) of the commit under construction.
    /// Will throw an exception if this commit is not a deploy and thus
    /// cannot be yanked, or if the deploy was already yanked.
    #[wasm_bindgen(js_name = buildYank)]
    pub fn build_yank(&self, reason: JsString) -> Result<JsCommitBuilder, JsError> {
        todo!();
    }
}

/// Hash of a commit.
#[derive(Debug, Clone)]
#[wasm_bindgen(js_name = CommitHash)]
pub struct JsCommitHash {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub hash: JsString,
}

/// Error during commit submission. Note, this error does not rise due to failed validation.
/// This error is thrown when the server cannot accept the commit for some reason, or
/// when there are network issues.
#[derive(Debug)]
#[wasm_bindgen(js_name = CommitError)]
pub struct JsCommitError {
    // TODO
}

/// The result of a commit or deploy operation.
#[derive(Debug)]
#[wasm_bindgen(js_name = CommitOutcome)]
pub struct JsCommitOutcome {
    // TODO
}

#[wasm_bindgen(js_class = CommitOutcome)]
impl JsCommitOutcome {
    /// The commit object. This is `undefined` if the commit was not successful.
    #[wasm_bindgen(getter)]
    pub fn commit(&self) -> Option<JsCommit> {
        todo!()
    }

    /// The validation outcome of the project. This comes from the server and
    /// thus accounts for all dependencies, including those that are not accessible
    /// by the user. Sensitive information that cannot be viewed by the user
    /// will be obscured or omitted as applicable.
    #[wasm_bindgen(getter, js_name = validationOutcome)]
    pub fn validation_outcome(&self) -> JsValidationOutcome {
        todo!()
    }

    /// The validation errors of the reverse dependencies.
    /// Dependencies without errors are omitted.
    /// This is an empty array if there are no errors in any
    /// of the dependencies.
    #[wasm_bindgen(getter, js_name = revdepErrors)]
    pub fn revdep_errors(&self) -> Vec<JsValidationOutcome> {
        todo!()
    }

    /// Whether the commit was successful. This implies no errors in the project or reverse
    /// dependencies, and the commit was accepted by the server. The builder of this commit
    /// has been invalidated if the commit was successful.
    #[wasm_bindgen(getter, js_name = isSuccess)]
    pub fn is_success(&self) -> bool {
        todo!()
    }
}

/// Outcome of a server-side validation.
/// This is returned by the server after a commit or deploy operation attempt.
/// This object describes such outcome for a single dependency, and is a part of
/// some collection (array) of such outcomes collected by the server.
#[derive(Debug)]
#[wasm_bindgen(js_name = ValidationOutcome)]
pub struct JsValidationOutcome {
    // TODO
}

#[wasm_bindgen(js_class = ValidationOutcome)]
impl JsValidationOutcome {
    /// The project that was validated. This is `undefined` if the project cannot be exposed
    /// due to permissions.
    #[wasm_bindgen(getter)]
    pub fn project(&self) -> Option<JsDependency> {
        todo!()
    }

    /// The validation errors of the project.
    /// May be censored if the user does not have permissions to see them or see full data
    /// in them. In such cases, this list may be useful for total count.
    ///
    /// This array is empty if there are no errors.
    #[wasm_bindgen(getter)]
    pub fn errors(&self) -> Vec<JsDepLogicError> {
        todo!()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = Dependency)]
pub struct JsDependency {
    // TODO
}

#[wasm_bindgen(js_class = Dependency)]
impl JsDependency {
    /// The name of the dependency. Can be empty if hidden.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> JsString {
        todo!()
    }

    /// The version of the dependency.
    /// Can be "0.0.0" if hidden.
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> JsVersion {
        todo!()
    }

    /// Get the protection settings of the dependency.
    /// This may effectively render this dependency unloadable.
    /// In this case we only know that this dependency exists, but not its contents.
    #[wasm_bindgen(getter)]
    pub fn protect(&self) -> JsProtect {
        todo!()
    }

    /// Get the project id of the dependency.
    /// This is `undefined` if the project id cannot be exposed due to permissions.
    #[wasm_bindgen(getter, js_name = projectId)]
    pub fn project_id(&self) -> Option<js_sys::Uint8Array> {
        todo!()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = LoadDepsError)]
pub struct JsLoadDepsError {
    // TODO
}

#[derive(Debug)]
#[wasm_bindgen(js_name = DepLogicError)]
pub struct JsDepLogicError {
    // TODO
}

/// Mechanism to select the dependency version during deployment.
#[derive(Debug, Clone)]
#[wasm_bindgen(js_name = VersionResolution)]
pub struct JsVersionResolution {
    inner: VersionResolution,
}

#[derive(Debug, Clone)]
pub enum VersionResolution {
    Exact(JsVersion),
    Latest,
}

#[wasm_bindgen(js_class = VersionResolution)]
impl JsVersionResolution {
    /// Use the specified version of the dependency.
    pub fn exact(version: JsVersion) -> Self {
        Self {
            inner: VersionResolution::Exact(version),
        }
    }

    /// Always attempt to upgrade to the latest version of the dependency.
    /// If this will cause validation failure, such update will not be performed.
    /// If update will be performed, corresponding commit with new version will
    /// automatically be created in the dependency project.
    pub fn latest() -> Self {
        Self {
            inner: VersionResolution::Latest,
        }
    }
}

/// The history in the version control system.
#[wasm_bindgen(js_name = VcsHistory)]
pub struct JsVcsHistory {
    project_uuid: Uuid,
}

#[wasm_bindgen(js_class = VcsHistory)]
impl JsVcsHistory {
    /// Get commits in the history of the project from given commit.
    /// Count argument specifies how many commits to move back or forward from the pivot commit.
    /// If count is negative, moves back (into the past),
    /// if positive, moves forward (into the future).
    ///
    /// If the pivot commit is not found, returns `undefined`.
    #[wasm_bindgen(js_name = slicePivotCount)]
    pub fn slice_pivot_count(&self, pivot: JsCommitHash, count: i32) -> Option<Vec<JsCommit>> {
        todo!()
    }

    /// Get the latest commits in the history of the project up to the given limit.
    #[wasm_bindgen(js_name = sliceLatest)]
    pub fn slice_latest(&self, limit: u32) -> Vec<JsCommit> {
        todo!()
    }
}

/// The version in the VCS.
#[derive(Debug, Clone, Copy)]
#[wasm_bindgen(js_name = Version)]
pub struct JsVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

#[wasm_bindgen(js_class = Version)]
impl JsVersion {
    /// Check if this is a valid version. Invalid is "0.0.0".
    #[wasm_bindgen(getter, js_name = isValid)]
    pub fn is_valid(&self) -> bool {
        !(self.major == 0 && self.minor == 0 && self.patch == 0)
    }
}
