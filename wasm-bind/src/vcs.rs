use project::JsProtect;

use crate::*;

/// Commit builder. Allows to create a commit, verify validity of the project and
/// downstream dependencies locally, if possible. And then commit the changes to the server.
///
/// On deploy commits,
/// the server will run full validations of the project and all dependencies, even those
/// that cannot be accessed by the user. In turn, server will return the validation results
/// to the client, with whether the commit was successful or was denied.
/// 
/// Server will save previous validation runs stored in cache, so even though effectively this is a
/// "full validation" run, in reality only changed things are incrementally re-validated,
/// which makes the process fast.
#[derive(Debug)]
#[wasm_bindgen(js_name = CommitBuilder)]
pub struct JsCommitBuilder {
    #[wasm_bindgen(getter_with_clone)]
    pub message: JsString,
}

#[wasm_bindgen(js_class = CommitBuilder)]
impl JsCommitBuilder {
    /// Create a new commit builder object with the given commit/deploy message.
    #[wasm_bindgen(constructor)]
    pub fn new(message: JsString) -> Self {
        Self { message }
    }

    /// Commit the project changes. If commit succeeds, returns the commit object, and makes
    /// this builder unusable (further operations will throw exceptions).
    pub async fn commit(&self) -> Result<JsCommitOutcome, JsCommitError> {
        todo!()
    }

    /// Deploy the project with the given version tag. The tag should be unique per project.
    /// Can throw exception if the tag is already used or if there are issues
    /// with the project, commit, or downstream dependencies, or server connectivity.
    ///
    /// The builder will be unusable after successful deploy
    /// (further operations will throw exceptions).
    #[wasm_bindgen(js_name = commitWithTag)]
    pub async fn deploy_with_tag(&self, tag: JsString) -> Result<JsCommitOutcome, JsCommitError> {
        todo!()
    }

    /// Get array of downstream dependencies of the project.
    /// Can throw exception if there are issues with loading of the dependencies.
    ///
    /// This should be called in order for validations to include the dependencies.
    /// Otherwise, them will be skipped locally. However, server always validates all dependencies.
    /// If dependency list is loaded, each individual dependency should in turn be loaded.
    /// Some may not be accessible due to permissions, so they effectively can be skipped and then
    /// they will not be accounted for in the validation.
    ///
    /// Will throw an exception if this builder was invalidated.
    pub async fn downstreams(&self) -> Result<Vec<JsDownstreamDependency>, JsLoadDepsError> {
        todo!()
    }

    /// Check if the builder was invalidated. It is invalidated after a successful commit or deploy.
    /// Further operations on the invalidated builder will throw exceptions.
    pub fn is_invalidated(&self) -> bool {
        todo!()
    }
}

/// A commit in the version control system.
#[derive(Debug)]
#[wasm_bindgen(js_name = Commit)]
pub struct JsCommit {
    /// The commit hash.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub hash: JsCommitHash,

    /// The tag of the commit, if it is a deploy.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub tag: Option<JsString>,

    /// For deploys, they can be yanked and
    /// if it was, this will return a hash of the commit on which this operation was carried out.
    /// Will return `undefined` if the deploy was not yanked or if the commit is not a deploy.
    #[wasm_bindgen(readonly, getter_with_clone, js_name = yankedWith)]
    pub yanked_with: Option<JsCommitHash>,

    /// Tag of the deploy which was yanked by this commit.
    /// Will return `undefined` if the commit is not a yank of a deploy.
    #[wasm_bindgen(readonly, getter_with_clone, js_name = yankOfTag)]
    pub yank_of_tag: Option<JsString>,

    /// The commit message.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub message: JsString,

    /// The commit timestamp.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub timestamp: u64,

    /// The commit author.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub author: JsString,
}

#[wasm_bindgen(js_class = Commit)]
impl JsCommit {
    /// Whether the commit is a deploy.
    #[wasm_bindgen(getter, js_name = isDeploy)]
    pub fn is_deploy(&self) -> bool {
        self.tag.is_some()
    }

    /// Whether the deploy was yanked.
    #[wasm_bindgen(getter, js_name = isYanked)]
    pub fn is_yanked(&self) -> bool {
        self.yanked_with.is_some()
    }

    /// For deploy commits, start a commit builder for a yank operation.
    /// This will return a builder object configured to yank this deploy on
    /// commit (or deploy) of the commit under construction.
    /// Will throw an exception if this commit is not a deploy and thus
    /// cannot be yanked, or if the deploy was already yanked.
    #[wasm_bindgen(js_name = buildYank)]
    pub fn build_yank(&self, reason: JsString) -> Result<JsCommitBuilder, JsError> {
        todo!()·
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

    /// The validation outcome of the project.
    #[wasm_bindgen(getter, js_name = validationOutcome)]
    pub fn validation_outcome(&self) -> JsValidationOutcome {
        todo!()
    }

    /// The validation errors of the downstream dependencies.
    /// Dependencies without errors are omitted.
    /// This is an empty array if there are no errors in any
    /// of the dependencies.
    #[wasm_bindgen(getter, js_name = downstreamErrors)]
    pub fn downstream_errors(&self) -> Vec<JsValidationOutcome> {
        todo!()
    }

    /// Whether the commit was successful. This implies no errors in the project or downstream
    /// dependencies, and the commit was accepted by the server. The builder of this commit
    /// has been invalidated if the commit was successful.
    #[wasm_bindgen(getter)]
    pub fn success(&self) -> bool {
        todo!()
    }
}

/// Outcome of a server-side validation.
/// This is returned by the server after a commit or deploy operation.
#[derive(Debug)]
#[wasm_bindgen(js_name = ValidationOutcome)]
pub struct JsValidationOutcome {
    // TODO
}

#[wasm_bindgen(js_class = ValidationOutcome)]
impl JsValidationOutcome {
    /// The name of the dependency. This is `undefined` if the dependency
    /// name cannot be exposed due to permissions.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> Option<JsString> {
        todo!()
    }

    /// The version of the dependency. This is `undefined` if the dependency
    /// version cannot be exposed due to permissions.
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> Option<JsString> {
        todo!()
    }

    /// Get the protection settings of the dependency.
    #[wasm_bindgen(getter)]
    pub fn protect(&self) -> JsProtect {
        todo!()
    }

    /// The validation errors of the dependency.
    /// May be censored if the user does not have permissions to see them or see full data
    /// in them. In such cases, this list may be useful for total count.
    ///
    /// This array is empty if there are no errors.
    #[wasm_bindgen(getter)]
    pub fn errors(&self) -> Vec<JsDownstreamLogicError> {
        todo!()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = DownstreamDependency)]
pub struct JsDownstreamDependency {
    // TODO
}

#[wasm_bindgen(js_class = DownstreamDependency)]
impl JsDownstreamDependency {
    /// The name of the dependency.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> JsString {
        todo!()
    }

    /// The version of the dependency.
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> JsString {
        todo!()
    }

    /// Get the protection settings of the dependency.
    /// This may effectively render this dependency unloadable.
    /// In this case we only know that this dependency exists, but not its contents.
    #[wasm_bindgen(getter)]
    pub fn protect(&self) -> JsProtect {
        todo!()
    }

    /// True if dependency is loadable per permissions and is significant for validation.
    /// It will be false if the user does not have the permissions.
    /// It also will be false if this downstream dependency is a dependency of a dependency which
    /// in turn is not loadable, without regard of whether former dependency
    /// is permitted to be loaded or not.
    /// Basically, this indicates whether or not this dependency should be loaded
    /// to be useful for validation, skipping any dependency chains that are interrupted due
    /// to permissions.
    #[wasm_bindgen(getter, js_name = shouldLoad)]
    pub fn should_load(&self) -> bool {
        todo!()
    }

    /// Load the dependency contents required for validation.
    /// This will fail if the user has insufficient permissions to load the dependency, or
    /// network issues prevent loading of the dependency.
    /// This is no-op if the dependency is already loaded into the work session either as a
    /// standalone project or as a dependency (without unneeded UI-related meta).
    ///
    /// This takes into account `resolveWith` setting which selects effective version
    /// that will be used for validation.
    /// Changes to the `resolveWith` setting may require reloading of the dependency.
    /// This is not always necessary, if the dependency was already loaded and cached into the
    /// working session with the required version some time in the past even without
    /// direct call to this method after the change.
    pub async fn load(&mut self) -> Result<(), JsLoadDepsError> {
        todo!()
    }

    /// Whether the contents of the dependency are loaded and could be used for validation.
    #[wasm_bindgen(getter, js_name = isLoaded)]
    pub fn is_loaded(&self) -> bool {
        todo!()
    }

    /// Validate the dependency, taking into account all relevant currently loaded dependencies
    /// in the working session. If there are any errors in the dependency, they will be returned.
    /// Empty array is returned if there are no errors.
    ///
    /// This function either runs the local validation to get the errors, or if the dependency
    /// was already validated either directly or during the validation of another dependency,
    /// it will return the cached errors immediately.
    ///
    /// This can return different errors on change to "resolveWith" mechanism and subsequent
    /// reloading of the dependency.
    ///
    /// Undefined is returned if the dependency is not loaded.
    pub async fn errors(&mut self) -> Option<Vec<JsDownstreamLogicError>> {
        todo!()
    }

    /// Set given mechanism to resolve the dependency version to select for the deployment.
    /// This will be taken into account by the server during full validation run.
    ///
    /// By default, `KeepCurrent` is used.
    /// Change of this setting may require reloading of the dependency.
    /// This can also in turn affect `errors` method result.
    #[wasm_bindgen(setter, js_name = resolveWith)]
    pub fn set_resolve_with(&mut self, resolution: JsDownstreamResolution) {
        todo!()
    }

    /// Get the selected mechanism to resolve the dependency version to select for the deployment.
    #[wasm_bindgen(getter, js_name = resolveWith)]
    pub fn get_resolve_with(&self) -> JsDownstreamResolution {
        todo!()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = LoadDepsError)]
pub struct JsLoadDepsError {
    // TODO
}

#[derive(Debug)]
#[wasm_bindgen(js_name = DownstreamLogicError)]
pub struct JsDownstreamLogicError {
    // TODO
}

/// Mechanism to select the dependency version during deployment.
#[derive(Debug, Clone, Copy)]
#[wasm_bindgen(js_name = DownstreamResolution)]
pub enum JsDownstreamResolution {
    /// Keep the current version of the dependency.
    KeepCurrent,

    /// Upgrade to the latest version of the dependency.
    UpgradeToLatest,
}

/// The history in the version control system.
#[wasm_bindgen(js_name = VcsHistory)]
pub struct JsVcsHistory {
    // TODO
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
    pub fn slice_pivot_count(&self, pivot: JsCommitHash, count: i32) -> Option<Vec<JsHistoryItem>> {
        todo!()
    }

    /// Get the latest commits in the history of the project up to the given limit.
    #[wasm_bindgen(js_name = sliceLatest)]
    pub fn slice_latest(&self, limit: u32) -> Vec<JsHistoryItem> {
        todo!()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = HistoryItem)]
pub struct JsHistoryItem {
    // TODO
}

#[wasm_bindgen(js_class = HistoryItem)]
impl JsHistoryItem {
    /// The commit hash.
    #[wasm_bindgen(getter)]
    pub fn hash(&self) -> JsCommitHash {
        todo!()
    }

    /// The commit message.
    #[wasm_bindgen(getter)]
    pub fn message(&self) -> JsString {
        todo!()
    }

    /// The commit timestamp.
    #[wasm_bindgen(getter)]
    pub fn timestamp(&self) -> u64 {
        todo!()
    }

    /// The commit author.
    #[wasm_bindgen(getter)]
    pub fn author(&self) -> JsString {
        todo!()
    }
}
