//! Reviewed content-addressed snapshots, cross-process leases, and explicit acquisition.
// Blocking filesystem work runs during startup, on the direct caller, or in spawn_blocking.
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "bounded synchronous IO is required for the direct engine and advisory file leases"
)]

#[cfg(feature = "hub")]
use std::sync::OnceLock;
use std::{
    fmt::{self, Debug, Formatter, Write as FmtWrite},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::task::spawn_blocking;

use crate::{
    Error, Result,
    encoding::ENCODING_VERSION,
    types::{CommitRevision, ModelPreset},
};

#[cfg(feature = "hub")]
mod hub;

/// Approved immutable artifact identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[non_exhaustive]
pub struct ArtifactInfo {
    /// Safe flat filename.
    pub name: String,
    /// Exact file bytes.
    pub size: u64,
    /// Reviewed SHA-256 identity.
    pub sha256: String,
    /// Upstream Git blob identity when present.
    pub git_blob: Option<String>,
}
/// Reviewed catalog manifest. Its bytes are shipped with the library.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[non_exhaustive]
pub struct Manifest {
    /// Manifest format version.
    pub schema_version: u32,
    /// Model release.
    pub preset: ModelPreset,
    /// Approved source repository.
    pub repository: String,
    /// Immutable revision.
    pub revision: CommitRevision,
    /// Locally versioned encoder identity.
    pub encoding_version: String,
    /// Complete mandatory files and reviewed identities.
    pub files: Vec<ArtifactInfo>,
}
impl Manifest {
    /// Retrieve the shipped catalog without network access.
    ///
    /// # Errors
    /// Returns an error if the embedded catalog cannot be decoded.
    pub fn catalog(preset: ModelPreset) -> Result<Self> {
        let bytes = match preset {
            ModelPreset::ClefFlash => include_bytes!("clef-flash-catalog.json").as_slice(),
        };
        let manifest: Self = serde_json::from_slice(bytes)?;
        manifest.validate()?;
        Ok(manifest)
    }
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.repository != self.preset.repository()
            || self.revision.as_str() != self.preset.revision()
            || self.encoding_version != ENCODING_VERSION
            || self.files.len() != 14
        {
            return Err(Error::IntegrityMismatch("manifest identity".into()));
        }
        let mut names = std::collections::HashSet::new();
        for f in &self.files {
            safe_name(&f.name)?;
            if f.size == 0
                || f.size > 5 * 1024 * 1024 * 1024
                || !valid_digest(&f.sha256)
                || !names.insert(&f.name)
            {
                return Err(Error::IntegrityMismatch("manifest artifact".into()));
            }
        }
        Ok(())
    }
    /// SHA-256 identity of the canonical manifest.
    ///
    /// # Errors
    /// Returns a serialization error for an invalid manifest.
    pub fn digest(&self) -> Result<String> {
        Ok(hex(&Sha256::digest(serde_json::to_vec(self)?)))
    }
}
fn valid_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub(crate) fn safe_name(s: &str) -> Result<()> {
    if s.is_empty()
        || s.len() > 128
        || s.starts_with('.')
        || !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        || s.contains("..")
    {
        return Err(Error::IntegrityMismatch("artifact filename".into()));
    }
    Ok(())
}
/// Verified immutable snapshot holding a shared cache lease.
#[derive(Clone)]
pub struct VerifiedSnapshot {
    root: PathBuf,
    manifest: Arc<Manifest>,
    digest: String,
    _lease: Arc<File>,
}
impl Debug for VerifiedSnapshot {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VerifiedSnapshot")
            .field("preset", &self.manifest.preset)
            .field("revision", &self.manifest.revision)
            .field("digest", &self.digest)
            .finish_non_exhaustive()
    }
}
impl VerifiedSnapshot {
    /// Reviewed artifact manifest.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    /// Snapshot content identity.
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }
    pub(crate) fn file(&self, name: &str) -> Result<PathBuf> {
        let f = self
            .manifest
            .files
            .iter()
            .find(|f| f.name == name)
            .ok_or(Error::ArtifactMissing)?;
        Ok(self.root.join("blobs/sha256").join(&f.sha256))
    }
    pub(crate) fn read_verified(&self, name: &str, deadline: Instant) -> Result<Vec<u8>> {
        let info = self
            .manifest
            .files
            .iter()
            .find(|f| f.name == name)
            .ok_or(Error::ArtifactMissing)?;
        let bytes = read_bounded(&self.file(name)?, info.size, deadline)?;
        if bytes.len() as u64 != info.size || hex(&Sha256::digest(&bytes)) != info.sha256 {
            return Err(Error::IntegrityMismatch("loaded artifact identity".into()));
        }
        Ok(bytes)
    }
    /// Reverify every referenced blob, without networking. Blocking disk work.
    ///
    /// # Errors
    /// Rejects missing, modified or nonregular files.
    pub fn verify(&self) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(600);
        for f in &self.manifest.files {
            verify_file(&self.file(&f.name)?, f, deadline)?;
        }
        Ok(())
    }
}
/// Snapshot cache metadata, without allocating weights.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct SnapshotInfo {
    /// Pinned model.
    pub preset: ModelPreset,
    /// Full immutable revision.
    pub revision: String,
    /// Manifest SHA-256.
    pub manifest_digest: String,
    /// Complete file bytes.
    pub bytes: u64,
}
/// Content-addressed artifact store. Global mutation locking bounds transactions
/// to one per cache across processes; inference leases never block verification.
#[derive(Debug, Clone)]
pub struct ArtifactStore {
    root: Arc<PathBuf>,
    max_bytes: u64,
    #[cfg(feature = "hub")]
    fetcher: Arc<OnceLock<Fetcher>>,
}
impl ArtifactStore {
    /// Open/create a private operator-configured cache.
    ///
    /// # Errors
    /// Returns IO errors for unavailable or unsafe cache directories.
    pub fn new(root: PathBuf, max_bytes: u64) -> Result<Self> {
        if max_bytes == 0 {
            return Err(Error::StorageLimit);
        }
        fs::create_dir_all(&root)?;
        let root = fs::canonicalize(root)?;
        for name in ["blobs", "blobs/sha256", "snapshots", "staging", "locks"] {
            let path = root.join(name);
            fs::create_dir_all(&path)?;
            let meta = fs::symlink_metadata(&path)?;
            if !meta.is_dir()
                || meta.file_type().is_symlink()
                || !fs::canonicalize(&path)?.starts_with(&root)
            {
                return Err(Error::IntegrityMismatch("cache directory".into()));
            }
        }
        Ok(Self {
            root: Arc::new(root),
            max_bytes,
            #[cfg(feature = "hub")]
            fetcher: Arc::new(OnceLock::new()),
        })
    }
    /// Stop artifact admission and join in-flight acquisition when this is the last owner.
    /// Dropping all owners also closes admission; an active transaction finishes before the actor
    /// exits.
    ///
    /// # Errors
    /// Returns an error if other store owners remain or the actor failed.
    pub async fn shutdown(self) -> Result<()> {
        #[cfg(feature = "hub")]
        {
            let cell = Arc::try_unwrap(self.fetcher).map_err(|_| Error::ShuttingDown)?;
            if let Some(fetcher) = cell.into_inner() {
                drop(fetcher.send);
                fetcher.worker.await.map_err(|_| Error::WorkerUnavailable)?;
            }
        }
        Ok(())
    }
    fn snapshot_dir(&self, preset: ModelPreset) -> PathBuf {
        self.root
            .join("snapshots")
            .join(preset.alias())
            .join(preset.revision())
    }
    fn lease_path(&self, preset: ModelPreset) -> PathBuf {
        self.root
            .join("locks")
            .join(format!("{}.lease", preset.alias()))
    }
    pub(crate) fn blob(&self, file: &ArtifactInfo) -> PathBuf {
        self.root.join("blobs/sha256").join(&file.sha256)
    }
    fn lock_path(&self, name: &str) -> PathBuf {
        self.root.join("locks").join(name)
    }
    fn open_sync(&self, preset: ModelPreset) -> Result<VerifiedSnapshot> {
        let lease_path = self.lease_path(preset);
        let lease = match File::open(&lease_path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => open_lock(&lease_path)?,
            Err(error) => return Err(error.into()),
        };
        lease
            .try_lock_shared()
            .map_err(|_| Error::WorkerUnavailable)?;
        let path = self.snapshot_dir(preset).join("manifest.json");
        if !path.try_exists()? {
            return Err(Error::ArtifactMissing);
        }
        let bytes = read_bounded(&path, 64 * 1024, Instant::now() + Duration::from_secs(30))?;
        let actual: Manifest = serde_json::from_slice(&bytes)?;
        let expected = Manifest::catalog(preset)?;
        if actual.digest()? != expected.digest()? {
            return Err(Error::IntegrityMismatch("published manifest".into()));
        }
        let snapshot = VerifiedSnapshot {
            root: (*self.root).clone(),
            digest: actual.digest()?,
            manifest: Arc::new(actual),
            _lease: Arc::new(lease),
        };
        snapshot.verify()?;
        Ok(snapshot)
    }
    /// Open a committed snapshot and verify all files. Never performs network IO.
    ///
    /// # Errors
    /// Returns `ArtifactMissing` on cache miss, or an integrity error on corruption.
    pub async fn open(&self, preset: ModelPreset) -> Result<VerifiedSnapshot> {
        let store = self.clone();
        spawn_blocking(move || store.open_sync(preset))
            .await
            .map_err(|_| Error::WorkerUnavailable)?
    }
    /// Explicitly acquire the complete pinned release; offline builds reject it.
    ///
    /// # Errors
    /// Returns acquisition, integrity, deadline or storage errors.
    pub async fn fetch(&self, preset: ModelPreset) -> Result<VerifiedSnapshot> {
        #[cfg(feature = "hub")]
        {
            let fetcher = self.fetcher.get_or_init(|| {
                let (send, receiver) = tokio::sync::mpsc::channel(16);
                let worker_store = Self {
                    root: self.root.clone(),
                    max_bytes: self.max_bytes,
                    fetcher: Arc::new(OnceLock::new()),
                };
                let worker = tokio::spawn(fetch_actor(worker_store, receiver));
                Fetcher { send, worker }
            });
            let (reply, result) = tokio::sync::oneshot::channel();
            fetcher
                .send
                .try_send(FetchRequest { preset, reply })
                .map_err(|e| match e {
                    tokio::sync::mpsc::error::TrySendError::Full(_) => Error::QueueFull,
                    tokio::sync::mpsc::error::TrySendError::Closed(_) => Error::WorkerUnavailable,
                })?;
            result.await.map_err(|_| Error::WorkerUnavailable)?
        }
        #[cfg(not(feature = "hub"))]
        {
            let _ = preset;
            Err(Error::UnsupportedCapability(
                "hub feature is disabled".into(),
            ))
        }
    }
    /// Import a local release by copying reviewed bytes into private storage.
    /// Symlinks and unknown files are rejected, and no adjacent manifest is trusted.
    ///
    /// # Errors
    /// Returns integrity, storage or IO errors.
    pub async fn import(&self, path: PathBuf, preset: ModelPreset) -> Result<VerifiedSnapshot> {
        let store = self.clone();
        spawn_blocking(move || {
            let _lock = exclusive_lock(&store.lock_path("allocation"))?;
            let catalog = Manifest::catalog(preset)?;
            let deadline = Instant::now() + Duration::from_secs(7200);
            store.reserve(&catalog)?;
            let source = fs::canonicalize(path)?;
            for entry in fs::read_dir(&source)? {
                let entry = entry?;
                let name = entry.file_name();
                if !catalog
                    .files
                    .iter()
                    .any(|f| name == std::ffi::OsStr::new(&f.name))
                    || !entry.file_type()?.is_file()
                {
                    return Err(Error::IntegrityMismatch(
                        "import contains unreviewed entries".into(),
                    ));
                }
            }
            for f in &catalog.files {
                let input = source.join(&f.name);
                verify_file(&input, f, deadline)?;
                if store.blob(f).try_exists()? {
                    verify_file(&store.blob(f), f, deadline)?;
                    continue;
                }
                let partial = store
                    .root
                    .join("staging")
                    .join(format!("{}.partial", f.sha256));
                if let Ok(meta) = fs::symlink_metadata(&partial)
                    && (!meta.is_file() || meta.file_type().is_symlink())
                {
                    return Err(Error::IntegrityMismatch("import staging entry".into()));
                }
                let mut input = File::open(input)?;
                let mut output = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&partial)?;
                let mut buffer = vec![0; 1024 * 1024];
                let mut total = 0_u64;
                loop {
                    if Instant::now() >= deadline {
                        return Err(Error::DeadlineExceeded);
                    }
                    let n = input.read(&mut buffer)?;
                    if n == 0 {
                        break;
                    }
                    total = total.checked_add(n as u64).ok_or(Error::StorageLimit)?;
                    if total > f.size {
                        return Err(Error::IntegrityMismatch("import size".into()));
                    }
                    output.write_all(buffer.get(..n).ok_or(Error::StorageLimit)?)?;
                }
                output.sync_all()?;
                verify_file(&partial, f, deadline)?;
                publish_blob(&partial, &store.blob(f))?;
            }
            store.publish(&catalog)?;
            store.open_sync(preset)
        })
        .await
        .map_err(|_| Error::WorkerUnavailable)?
    }
    fn reserve(&self, catalog: &Manifest) -> Result<()> {
        let mut missing = 0_u64;
        for f in &catalog.files {
            if !self.blob(f).try_exists()? {
                let partial = self
                    .root
                    .join("staging")
                    .join(format!("{}.partial", f.sha256));
                let partial_bytes = match fs::symlink_metadata(partial) {
                    Ok(meta)
                        if meta.is_file()
                            && !meta.file_type().is_symlink()
                            && meta.len() <= f.size =>
                    {
                        meta.len()
                    }
                    Ok(_) => return Err(Error::IntegrityMismatch("partial file".into())),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
                    Err(error) => return Err(error.into()),
                };
                missing = missing
                    .checked_add(f.size.saturating_sub(partial_bytes))
                    .ok_or(Error::StorageLimit)?;
            }
        }
        let used = directory_bytes(&self.root.join("blobs/sha256"))?
            .checked_add(directory_bytes(&self.root.join("staging"))?)
            .ok_or(Error::StorageLimit)?;
        if used.checked_add(missing).ok_or(Error::StorageLimit)? > self.max_bytes
            || fs4::available_space(&*self.root)? < missing.saturating_add(1024 * 1024 * 1024)
        {
            return Err(Error::StorageLimit);
        }
        Ok(())
    }
    fn publish(&self, catalog: &Manifest) -> Result<()> {
        let final_dir = self.snapshot_dir(catalog.preset);
        if final_dir.try_exists()? {
            return Ok(());
        }
        let stage = self
            .root
            .join("staging")
            .join(format!("{}-snapshot", catalog.preset.alias()));
        fs::create_dir_all(&stage)?;
        let path = stage.join("manifest.json");
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        file.write_all(&serde_json::to_vec(catalog)?)?;
        file.sync_all()?;
        let parent = final_dir.parent().ok_or(Error::ArtifactMissing)?;
        fs::create_dir_all(parent)?;
        fs::rename(stage, &final_dir)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    }
    /// List the two supported committed releases; no Hub enumeration.
    ///
    /// # Errors
    /// Returns errors for damaged manifests.
    pub async fn list(&self) -> Result<Vec<SnapshotInfo>> {
        let store = self.clone();
        spawn_blocking(move || {
            let mut out = Vec::new();
            {
                let preset = ModelPreset::ClefFlash;
                if store
                    .snapshot_dir(preset)
                    .join("manifest.json")
                    .try_exists()?
                {
                    let expected = Manifest::catalog(preset)?;
                    let bytes = read_bounded(
                        &store.snapshot_dir(preset).join("manifest.json"),
                        64 * 1024,
                        Instant::now() + Duration::from_secs(30),
                    )?;
                    let m: Manifest = serde_json::from_slice(&bytes)?;
                    if m.digest()? != expected.digest()? {
                        return Err(Error::IntegrityMismatch("published manifest".into()));
                    }
                    out.push(SnapshotInfo {
                        preset,
                        revision: preset.revision().into(),
                        manifest_digest: m.digest()?,
                        bytes: m.files.iter().map(|f| f.size).sum(),
                    });
                }
            }
            Ok(out)
        })
        .await
        .map_err(|_| Error::WorkerUnavailable)?
    }
    /// Prune a selected unleased release and unreachable blobs. Dry run is read-only.
    ///
    /// # Errors
    /// Refuses active engine leases, unsafe cache entries and IO failures.
    pub async fn prune(&self, preset: ModelPreset, dry_run: bool) -> Result<u64> {
        let store = self.clone();
        spawn_blocking(move || {
            let _allocation = exclusive_lock(&store.lock_path("allocation"))?;
            let _lease = exclusive_lock(&store.lease_path(preset))?;
            let mut freed = 0_u64;
            if !dry_run && store.snapshot_dir(preset).try_exists()? {
                fs::remove_file(store.snapshot_dir(preset).join("manifest.json"))?;
                fs::remove_dir(store.snapshot_dir(preset))?;
            }
            for f in Manifest::catalog(preset)?.files {
                if store.blob(&f).try_exists()? {
                    let metadata = fs::symlink_metadata(store.blob(&f))?;
                    if !metadata.is_file() || metadata.file_type().is_symlink() {
                        return Err(Error::IntegrityMismatch("cache entry".into()));
                    }
                    freed = freed
                        .checked_add(metadata.len())
                        .ok_or(Error::StorageLimit)?;
                    if !dry_run {
                        fs::remove_file(store.blob(&f))?;
                    }
                }
            }
            Ok(freed)
        })
        .await
        .map_err(|_| Error::WorkerUnavailable)?
    }
}
fn open_lock(path: &Path) -> Result<File> {
    Ok(OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?)
}
fn exclusive_lock(path: &Path) -> Result<File> {
    let file = open_lock(path)?;
    file.try_lock().map_err(|_| Error::WorkerUnavailable)?;
    Ok(file)
}
fn directory_bytes(path: &Path) -> Result<u64> {
    let mut total = 0_u64;
    for entry in fs::read_dir(path)? {
        let metadata = entry?.metadata()?;
        if metadata.is_file() {
            total = total
                .checked_add(metadata.len())
                .ok_or(Error::StorageLimit)?;
        }
    }
    Ok(total)
}
pub(crate) fn read_bounded(path: &Path, cap: u64, deadline: Instant) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > cap {
        return Err(Error::IntegrityMismatch("artifact size/type".into()));
    }
    let size = usize::try_from(metadata.len()).map_err(|_| Error::StorageLimit)?;
    let mut out = Vec::with_capacity(size);
    let mut file = File::open(path)?;
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        if Instant::now() >= deadline {
            return Err(Error::DeadlineExceeded);
        }
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        if out.len().checked_add(n).ok_or(Error::StorageLimit)? > size {
            return Err(Error::IntegrityMismatch("artifact grew".into()));
        }
        out.extend_from_slice(buffer.get(..n).ok_or(Error::StorageLimit)?);
    }
    if out.len() != size {
        return Err(Error::IntegrityMismatch("truncated artifact".into()));
    }
    Ok(out)
}
pub(crate) fn verify_file(path: &Path, expected: &ArtifactInfo, deadline: Instant) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() != expected.size {
        return Err(Error::IntegrityMismatch("artifact size/type".into()));
    }
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        if Instant::now() >= deadline {
            return Err(Error::DeadlineExceeded);
        }
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        total = total.checked_add(n as u64).ok_or(Error::StorageLimit)?;
        if total > expected.size {
            return Err(Error::IntegrityMismatch("artifact grew".into()));
        }
        hash.update(buffer.get(..n).ok_or(Error::StorageLimit)?);
    }
    if total != expected.size || hex(&hash.finalize()) != expected.sha256 {
        return Err(Error::IntegrityMismatch("artifact digest".into()));
    }
    Ok(())
}
fn publish_blob(partial: &Path, blob: &Path) -> Result<()> {
    match fs::hard_link(partial, blob) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e.into()),
    }
    fs::remove_file(partial)?;
    File::open(blob)?.sync_all()?;
    if let Some(parent) = blob.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_should_verify_flash_catalog_and_reject_paths() -> Result<()> {
        {
            let p = ModelPreset::ClefFlash;
            let c = Manifest::catalog(p)?;
            assert_eq!(c.revision.as_str(), p.revision());
            assert_eq!(c.digest()?.len(), 64);
        }
        for name in ["../weights", "/tmp/weights", "a/b", "a\\b", "..", "a\0b"] {
            assert!(safe_name(name).is_err());
        }
        Ok(())
    }
    #[tokio::test]
    async fn test_should_fail_offline_without_network() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = ArtifactStore::new(directory.path().into(), 1024)?;
        assert!(matches!(
            store.open(ModelPreset::ClefFlash).await,
            Err(Error::ArtifactMissing)
        ));
        assert!(store.list().await?.is_empty());
        Ok(())
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(
        String::with_capacity(bytes.len() * 2),
        |mut output, byte| {
            let _ = write!(output, "{byte:02x}");
            output
        },
    )
}

#[cfg(test)]
mod fault_tests {
    use super::*;
    #[test]
    fn test_should_reject_truncation_wrong_digest_and_symlinks() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("file");
        fs::write(&path, b"weights")?;
        let artifact = ArtifactInfo {
            name: "weights".into(),
            size: 7,
            sha256: hex(&Sha256::digest(b"weights")),
            git_blob: None,
        };
        let deadline = Instant::now() + Duration::from_secs(10);
        verify_file(&path, &artifact, deadline)?;
        fs::write(&path, b"weightz")?;
        assert!(matches!(
            verify_file(&path, &artifact, deadline),
            Err(Error::IntegrityMismatch(_))
        ));
        fs::write(&path, b"weight")?;
        assert!(matches!(
            verify_file(&path, &artifact, deadline),
            Err(Error::IntegrityMismatch(_))
        ));
        assert!(read_bounded(&path, 2, deadline).is_err());
        #[cfg(unix)]
        {
            let link = directory.path().join("link");
            std::os::unix::fs::symlink(&path, &link)?;
            assert!(verify_file(&link, &artifact, deadline).is_err());
        }
        Ok(())
    }
    #[test]
    fn test_should_keep_exclusive_cache_lock_until_owner_releases() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("lock");
        let first = exclusive_lock(&path)?;
        assert!(exclusive_lock(&path).is_err());
        drop(first);
        exclusive_lock(&path)?;
        Ok(())
    }
}

#[cfg(feature = "hub")]
#[derive(Debug)]
struct FetchRequest {
    preset: ModelPreset,
    reply: tokio::sync::oneshot::Sender<Result<VerifiedSnapshot>>,
}
#[cfg(feature = "hub")]
#[derive(Debug)]
struct Fetcher {
    send: tokio::sync::mpsc::Sender<FetchRequest>,
    worker: tokio::task::JoinHandle<()>,
}
#[cfg(feature = "hub")]
async fn fetch_actor(
    store: ArtifactStore,
    mut requests: tokio::sync::mpsc::Receiver<FetchRequest>,
) {
    while let Some(first) = requests.recv().await {
        let preset = first.preset;
        let mut subscribers = vec![first.reply];
        let operation = hub::fetch(&store, preset);
        tokio::pin!(operation);
        let result = loop {
            tokio::select! {
                next=requests.recv(),if subscribers.len()<16 && !requests.is_closed()=>{
                    if let Some(next)=next{subscribers.push(next.reply);}
                },
                result=&mut operation=>break result,
            }
        };
        for subscriber in subscribers {
            let _ = subscriber.send(result.clone());
        }
    }
}
