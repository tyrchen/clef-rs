//! Restricted pinned HTTPS transport with DNS pinning and reviewed redirects.
use std::{
    net::IpAddr,
    time::{Duration, Instant},
};

use reqwest::{Client, StatusCode, redirect::Policy};
use secrecy::{ExposeSecret, SecretString};
use tokio::{
    io::AsyncWriteExt,
    net::lookup_host,
    task::spawn_blocking,
    time::{sleep, timeout},
};
use url::Url;

use super::{
    ArtifactInfo, ArtifactStore, Manifest, VerifiedSnapshot, exclusive_lock, publish_blob,
    verify_file,
};
use crate::{Error, Result, types::ModelPreset};

fn approved_host(host: &str) -> bool {
    matches!(
        host,
        "huggingface.co"
            | "us.aws.cdn.hf.co"
            | "cdn-lfs.huggingface.co"
            | "cdn-lfs-us-1.hf.co"
            | "cdn-lfs-eu-1.hf.co"
            | "cas-bridge.xethub.hf.co"
    )
}
#[allow(
    clippy::nonminimal_bool,
    reason = "explicit address rejection predicates are easier to security-review"
)]
fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            !ip.is_private()
                && !ip.is_loopback()
                && !ip.is_link_local()
                && !ip.is_multicast()
                && !ip.is_unspecified()
                && !ip.is_broadcast()
                && !ip.is_documentation()
                && ip.octets().first().copied().unwrap_or_default() != 0
                && ip.octets().first().copied().unwrap_or_default() < 224
                && !(ip.octets().first() == Some(&100)
                    && (64..=127).contains(ip.octets().get(1).unwrap_or(&0)))
        }
        IpAddr::V6(ip) => ip.to_ipv4_mapped().map_or_else(
            || {
                let s = ip.segments();
                !ip.is_loopback()
                    && !ip.is_unspecified()
                    && (s.first().copied().unwrap_or_default() & 0xe000) == 0x2000
                    && !(s.first() == Some(&0x2001) && s.get(1) == Some(&0x0db8))
            },
            |ip| public_ip(IpAddr::V4(ip)),
        ),
    }
}
async fn client(url: &Url) -> Result<Client> {
    let host = url
        .host_str()
        .ok_or_else(|| Error::IntegrityMismatch("download host".into()))?;
    if url.scheme() != "https"
        || url.port_or_known_default() != Some(443)
        || !url.username().is_empty()
        || url.password().is_some()
        || !approved_host(host)
    {
        return Err(Error::IntegrityMismatch("download origin".into()));
    }
    let addresses = timeout(Duration::from_secs(10), lookup_host((host, 443)))
        .await
        .map_err(|_| Error::DeadlineExceeded)??
        .collect::<Vec<_>>();
    if addresses.is_empty() || addresses.iter().any(|a| !public_ip(a.ip())) {
        return Err(Error::IntegrityMismatch("download DNS address".into()));
    }
    Ok(Client::builder()
        .no_proxy()
        .redirect(Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .read_timeout(Duration::from_secs(30))
        .timeout(Duration::from_mins(30))
        .resolve_to_addrs(host, &addresses)
        .build()?)
}
async fn request(
    mut url: Url,
    offset: u64,
    token: Option<&SecretString>,
) -> Result<reqwest::Response> {
    for _ in 0..6 {
        let client = client(&url).await?;
        let mut builder = client
            .get(url.clone())
            .header("Accept-Encoding", "identity");
        if offset > 0 {
            builder = builder.header("Range", format!("bytes={offset}-"));
        }
        if url.host_str() == Some("huggingface.co")
            && let Some(token) = token
        {
            builder = builder.bearer_auth(token.expose_secret());
        }
        let response = builder.send().await?;
        if response.status().is_redirection() {
            let location = response
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| Error::IntegrityMismatch("download redirect".into()))?;
            if location.len() > 8192 {
                return Err(Error::IntegrityMismatch("download redirect size".into()));
            }
            url = url
                .join(location)
                .map_err(|_| Error::IntegrityMismatch("download redirect URL".into()))?;
            continue;
        }
        return Ok(response);
    }
    Err(Error::IntegrityMismatch("download redirect limit".into()))
}
pub(super) async fn fetch(store: &ArtifactStore, preset: ModelPreset) -> Result<VerifiedSnapshot> {
    let store = store.clone();
    let catalog = Manifest::catalog(preset)?;
    let lock_path = store.lock_path("allocation");
    // A retained lock serializes acquisition and publication with pruning across processes.
    let _lock = spawn_blocking(move || exclusive_lock(&lock_path))
        .await
        .map_err(|_| Error::WorkerUnavailable)??;
    if store
        .snapshot_dir(preset)
        .join("manifest.json")
        .try_exists()?
    {
        return store.open(preset).await;
    }
    let capacity_store = store.clone();
    let capacity_catalog = catalog.clone();
    spawn_blocking(move || capacity_store.reserve(&capacity_catalog))
        .await
        .map_err(|_| Error::WorkerUnavailable)??;
    let deadline = Instant::now() + Duration::from_secs(7200);
    let token = std::env::var("HF_TOKEN")
        .ok()
        .filter(|s| s.len() <= 8192)
        .map(SecretString::from);
    for artifact in &catalog.files {
        if store.blob(artifact).try_exists()? {
            let path = store.blob(artifact);
            let info = artifact.clone();
            spawn_blocking(move || verify_file(&path, &info, deadline))
                .await
                .map_err(|_| Error::WorkerUnavailable)??;
            continue;
        }
        download(&store, &catalog, artifact, token.as_ref(), deadline).await?;
    }
    let publication = store.clone();
    let manifest = catalog.clone();
    spawn_blocking(move || publication.publish(&manifest))
        .await
        .map_err(|_| Error::WorkerUnavailable)??;
    store.open(preset).await
}
#[allow(
    clippy::too_many_lines,
    reason = "transaction ownership and retry state remain together, within the repository \
              150-line ceiling"
)]
async fn download(
    store: &ArtifactStore,
    catalog: &Manifest,
    artifact: &ArtifactInfo,
    token: Option<&SecretString>,
    deadline: Instant,
) -> Result<()> {
    let partial = store
        .root
        .join("staging")
        .join(format!("{}.partial", artifact.sha256));
    let mut final_error = Error::ArtifactMissing;
    for attempt in 0..3 {
        if Instant::now() >= deadline {
            return Err(Error::DeadlineExceeded);
        }
        let mut retry_delay = Duration::from_millis(250 * (1 << attempt));
        let result = async {
            let metadata = tokio::fs::symlink_metadata(&partial).await;
            let offset = match metadata {
                Ok(m) if m.is_file() && !m.file_type().is_symlink() && m.len() <= artifact.size => {
                    m.len()
                }
                Ok(_) => return Err(Error::IntegrityMismatch("partial file".into())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
                Err(e) => return Err(e.into()),
            };
            if offset == artifact.size {
                return Ok(());
            }
            let url = Url::parse(&format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                catalog.repository,
                catalog.revision.as_str(),
                artifact.name
            ))
            .map_err(|_| Error::IntegrityMismatch("catalog URL".into()))?;
            let mut response = request(url, offset, token).await?;
            if let Some(seconds) = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|s| (1..=60).contains(s))
            {
                retry_delay = Duration::from_secs(seconds);
            }
            let resume = response.status() == StatusCode::PARTIAL_CONTENT && offset > 0;
            if resume {
                let expected = format!("bytes {offset}-{} /{}", artifact.size - 1, artifact.size)
                    .replace(" /", "/");
                let range = response
                    .headers()
                    .get("content-range")
                    .and_then(|h| h.to_str().ok())
                    .ok_or_else(|| Error::IntegrityMismatch("missing content range".into()))?;
                if range != expected {
                    return Err(Error::IntegrityMismatch("content range".into()));
                }
            } else if response.status() != StatusCode::OK {
                return Err(response.error_for_status().err().map_or_else(
                    || Error::IntegrityMismatch("unexpected download status".into()),
                    Error::from,
                ));
            }
            if response
                .headers()
                .get("content-encoding")
                .is_some_and(|v| v != "identity")
            {
                return Err(Error::IntegrityMismatch("compressed download".into()));
            }
            let start = if resume { offset } else { 0 };
            if response
                .content_length()
                .is_some_and(|n| n != artifact.size - start)
            {
                return Err(Error::IntegrityMismatch("download size".into()));
            }
            let mut output = tokio::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .append(resume)
                .truncate(!resume)
                .open(&partial)
                .await?;
            let mut total = start;
            while let Some(chunk) = response.chunk().await? {
                if Instant::now() >= deadline {
                    return Err(Error::DeadlineExceeded);
                }
                total = total
                    .checked_add(chunk.len() as u64)
                    .ok_or(Error::StorageLimit)?;
                if total > artifact.size {
                    return Err(Error::IntegrityMismatch(
                        "download exceeded reviewed size".into(),
                    ));
                }
                timeout(Duration::from_secs(30), output.write_all(&chunk))
                    .await
                    .map_err(|_| Error::DeadlineExceeded)??;
            }
            if total != artifact.size {
                return Err(Error::IntegrityMismatch("incomplete download".into()));
            }
            output.sync_all().await?;
            Ok(())
        }
        .await;
        match result {
            Ok(()) => {
                let source = partial.clone();
                let target = store.blob(artifact);
                let info = artifact.clone();
                return spawn_blocking(move || {
                    if let Err(error) = verify_file(&source, &info, deadline) {
                        if matches!(error, Error::IntegrityMismatch(_)) {
                            std::fs::remove_file(&source)?;
                        }
                        return Err(error);
                    }
                    publish_blob(&source, &target)
                })
                .await
                .map_err(|_| Error::WorkerUnavailable)?;
            }
            Err(e) => {
                let retry = match &e {
                    Error::Download(e) => {
                        e.is_timeout()
                            || e.is_connect()
                            || e.is_body()
                            || e.status().is_some_and(|s| {
                                s.is_server_error() || matches!(s.as_u16(), 408 | 429)
                            })
                    }
                    _ => false,
                };
                if !retry || attempt == 2 {
                    return Err(e);
                }
                final_error = e;
                let jitter = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| u64::from(d.subsec_millis()) % 251);
                sleep(retry_delay + Duration::from_millis(jitter)).await;
            }
        }
    }
    Err(final_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_should_reject_private_and_unapproved_origins() {
        for ip in [
            "127.0.0.1",
            "10.0.0.1",
            "169.254.1.1",
            "100.64.0.1",
            "::1",
            "::ffff:127.0.0.1",
            "fc00::1",
        ] {
            assert!(!public_ip(ip.parse().unwrap()));
        }
        assert!(!approved_host("huggingface.co.evil.example"));
        assert!(approved_host("huggingface.co"));
    }
}
