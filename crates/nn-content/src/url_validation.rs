//! Outbound URL validation (the SSRF guard).
//!
//! Port of `open_notebook/utils/url_validation.py`.
//!
//! The original is deliberately permissive because the application is
//! self-hosted: private ranges and localhost are allowed (Ollama, LM Studio, a
//! NAS on the LAN). It blocks only
//!
//! * schemes other than `http` / `https`,
//! * malformed URLs and URLs without a host,
//! * link-local addresses (`169.254.0.0/16`, `fe80::/10`, and IPv4-mapped
//!   forms of them), which is where cloud metadata services live,
//! * the AWS IMDSv6 address `fd00:ec2::254`, and the cloud metadata
//!   addresses outside link-local (Alibaba `100.100.100.200`, Azure
//!   `168.63.129.16`, Oracle `192.0.0.192`, Google `fd20:ce::254`),
//! * IPv6 forms that embed a blocked IPv4 address (IPv4-mapped,
//!   IPv4-compatible, NAT64 `64:ff9b::/96`, 6to4 `2002::/16`),
//! * host names that resolve to any of those.
//!
//! That is [`UrlPolicy::default`], kept for operator-configured endpoints
//! (AI provider base URLs). [`UrlPolicy::public_only`] also rejects loopback,
//! private, carrier-grade NAT, unique-local, unspecified and multicast
//! addresses; the server uses it for user-supplied source URLs unless the
//! operator opts out.
//!
//! [`validate_url`] checks a URL. [`prepare_pinned_http_target`] additionally
//! resolves the host once and returns the vetted addresses, so the request can
//! be sent to exactly the addresses that were checked (closing the
//! DNS-rebinding window between validation and connect).

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use async_trait::async_trait;
use url::{Host, Url};

/// A rejected URL. The message is written for end users and matches the
/// wording of the original.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct UrlValidationError(pub String);

impl UrlValidationError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// True when the URL was acceptable but its host name did not resolve
    /// (only [`prepare_pinned_http_target`] reports this). That is a network
    /// condition, not a policy violation.
    pub fn is_unresolved_host(&self) -> bool {
        self.0.starts_with(UNRESOLVED_PREFIX)
    }
}

const UNRESOLVED_PREFIX: &str = "Could not resolve hostname";

/// Which address classes are rejected in addition to the always-blocked
/// link-local and metadata addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UrlPolicy {
    /// Also reject loopback, RFC 1918 private, carrier-grade NAT,
    /// unique-local, unspecified, broadcast and multicast addresses.
    /// `false` reproduces the original's self-hosted behaviour.
    pub block_private_networks: bool,
}

impl UrlPolicy {
    /// The original behaviour: only link-local and metadata addresses are blocked.
    pub fn self_hosted() -> Self {
        Self::default()
    }

    /// Additionally block every non-public address.
    pub fn public_only() -> Self {
        Self { block_private_networks: true }
    }
}

/// Host name resolution, replaceable in tests.
#[async_trait]
pub trait Resolver: Send + Sync {
    /// All addresses `host` resolves to. An error means the name does not resolve.
    async fn resolve(&self, host: &str) -> std::io::Result<Vec<IpAddr>>;
}

/// Resolves through the operating system (`getaddrinfo`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemResolver;

#[async_trait]
impl Resolver for SystemResolver {
    async fn resolve(&self, host: &str) -> std::io::Result<Vec<IpAddr>> {
        let addrs = tokio::net::lookup_host((host, 0)).await?;
        Ok(addrs.map(|a| a.ip()).collect())
    }
}

/// Fixed host-to-address table. Unknown hosts fail to resolve.
#[derive(Debug, Clone, Default)]
pub struct StaticResolver {
    entries: std::collections::HashMap<String, Vec<IpAddr>>,
}

impl StaticResolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Map `host` (case-insensitive) to `addrs`.
    pub fn with(mut self, host: &str, addrs: &[IpAddr]) -> Self {
        self.entries.insert(host.to_ascii_lowercase(), addrs.to_vec());
        self
    }
}

#[async_trait]
impl Resolver for StaticResolver {
    async fn resolve(&self, host: &str) -> std::io::Result<Vec<IpAddr>> {
        self.entries
            .get(&host.to_ascii_lowercase())
            .cloned()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, format!("unknown host {host}")))
    }
}

/// An outbound request target whose host has been resolved and vetted.
///
/// `url` has the host replaced by the vetted address, with `headers` carrying
/// the original `Host` and `sni_hostname` the name TLS must present; that is
/// the shape the Python original handed to httpx. With reqwest the simpler
/// route is [`PinnedHttpTarget::apply`], which keeps the original URL and
/// pins its host to the vetted addresses on the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedHttpTarget {
    /// The URL with the host replaced by the pinned address. Unchanged when
    /// the original host is already an IP literal.
    pub url: String,
    /// `Host` header to send with `url`; empty for IP literals.
    pub headers: Vec<(String, String)>,
    /// TLS server name for `url`; set for `https` targets with a host name.
    pub sni_hostname: Option<String>,
    /// The URL as given (trimmed).
    pub original_url: String,
    /// The original host name (ASCII / punycode); `None` for IP literals.
    pub host: Option<String>,
    /// Every vetted address of `host`, with the target port. Empty for IP literals.
    pub addrs: Vec<SocketAddr>,
}

impl PinnedHttpTarget {
    /// Pin `host` to the vetted addresses on a reqwest client builder. Requests
    /// for `original_url` made with the resulting client connect only to those
    /// addresses, while `Host` and TLS SNI stay on the original name.
    pub fn apply(&self, builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
        match &self.host {
            Some(host) if !self.addrs.is_empty() => builder.resolve_to_addrs(host, &self.addrs),
            _ => builder,
        }
    }
}

const LINK_LOCAL_LITERAL: &str = "Link-local addresses (169.254.x.x) are not allowed for security reasons. \
These addresses are used for cloud metadata endpoints.";
const IMDS_V6_LITERAL: &str = "The AWS IMDSv6 metadata address (fd00:ec2::254) is not allowed for \
security reasons.";
const METADATA_LITERAL: &str = "Cloud metadata addresses are not allowed for security reasons.";
const NO_HOST: &str = "Invalid URL: hostname could not be determined.";

const AWS_IMDS_V6: Ipv6Addr = Ipv6Addr::new(0xfd00, 0x0ec2, 0, 0, 0, 0, 0, 0x0254);

/// Cloud metadata and host-agent addresses outside link-local, blocked under
/// every policy: Alibaba Cloud metadata, the Azure WireServer and the Oracle
/// Cloud legacy metadata address.
const METADATA_V4: [Ipv4Addr; 3] =
    [Ipv4Addr::new(100, 100, 100, 200), Ipv4Addr::new(168, 63, 129, 16), Ipv4Addr::new(192, 0, 0, 192)];

/// Google Cloud's IPv6 metadata server address.
const GCP_METADATA_V6: Ipv6Addr = Ipv6Addr::new(0xfd20, 0x00ce, 0, 0, 0, 0, 0, 0x0254);

fn is_link_local_v4(ip: Ipv4Addr) -> bool {
    ip.octets()[0] == 169 && ip.octets()[1] == 254
}

/// The IPv4 address an IPv6 address stands for, when it embeds one:
/// IPv4-mapped (`::ffff:a.b.c.d`), IPv4-compatible (`::a.b.c.d`, except `::`
/// and `::1`), NAT64 (`64:ff9b::/96`) and 6to4 (`2002:aabb:ccdd::/48`).
/// These all reach the IPv4 host on a dual-stack or translated network, so
/// they are judged as that host.
fn embedded_v4(v6: Ipv6Addr) -> Option<Ipv4Addr> {
    if let Some(v4) = v6.to_ipv4_mapped() {
        return Some(v4);
    }
    let s = v6.segments();
    let low = |hi: u16, lo: u16| Ipv4Addr::new((hi >> 8) as u8, hi as u8, (lo >> 8) as u8, lo as u8);
    if s[..6] == [0, 0, 0, 0, 0, 0] && !(v6.is_unspecified() || v6.is_loopback()) {
        return Some(low(s[6], s[7]));
    }
    if s[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
        return Some(low(s[6], s[7]));
    }
    if s[0] == 0x2002 {
        return Some(low(s[1], s[2]));
    }
    None
}

fn is_internal_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    ip.is_loopback()
        || ip.is_private()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || ip.is_multicast()
        || o[0] == 0
        || (o[0] == 100 && (o[1] & 0xc0) == 64)
}

/// Reject `ip` when it is link-local or a metadata address (always), or an
/// internal address under [`UrlPolicy::public_only`]. `resolved_from` is the
/// host name the address came from, when it was not written literally.
fn reject_dangerous_ip(ip: IpAddr, resolved_from: Option<&str>, policy: &UrlPolicy) -> Result<(), UrlValidationError> {
    let embedded = match ip {
        IpAddr::V6(v6) => embedded_v4(v6),
        IpAddr::V4(_) => None,
    };
    let link_local = match ip {
        IpAddr::V4(v4) => is_link_local_v4(v4),
        IpAddr::V6(v6) => (v6.segments()[0] & 0xffc0) == 0xfe80 || embedded.is_some_and(is_link_local_v4),
    };
    if link_local {
        return Err(UrlValidationError::new(match resolved_from {
            Some(host) => format!(
                "Hostname '{host}' resolves to a link-local address (169.254.x.x) \
                 which is not allowed for security reasons. These addresses are used \
                 for cloud metadata endpoints."
            ),
            None => LINK_LOCAL_LITERAL.to_string(),
        }));
    }

    if ip == IpAddr::V6(AWS_IMDS_V6) {
        return Err(UrlValidationError::new(match resolved_from {
            Some(host) => format!(
                "Hostname '{host}' resolves to the AWS IMDSv6 metadata address \
                 (fd00:ec2::254), which is not allowed for security reasons."
            ),
            None => IMDS_V6_LITERAL.to_string(),
        }));
    }

    let metadata = match ip {
        IpAddr::V4(v4) => METADATA_V4.contains(&v4),
        IpAddr::V6(v6) => v6 == GCP_METADATA_V6 || embedded.is_some_and(|v4| METADATA_V4.contains(&v4)),
    };
    if metadata {
        return Err(UrlValidationError::new(match resolved_from {
            Some(host) => format!(
                "Hostname '{host}' resolves to a cloud metadata address, which is not allowed for security reasons."
            ),
            None => METADATA_LITERAL.to_string(),
        }));
    }

    if policy.block_private_networks {
        let internal = match ip {
            IpAddr::V4(v4) => is_internal_v4(v4),
            IpAddr::V6(v6) => {
                v6.is_loopback()
                    || v6.is_unspecified()
                    || v6.is_multicast()
                    || (v6.segments()[0] & 0xfe00) == 0xfc00
                    || (v6.segments()[0] & 0xffc0) == 0xfec0
                    || embedded.is_some_and(is_internal_v4)
            }
        };
        if internal {
            return Err(UrlValidationError::new(match resolved_from {
                Some(host) => format!(
                    "Hostname '{host}' resolves to a private or internal address, \
                     which is not allowed for security reasons."
                ),
                None => "Private and internal addresses are not allowed for security reasons.".to_string(),
            }));
        }
    }
    Ok(())
}

/// True when `ip` passes [`UrlPolicy::public_only`]: not link-local, metadata,
/// loopback, private, carrier-grade NAT, unique-local or otherwise internal.
pub fn is_public_ip(ip: IpAddr) -> bool {
    reject_dangerous_ip(ip, None, &UrlPolicy::public_only()).is_ok()
}

/// Check one resolved address of `host` against `policy`, for callers that
/// resolve names themselves (a connect-time DNS resolver). Same rules and
/// messages as [`validate_url_with`].
pub fn check_resolved_ip(host: &str, ip: IpAddr, policy: &UrlPolicy) -> Result<(), UrlValidationError> {
    reject_dangerous_ip(ip, Some(host), policy)
}

enum Target {
    Ip(IpAddr),
    Name { host: String, url: Url },
}

/// Scheme and host checks shared by both entry points.
fn parse_target(raw: &str) -> Result<Target, UrlValidationError> {
    let scheme_end = raw.find(':').filter(|&end| {
        let scheme = &raw[..end];
        scheme.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            && scheme.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
    });
    let scheme = scheme_end.map(|end| raw[..end].to_ascii_lowercase()).unwrap_or_default();
    if scheme != "http" && scheme != "https" {
        return Err(UrlValidationError::new(format!(
            "Invalid URL scheme: '{scheme}'. Only http and https are allowed."
        )));
    }

    // Zoned IPv6 literals ([fe80::1%eth0], [..%25eth0]) are not valid in a URL
    // for the parser, but must still be checked so a zone cannot hide a
    // blocked address.
    if let Some(literal) = bracketed_host(raw) {
        if let Some((address, _zone)) = literal.split_once('%') {
            return match address.parse::<Ipv6Addr>() {
                Ok(v6) => Ok(Target::Ip(IpAddr::V6(v6))),
                Err(_) => Err(UrlValidationError::new("Invalid URL format. Check server logs for details.")),
            };
        }
    }

    let url =
        Url::parse(raw).map_err(|_| UrlValidationError::new("Invalid URL format. Check server logs for details."))?;
    match url.host() {
        None => Err(UrlValidationError::new(NO_HOST)),
        Some(Host::Ipv4(v4)) => Ok(Target::Ip(IpAddr::V4(v4))),
        Some(Host::Ipv6(v6)) => Ok(Target::Ip(IpAddr::V6(v6))),
        Some(Host::Domain("")) => Err(UrlValidationError::new(NO_HOST)),
        Some(Host::Domain(domain)) => {
            let host = domain.to_string();
            Ok(Target::Name { host, url })
        }
    }
}

/// The text between `[` and `]` of the authority, if the host is bracketed.
fn bracketed_host(raw: &str) -> Option<&str> {
    let after_scheme = raw.split_once("://")?.1;
    let authority = after_scheme.split(['/', '?', '#']).next()?;
    let host_part = authority.rsplit_once('@').map_or(authority, |(_, host)| host);
    let inner = host_part.strip_prefix('[')?;
    Some(&inner[..inner.find(']')?])
}

async fn resolve_safe_ips(
    host: &str,
    policy: &UrlPolicy,
    resolver: &dyn Resolver,
) -> Result<std::io::Result<Vec<IpAddr>>, UrlValidationError> {
    match resolver.resolve(host).await {
        Err(e) => Ok(Err(e)),
        Ok(addresses) => {
            for ip in &addresses {
                reject_dangerous_ip(*ip, Some(host), policy)?;
            }
            Ok(Ok(addresses))
        }
    }
}

/// Validate `url` with the default (self-hosted) policy and the system resolver.
pub async fn validate_url(url: &str) -> Result<(), UrlValidationError> {
    validate_url_with(url, &UrlPolicy::default(), &SystemResolver).await
}

/// Validate `url`.
///
/// An empty URL passes (emptiness is the caller's concern, as in the
/// original). A host name that does not resolve also passes: it may be valid
/// from where the request will actually run, and nothing can be fetched from
/// it here anyway.
pub async fn validate_url_with(url: &str, policy: &UrlPolicy, resolver: &dyn Resolver) -> Result<(), UrlValidationError> {
    let raw = url.trim();
    if raw.is_empty() {
        return Ok(());
    }
    match parse_target(raw)? {
        Target::Ip(ip) => reject_dangerous_ip(ip, None, policy),
        Target::Name { host, .. } => {
            let _unresolvable_is_allowed = resolve_safe_ips(&host, policy, resolver).await?;
            Ok(())
        }
    }
}

/// [`prepare_pinned_http_target_with`] using the default policy and the system resolver.
pub async fn prepare_pinned_http_target(url: &str) -> Result<PinnedHttpTarget, UrlValidationError> {
    prepare_pinned_http_target_with(url, &UrlPolicy::default(), &SystemResolver).await
}

/// Validate `url`, resolve its host once and pin the target to the vetted
/// addresses. Unlike [`validate_url_with`], an empty URL or a host that does
/// not resolve is an error, because the result is about to be fetched.
pub async fn prepare_pinned_http_target_with(
    url: &str,
    policy: &UrlPolicy,
    resolver: &dyn Resolver,
) -> Result<PinnedHttpTarget, UrlValidationError> {
    let raw = url.trim();
    if raw.is_empty() {
        return Err(UrlValidationError::new(NO_HOST));
    }

    let (host, parsed) = match parse_target(raw)? {
        Target::Ip(ip) => {
            reject_dangerous_ip(ip, None, policy)?;
            return Ok(PinnedHttpTarget {
                url: raw.to_string(),
                headers: Vec::new(),
                sni_hostname: None,
                original_url: raw.to_string(),
                host: None,
                addrs: Vec::new(),
            });
        }
        Target::Name { host, url } => (host, url),
    };

    let unresolved = || UrlValidationError::new(format!("{UNRESOLVED_PREFIX} '{host}' for outbound request."));
    let safe_ips = resolve_safe_ips(&host, policy, resolver).await?.map_err(|_| unresolved())?;
    // Prefer IPv4 when available (same vetted set, simpler URL form).
    let pinned_ip = safe_ips.iter().find(|ip| ip.is_ipv4()).or(safe_ips.first()).copied().ok_or_else(unresolved)?;

    let port = parsed.port_or_known_default().unwrap_or(80);
    let host_header = match parsed.port() {
        Some(explicit) => format!("{host}:{explicit}"),
        None => host.clone(),
    };

    let mut pinned = parsed.clone();
    pinned
        .set_ip_host(pinned_ip)
        .map_err(|_| UrlValidationError::new("Invalid URL format. Check server logs for details."))?;
    // The pinned URL carries only host, port, path, query and fragment.
    let _ = pinned.set_username("");
    let _ = pinned.set_password(None);

    Ok(PinnedHttpTarget {
        url: pinned.to_string(),
        headers: vec![("Host".to_string(), host_header)],
        sni_hostname: (parsed.scheme() == "https").then(|| host.clone()),
        original_url: raw.to_string(),
        addrs: safe_ips.iter().map(|ip| SocketAddr::new(*ip, port)).collect(),
        host: Some(host),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    /// Resolver for tests that must not touch DNS: every name resolves to a
    /// public address unless listed.
    struct PublicResolver;

    #[async_trait]
    impl Resolver for PublicResolver {
        async fn resolve(&self, _host: &str) -> std::io::Result<Vec<IpAddr>> {
            Ok(vec![ip("93.184.216.34")])
        }
    }

    async fn check(url: &str) -> Result<(), UrlValidationError> {
        validate_url_with(url, &UrlPolicy::default(), &PublicResolver).await
    }

    async fn pin(url: &str, resolver: &dyn Resolver) -> Result<PinnedHttpTarget, UrlValidationError> {
        prepare_pinned_http_target_with(url, &UrlPolicy::default(), resolver).await
    }

    fn host_header(target: &PinnedHttpTarget) -> Vec<(String, String)> {
        target.headers.clone()
    }

    fn expect_host(value: &str) -> Vec<(String, String)> {
        vec![("Host".to_string(), value.to_string())]
    }

    // --- validate_url ---

    #[tokio::test]
    async fn valid_http_and_https_urls_pass() {
        for url in [
            "https://api.openai.com",
            "https://example.com/api",
            "http://example.com",
            "https://api.openai.com/v1",
            "https://generativelanguage.googleapis.com",
            "https://my-resource.openai.azure.com",
            "https://api.together.xyz",
        ] {
            check(url).await.unwrap_or_else(|e| panic!("{url}: {e}"));
        }
    }

    #[tokio::test]
    async fn invalid_schemes_are_rejected() {
        for url in ["ftp://example.com", "file:///etc/passwd", "gopher://example.com", "javascript:alert(1)"] {
            let error = check(url).await.unwrap_err();
            assert!(error.0.contains("Invalid URL scheme"), "{url}: {error}");
        }
        assert_eq!(
            check("ftp://example.com").await.unwrap_err().0,
            "Invalid URL scheme: 'ftp'. Only http and https are allowed."
        );
    }

    #[tokio::test]
    async fn localhost_and_loopback_are_allowed_for_self_hosted() {
        let resolver = StaticResolver::new().with("localhost", &[ip("::1"), ip("127.0.0.1")]);
        for url in [
            "http://localhost:8000",
            "http://localhost:11434",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:11434",
            "http://127.0.0.2",
            "http://[::1]:8000",
        ] {
            validate_url_with(url, &UrlPolicy::default(), &resolver).await.unwrap_or_else(|e| panic!("{url}: {e}"));
        }
    }

    #[tokio::test]
    async fn private_ips_are_allowed_for_self_hosted() {
        for url in [
            "http://10.0.0.1",
            "http://172.16.0.1:8080",
            "http://192.168.1.1",
            "http://192.168.1.100:11434",
            "http://10.0.0.50:11434",
            "http://[::ffff:192.168.1.1]",
        ] {
            check(url).await.unwrap_or_else(|e| panic!("{url}: {e}"));
        }
    }

    #[tokio::test]
    async fn link_local_is_rejected() {
        let error = check("http://169.254.169.254").await.unwrap_err();
        assert!(error.0.contains("Link-local addresses"));
        let error = check("http://169.254.169.254/latest/meta-data").await.unwrap_err();
        assert!(error.0.contains("Link-local addresses"));
        let error = check("http://[fe80::1]/").await.unwrap_err();
        assert!(error.0.contains("Link-local addresses"));
    }

    #[tokio::test]
    async fn numeric_host_forms_cannot_hide_link_local() {
        // 169.254.169.254 written as a single integer and in hex.
        for url in ["http://2852039166/", "http://0xa9.0xfe.0xa9.0xfe/"] {
            let error = check(url).await.unwrap_err();
            assert!(error.0.contains("Link-local addresses"), "{url}: {error}");
        }
    }

    #[tokio::test]
    async fn empty_url_is_not_an_error() {
        check("").await.unwrap();
        check("   ").await.unwrap();
    }

    #[tokio::test]
    async fn malformed_urls_are_rejected() {
        assert!(check("not-a-url").await.is_err());
        assert!(check("http://").await.is_err());
        assert!(check("http://[not-an-ip]/").await.is_err());
    }

    #[tokio::test]
    async fn ipv4_mapped_ipv6_link_local_is_rejected() {
        let error = check("http://[::ffff:169.254.169.254]").await.unwrap_err();
        assert!(error.0.contains("Link-local addresses"));
    }

    #[tokio::test]
    async fn aws_imds_v6_is_rejected() {
        let error = check("http://[fd00:ec2::254]/").await.unwrap_err();
        assert!(error.0.contains("IMDSv6"));
    }

    #[tokio::test]
    async fn scoped_aws_imds_v6_is_rejected() {
        for url in ["http://[fd00:ec2::254%eth0]/", "http://[fd00:ec2::254%25eth0]/"] {
            let error = check(url).await.unwrap_err();
            assert!(error.0.contains("IMDSv6"), "{url}: {error}");
        }
    }

    #[tokio::test]
    async fn hostname_resolving_to_link_local_is_rejected() {
        let resolver = StaticResolver::new().with("evil.example", &[ip("93.184.216.34"), ip("169.254.169.254")]);
        let error = validate_url_with("http://evil.example/x", &UrlPolicy::default(), &resolver).await.unwrap_err();
        assert!(error.0.contains("resolves to a link-local address"));
        assert!(error.0.contains("evil.example"));
    }

    #[tokio::test]
    async fn unresolvable_hostname_is_allowed() {
        let resolver = StaticResolver::new();
        validate_url_with("https://internal.corp.example/v1", &UrlPolicy::default(), &resolver).await.unwrap();
    }

    #[tokio::test]
    async fn public_only_policy_blocks_internal_addresses() {
        let strict = UrlPolicy::public_only();
        let resolver = StaticResolver::new()
            .with("localhost", &[ip("127.0.0.1")])
            .with("nas.lan", &[ip("192.168.1.20")])
            .with("example.com", &[ip("93.184.216.34")]);
        for url in [
            "http://127.0.0.1:8000",
            "http://10.0.0.1",
            "http://172.16.0.1",
            "http://192.168.1.1",
            "http://100.101.102.103",
            "http://100.64.0.1",
            "http://0.0.0.0",
            "http://[::1]/",
            "http://[fd12:3456::1]/",
            "http://[::ffff:10.0.0.1]/",
            "http://localhost:11434",
            "http://nas.lan/",
        ] {
            let error = validate_url_with(url, &strict, &resolver).await.unwrap_err();
            assert!(error.0.contains("rivate"), "{url}: {error}");
        }
        validate_url_with("https://example.com/", &strict, &resolver).await.unwrap();
        validate_url_with("http://8.8.8.8/", &strict, &resolver).await.unwrap();
        // The always-on blocks keep their own wording.
        let error = validate_url_with("http://169.254.169.254", &strict, &resolver).await.unwrap_err();
        assert!(error.0.contains("Link-local"));
    }

    #[tokio::test]
    async fn cloud_metadata_addresses_are_blocked_under_every_policy() {
        let resolver = StaticResolver::new().with("meta.example", &[ip("100.100.100.200")]);
        for policy in [UrlPolicy::self_hosted(), UrlPolicy::public_only()] {
            for url in [
                "http://100.100.100.200/latest/meta-data/",
                "http://168.63.129.16/machine?comp=goalstate",
                "http://192.0.0.192/latest/",
                "http://[fd20:ce::254]/computeMetadata/v1/",
                "http://[::ffff:100.100.100.200]/",
                "http://[64:ff9b::a83f:8110]/",
            ] {
                let error = validate_url_with(url, &policy, &resolver).await.unwrap_err();
                assert!(error.0.contains("metadata"), "{url}: {error}");
            }
            let error = validate_url_with("http://meta.example/", &policy, &resolver).await.unwrap_err();
            assert!(error.0.contains("meta.example") && error.0.contains("metadata"), "{error}");
        }
        // Tailscale MagicDNS (100.100.100.100) is not a metadata address.
        validate_url_with("http://100.100.100.100/", &UrlPolicy::self_hosted(), &resolver).await.unwrap();
    }

    #[tokio::test]
    async fn ipv6_forms_embedding_ipv4_are_judged_as_that_ipv4() {
        // 169.254.169.254 as IPv4-compatible, NAT64 and 6to4.
        for url in ["http://[::a9fe:a9fe]/", "http://[64:ff9b::a9fe:a9fe]/", "http://[2002:a9fe:a9fe::1]/"] {
            let error = check(url).await.unwrap_err();
            assert!(error.0.contains("Link-local"), "{url}: {error}");
        }
        // 127.0.0.1 / 10.0.0.1 in the same forms are internal under public_only.
        let strict = UrlPolicy::public_only();
        for url in ["http://[::7f00:1]/", "http://[64:ff9b::a00:1]/", "http://[2002:a00:1::]/"] {
            let error = validate_url_with(url, &strict, &PublicResolver).await.unwrap_err();
            assert!(error.0.contains("rivate"), "{url}: {error}");
        }
        // A NAT64 / 6to4 form of a public address is public.
        validate_url_with("http://[64:ff9b::808:808]/", &strict, &PublicResolver).await.unwrap();
        validate_url_with("http://[2002:808:808::1]/", &strict, &PublicResolver).await.unwrap();
        assert!(is_public_ip(ip("8.8.8.8")));
        assert!(!is_public_ip(ip("100.64.0.1")));
        assert!(!is_public_ip(ip("::7f00:1")));
    }

    // --- prepare_pinned_http_target ---

    #[tokio::test]
    async fn ip_literal_is_unchanged() {
        let target = pin("http://127.0.0.1:11434/api/tags", &StaticResolver::new()).await.unwrap();
        assert_eq!(target.url, "http://127.0.0.1:11434/api/tags");
        assert!(target.headers.is_empty());
        assert_eq!(target.sni_hostname, None);
        assert!(target.addrs.is_empty());

        let target = pin("http://192.168.1.30:1234/v1/models", &StaticResolver::new()).await.unwrap();
        assert_eq!(target.url, "http://192.168.1.30:1234/v1/models");
        let target = pin("http://[::1]:11434/api/tags", &StaticResolver::new()).await.unwrap();
        assert_eq!(target.url, "http://[::1]:11434/api/tags");
        assert!(target.headers.is_empty());
    }

    #[tokio::test]
    async fn pinning_rejects_blocked_literals() {
        let resolver = StaticResolver::new();
        assert!(pin("http://169.254.169.254/latest/meta-data", &resolver).await.unwrap_err().0.contains("Link-local"));
        for url in ["http://[fd00:ec2::254%eth0]/", "http://[fd00:ec2::254%25eth0]/"] {
            assert!(pin(url, &resolver).await.unwrap_err().0.contains("IMDSv6"), "{url}");
        }
        assert!(pin("", &resolver).await.is_err());
        assert!(pin("ftp://example.com/", &resolver).await.unwrap_err().0.contains("Invalid URL scheme"));
    }

    #[tokio::test]
    async fn hostname_is_pinned_to_resolved_ip() {
        let resolver = StaticResolver::new().with("ollama.local", &[ip("192.168.1.30")]);
        let target = pin("http://ollama.local:11434/api/tags", &resolver).await.unwrap();
        assert_eq!(target.url, "http://192.168.1.30:11434/api/tags");
        assert_eq!(host_header(&target), expect_host("ollama.local:11434"));
        assert_eq!(target.sni_hostname, None);
        assert_eq!(target.host.as_deref(), Some("ollama.local"));
        assert_eq!(target.addrs, vec!["192.168.1.30:11434".parse().unwrap()]);
        assert_eq!(target.original_url, "http://ollama.local:11434/api/tags");
    }

    #[tokio::test]
    async fn https_sets_sni_hostname() {
        let resolver = StaticResolver::new().with("api.example.com", &[ip("93.184.216.34")]);
        let target = pin("https://api.example.com/v1/models", &resolver).await.unwrap();
        assert_eq!(target.url, "https://93.184.216.34/v1/models");
        assert_eq!(host_header(&target), expect_host("api.example.com"));
        assert_eq!(target.sni_hostname.as_deref(), Some("api.example.com"));
        assert_eq!(target.addrs, vec!["93.184.216.34:443".parse().unwrap()]);
    }

    #[tokio::test]
    async fn unicode_hostname_is_idna_encoded() {
        let resolver = StaticResolver::new().with("xn--bcher-kva.example", &[ip("93.184.216.34")]);
        let target = pin("https://bücher.example/v1/models", &resolver).await.unwrap();
        assert_eq!(target.url, "https://93.184.216.34/v1/models");
        assert_eq!(host_header(&target), expect_host("xn--bcher-kva.example"));
        assert_eq!(target.sni_hostname.as_deref(), Some("xn--bcher-kva.example"));
    }

    #[tokio::test]
    async fn pinning_rejects_hostname_resolving_to_link_local() {
        let resolver = StaticResolver::new().with("evil.example", &[ip("169.254.169.254")]);
        let error = pin("http://evil.example/v1/models", &resolver).await.unwrap_err();
        assert!(error.0.contains("link-local"));
    }

    #[tokio::test]
    async fn pinning_fails_for_unresolvable_hostname() {
        let error = pin("http://nowhere.invalid/", &StaticResolver::new()).await.unwrap_err();
        assert_eq!(error.0, "Could not resolve hostname 'nowhere.invalid' for outbound request.");
        assert!(error.is_unresolved_host());
        assert!(!pin("http://169.254.1.1/", &StaticResolver::new()).await.unwrap_err().is_unresolved_host());
    }

    // --- self-hosted setups must survive pinning ---

    #[tokio::test]
    async fn localhost_prefers_ipv4() {
        let resolver = StaticResolver::new().with("localhost", &[ip("::1"), ip("127.0.0.1")]);
        let target = pin("http://localhost:11434/api/tags", &resolver).await.unwrap();
        assert_eq!(target.url, "http://127.0.0.1:11434/api/tags");
        assert_eq!(host_header(&target), expect_host("localhost:11434"));
        assert_eq!(target.sni_hostname, None);
        assert_eq!(target.addrs.len(), 2);
    }

    #[tokio::test]
    async fn docker_lan_and_tailscale_hosts_survive_pinning() {
        let resolver = StaticResolver::new()
            .with("host.docker.internal", &[ip("192.168.65.2")])
            .with("llm.lan", &[ip("10.0.1.20")])
            .with("ollama.example.ts.net", &[ip("100.101.102.103")]);

        let target = pin("http://host.docker.internal:11434/v1/models", &resolver).await.unwrap();
        assert_eq!(target.url, "http://192.168.65.2:11434/v1/models");
        assert_eq!(host_header(&target), expect_host("host.docker.internal:11434"));

        let target = pin("http://llm.lan/v1/models", &resolver).await.unwrap();
        assert_eq!(target.url, "http://10.0.1.20/v1/models");
        assert_eq!(host_header(&target), expect_host("llm.lan"));

        let target = pin("http://ollama.example.ts.net:11434/api/tags", &resolver).await.unwrap();
        assert_eq!(target.url, "http://100.101.102.103:11434/api/tags");
        assert_eq!(host_header(&target), expect_host("ollama.example.ts.net:11434"));
    }

    #[tokio::test]
    async fn ipv6_only_host_is_pinned_with_brackets() {
        let resolver = StaticResolver::new().with("v6.example.com", &[ip("2606:4700::1111")]);
        let target = pin("https://v6.example.com/v1/models", &resolver).await.unwrap();
        assert_eq!(target.url, "https://[2606:4700::1111]/v1/models");
        assert_eq!(host_header(&target), expect_host("v6.example.com"));
        assert_eq!(target.sni_hostname.as_deref(), Some("v6.example.com"));
    }

    #[tokio::test]
    async fn query_string_and_path_are_preserved() {
        let resolver = StaticResolver::new().with("api.ppq.ai", &[ip("93.184.216.34")]);
        let target = pin("https://api.ppq.ai/v1/models?type=all", &resolver).await.unwrap();
        assert_eq!(target.url, "https://93.184.216.34/v1/models?type=all");
    }

    #[tokio::test]
    async fn credentials_are_not_carried_into_the_pinned_url() {
        let resolver = StaticResolver::new().with("example.com", &[ip("93.184.216.34")]);
        let target = pin("https://user:secret@example.com/x", &resolver).await.unwrap();
        assert_eq!(target.url, "https://93.184.216.34/x");
    }

    #[tokio::test]
    async fn system_resolver_handles_ip_literals_without_dns() {
        validate_url("http://127.0.0.1:8000").await.unwrap();
        assert!(validate_url("http://169.254.169.254").await.is_err());
        let target = prepare_pinned_http_target("http://127.0.0.1:1/x").await.unwrap();
        assert_eq!(target.url, "http://127.0.0.1:1/x");
    }
}
