# Nova Notes (Rust): one binary serving the JSON API, the web UI and the
# background worker. Prompts, migrations and static files are compiled in.
#
#   docker build -t nova-notes:<version> .
#
# Build on the machine that will run it (linux/amd64 on the lab). A release
# build needs about 4 GB of RAM; BUILD_JOBS limits parallel compiles on a
# busy host:  docker build --build-arg BUILD_JOBS=2 ...

ARG RUST_VERSION=1.100
ARG DEBIAN_RELEASE=bookworm

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
FROM rust:${RUST_VERSION}-${DEBIAN_RELEASE} AS build
ARG BUILD_JOBS=
WORKDIR /src

# The workspace's .cargo/config.toml points target-dir at target.nosync (an
# iCloud workaround); inside the image the default is fine.
ENV CARGO_TARGET_DIR=/src/target

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p nova-notes ${BUILD_JOBS:+--jobs "$BUILD_JOBS"} \
    && install -D -m 0755 /src/target/release/nova-notes /out/nova-notes

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM debian:${DEBIAN_RELEASE}-slim AS runtime

# ffmpeg/ffprobe: podcast audio (MP3/M4A) and audio/video source extraction.
# curl: the health check. tini: clean signal handling as PID 1.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/nova-notes /usr/local/bin/nova-notes

# Same data path as the Python image, so file paths stored in the database
# (uploads, podcasts, note files) stay valid on the same volume.
ENV DATA_FOLDER=/app/data \
    API_HOST=0.0.0.0 \
    API_PORT=5055 \
    RUST_MIN_STACK=16777216 \
    RUST_LOG=info

WORKDIR /app
VOLUME ["/app/data"]
EXPOSE 5055

# Runs as root, like the Python image: the existing notebook_data volume is
# owned by root. Revisit once the volume's ownership is migrated.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:5055/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["nova-notes"]
