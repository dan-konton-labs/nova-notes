// Bundles the note editor (BlockSuite 0.19.5, AFFiNE's editor engine) into
// ../static/editor/, which the server embeds. Run with `npm run build` after
// changing src/ or the pinned packages, and commit the output.
//
// BlockSuite ships hard-coded calls to third parties: AFFiNE's hosted services
// (link previews, an image proxy), a Twitter preview API, the embed blocks
// (YouTube/Loom/Figma iframes and oEmbed lookups, the GitHub API) and the
// Lottie player, which loads its WebAssembly from a public CDN. They are all
// rewritten here to a same-origin path the server does not serve, so the
// editor makes no third-party requests; links stay plain links. The embed
// blocks are also removed from the editor itself (src/editor.js), and a Rust
// test (services::note_docs) fails if any of these hosts reappears in the
// bundle.

import { build } from "esbuild";
import { cpSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, "..", "static", "editor");

const REWRITES = [
  ["https://affine-worker.toeverything.workers.dev/api/worker/image-proxy", "/note-editor/offline"],
  ["https://affine-worker.toeverything.workers.dev/api/worker/link-preview", "/note-editor/offline"],
  ["https://api.fxtwitter.com/status/", "/note-editor/offline/"],
  ["https://www.youtube.com/embed/", "/note-editor/offline/"],
  ["https://img.youtube.com/vi/", "/note-editor/offline/"],
  ["https://youtube.com/oembed", "/note-editor/offline"],
  ["https://www.loom.com/embed/", "/note-editor/offline/"],
  ["https://www.loom.com/v1/oembed", "/note-editor/offline"],
  ["https://www.figma.com/embed", "/note-editor/offline"],
  ["https://api.github.com/repos/", "/note-editor/offline/"],
  ["https://unpkg.com/", "/note-editor/offline/"],
  ["https://cdn.jsdelivr.net/npm/", "/note-editor/offline/"],
];

const offline = {
  name: "offline",
  setup(b) {
    b.onLoad({ filter: /(@blocksuite[\\/](affine-shared[\\/]dist[\\/]consts|affine-block-embed[\\/]dist)|@lottiefiles[\\/]dotlottie-(web|wc)[\\/]dist)[\\/].*\.js$/ }, args => {
      let contents = readFileSync(args.path, "utf8");
      for (const [from, to] of REWRITES) contents = contents.split(from).join(to);
      return { contents, loader: "js" };
    });
  },
};

// See src/icons-shim.js.
const iconsShim = {
  name: "icons-shim",
  setup(b) {
    b.onResolve({ filter: /^@blocksuite\/icons\/lit\?real$/ }, async args => {
      const real = await b.resolve("@blocksuite/icons/lit", { kind: args.kind, resolveDir: args.resolveDir, pluginData: { real: true } });
      return { path: real.path };
    });
    b.onResolve({ filter: /^@blocksuite\/icons\/lit$/ }, args =>
      args.pluginData && args.pluginData.real ? undefined : { path: join(here, "src", "icons-shim.js") });
  },
};

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

await build({
  entryPoints: { editor: join(here, "src", "editor.js") },
  outdir: out,
  bundle: true,
  format: "esm",
  splitting: true,
  minify: true,
  target: "es2022",
  chunkNames: "chunks/[name]-[hash]",
  assetNames: "assets/[name]-[hash]",
  loader: { ".ttf": "file", ".woff": "file", ".woff2": "file", ".svg": "file", ".png": "file" },
  legalComments: "linked",
  logLevel: "info",
  plugins: [offline, iconsShim],
  define: { "process.env.NODE_ENV": '"production"' },
});

// Canvas fonts for edgeless text (SIL Open Font License, shipped in @toeverything/theme).
const fonts = join(here, "node_modules", "@toeverything", "theme", "fonts");
mkdirSync(join(out, "fonts"), { recursive: true });
for (const file of [
  "inter/Inter-VariableFont_slnt,wght.ttf",
  "inter/OFL.txt",
  "kalam/Kalam-Light.ttf",
  "kalam/Kalam-Regular.ttf",
  "kalam/Kalam-Bold.ttf",
  "kalam/OFL.txt",
]) {
  const name = file.replace("/OFL.txt", "-OFL.txt").split("/").pop().replace(",", "-");
  cpSync(join(fonts, file), join(out, "fonts", name));
}
