// The note editor: one BlockSuite doc (AFFiNE's editor engine) per note, in
// page or edgeless mode, loaded from and saved to the Nova Notes server.
//
// The server stores three things per note:
//   content    markdown exported from the doc (search, chat, embeddings, card previews)
//   doc_state  the doc itself, a base64 Yjs update (what the editor reopens)
//   links      the notes this one links to (@ mentions and linked-doc cards), for backlinks
// Files dropped or pasted into a note (images, attachments) go to the server's
// blob store by content hash; the doc keeps only the hash.

import "@toeverything/theme/style.css";
import "./editor.css";

import { ConfigExtension } from "@blocksuite/block-std";
import {
  AffineSchemas,
  AffineSlashMenuWidget,
  EdgelessEditorBlockSpecs,
  EmbedFigmaBlockSpec,
  EmbedGithubBlockSpec,
  EmbedHtmlBlockSpec,
  EmbedLoomBlockSpec,
  EmbedYoutubeBlockSpec,
  FontConfigExtension,
  FontFamily,
  FontStyle,
  FontWeight,
  LinkedWidgetUtils,
  MarkdownAdapter,
  NewDocIcon,
  PageEditorBlockSpecs,
  RefNodeSlotsExtension,
  titleMiddleware,
} from "@blocksuite/blocks";
import { effects as blocksEffects } from "@blocksuite/blocks/effects";
import { Slot } from "@blocksuite/global/utils";
import { AffineEditorContainer } from "@blocksuite/presets";
import { effects as presetsEffects } from "@blocksuite/presets/effects";
import { DocCollection, getAssetName, Job, Schema, Text } from "@blocksuite/store";

// Embeds that load or fetch from third parties (YouTube, Loom, Figma iframes,
// the GitHub API) and the raw-HTML embed. Without their specs the editor never
// creates these blocks: a pasted YouTube or GitHub URL stays a plain link.
// build.mjs also rewrites their URLs, so the bundle cannot reach those hosts.
const REMOVED_EMBEDS = new Set(
  [EmbedFigmaBlockSpec, EmbedGithubBlockSpec, EmbedHtmlBlockSpec, EmbedLoomBlockSpec, EmbedYoutubeBlockSpec].flat(),
);
const withoutRemovedEmbeds = specs => specs.filter(ext => !REMOVED_EMBEDS.has(ext));
// A doc saved before the embeds were removed may still hold one. With no view
// it would render as nothing and drop out of the exported markdown, so it is
// turned into a bookmark card (a plain link) in the same place. The raw-HTML
// embed has no URL to fall back to and is left as it is.
const LINK_EMBED_FLAVOURS = ["affine:embed-youtube", "affine:embed-loom", "affine:embed-figma", "affine:embed-github"];
function embedsToBookmarks(doc) {
  for (const block of doc.getBlocksByFlavour(LINK_EMBED_FLAVOURS)) {
    const model = block.model;
    const parent = doc.getParent(model);
    if (!parent || !model.url) continue;
    const props = { url: model.url, caption: model.caption ?? null, title: model.title ?? null };
    if (parent.flavour === "affine:surface") Object.assign(props, { xywh: model.xywh, index: model.index });
    doc.addBlock("affine:bookmark", props, parent.id, parent.children.indexOf(model));
    doc.deleteBlock(model);
  }
}

const HIDDEN_SLASH_ITEMS = new Set(["New Doc", "YouTube", "GitHub", "Github", "Figma", "Loom"]);

let registered = false;
function registerElements() {
  if (registered) return;
  registered = true;
  // The slash menu's "New Doc" makes an in-memory doc that is never saved as
  // a note; "Linked Doc" (the @ menu, which can create real notes) stays.
  // The third-party embeds are removed with their block specs (see below).
  const slash = AffineSlashMenuWidget.DEFAULT_CONFIG;
  AffineSlashMenuWidget.DEFAULT_CONFIG = { ...slash, items: slash.items.filter(item => !HIDDEN_SLASH_ITEMS.has(item.name)) };
  blocksEffects();
  presetsEffects();
}

const Y = DocCollection.Y;
const FONT_BASE = new URL("./fonts/", import.meta.url).href;
const UNTITLED = "Untitled";

// Edgeless canvas text fonts, served with the bundle.
const FONTS = [
  [FontFamily.Inter, "Inter-VariableFont_slnt-wght.ttf", FontWeight.Light],
  [FontFamily.Inter, "Inter-VariableFont_slnt-wght.ttf", FontWeight.Regular],
  [FontFamily.Inter, "Inter-VariableFont_slnt-wght.ttf", FontWeight.SemiBold],
  [FontFamily.Kalam, "Kalam-Light.ttf", FontWeight.Light],
  [FontFamily.Kalam, "Kalam-Regular.ttf", FontWeight.Regular],
  [FontFamily.Kalam, "Kalam-Bold.ttf", FontWeight.SemiBold],
].map(([font, file, weight]) => ({ font, url: FONT_BASE + file, weight, style: FontStyle.Normal }));

// ---------------------------------------------------------------------------
// Blob storage: images and attachments, stored on the server by content hash.
// ---------------------------------------------------------------------------

class ServerBlobSource {
  constructor(base) {
    this.name = "nova-notes";
    this.readonly = false;
    this.base = base;
    this.cache = new Map();
  }
  url(key) {
    return this.base + "/" + encodeURIComponent(key);
  }
  async get(key) {
    if (this.cache.has(key)) return this.cache.get(key);
    const response = await fetch(this.url(key), { credentials: "same-origin" });
    if (!response.ok) return null;
    const blob = await response.blob();
    this.cache.set(key, blob);
    return blob;
  }
  async set(key, value) {
    this.cache.set(key, value);
    const response = await fetch(this.url(key), {
      method: "PUT",
      credentials: "same-origin",
      headers: { "Content-Type": value.type || "application/octet-stream" },
      body: value,
    });
    if (!response.ok) {
      this.cache.delete(key);
      const message = await errorMessage(response);
      toast(message, true);
      throw new Error(message);
    }
    return key;
  }
  // Blobs may be shared by several notes (same file), so nothing is deleted
  // from the editor; the server keeps them.
  async delete() {}
  async list() {
    return [...this.cache.keys()];
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function toast(message, bad) {
  if (window.nn && window.nn.toast) window.nn.toast(message, bad);
}

async function errorMessage(response) {
  const text = await response.text().catch(() => "");
  if (window.nn && window.nn.errorMessage) return window.nn.errorMessage(text, response.status);
  return text || "Request failed (" + response.status + ")";
}

function toBase64(bytes) {
  let binary = "";
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) binary += String.fromCharCode.apply(null, bytes.subarray(i, i + step));
  return btoa(binary);
}

function fromBase64(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

// Every `pageId` in a doc snapshot (inline @ references and linked-doc
// cards) and every attachment block's file.
function scanSnapshot(node, found) {
  if (Array.isArray(node)) {
    for (const item of node) scanSnapshot(item, found);
  } else if (node && typeof node === "object") {
    if (node.flavour === "affine:attachment" && node.props && node.props.sourceId) {
      found.attachments.push({ key: node.props.sourceId, name: node.props.name || "file" });
    }
    for (const [key, value] of Object.entries(node)) {
      if (key === "pageId" && typeof value === "string") found.pageIds.add(value);
      else scanSnapshot(value, found);
    }
  }
  return found;
}

// BlockSuite exports a bookmark card as `[title](url)` and skips it when the
// title is not a string. Link previews are off, so cards have no title: use
// the URL, or the link would drop out of the note's text.
function titleBookmarks(node) {
  if (Array.isArray(node)) {
    node.forEach(titleBookmarks);
  } else if (node && typeof node === "object") {
    const props = node.props;
    if (node.flavour === "affine:bookmark" && props && typeof props.url === "string" && typeof props.title !== "string") {
      props.title = props.url;
    }
    if (node.children) titleBookmarks(node.children);
  }
}

// Text on the edgeless canvas (shapes, text elements, connector labels), used
// as the note's text when the doc has no blocks with text.
function canvasText(doc) {
  const surface = doc.getBlocksByFlavour("affine:surface")[0];
  const elements = surface && surface.model && surface.model.elementModels;
  if (!elements) return "";
  return elements
    .map(element => (element.text ? element.text.toString().trim() : ""))
    .filter(Boolean)
    .join("\n\n");
}

function initDoc(doc) {
  const rootId = doc.addBlock("affine:page", {});
  doc.addBlock("affine:surface", {}, rootId);
  const noteId = doc.addBlock("affine:note", {}, rootId);
  return noteId;
}

// Markdown images that point off this server become plain links before
// import. BlockSuite fetches every http(s) image it imports, so a model answer
// saved as a note could otherwise send data to any URL just by being opened.
// Same-origin paths (/note-blobs/...) and data: images are kept.
const REMOTE_IMAGE = /!\[([^\]]*)\]\(\s*<?((?:[a-z][a-z0-9+.-]*:|[\\/]{2})[^)\s>]*)>?[^)]*\)/gi;

function unlinkRemoteImages(markdown) {
  return markdown.replace(REMOTE_IMAGE, (all, alt, url) =>
    /^data:image\//i.test(url) ? all : "[" + (alt.trim() || url) + "](" + url + ")");
}

async function importMarkdown(doc, noteId, markdown) {
  const job = new Job({ collection: doc.collection, middlewares: [] });
  const adapter = new MarkdownAdapter(job);
  const snapshot = await adapter.toSliceSnapshot({
    file: unlinkRemoteImages(markdown),
    assets: job.assetsManager,
    workspaceId: doc.collection.id,
    pageId: doc.id,
  });
  if (!snapshot) return false;
  for (const block of snapshot.content.flatMap(x => x.children)) {
    await job.snapshotToBlock(block, doc, noteId);
  }
  return true;
}

async function exportMarkdown(doc) {
  const job = new Job({
    collection: doc.collection,
    middlewares: [titleMiddleware, ({ adapterConfigs }) => adapterConfigs.set("docLinkBaseUrl", "/notes")],
  });
  const snapshot = job.docToSnapshot(doc);
  if (!snapshot) return { markdown: "", links: [], attachments: [] };
  const found = scanSnapshot(snapshot, { pageIds: new Set(), attachments: [] });
  const links = [...found.pageIds].filter(id => id !== doc.id);
  titleBookmarks(snapshot.blocks);
  const result = await new MarkdownAdapter(job).fromDocSnapshot({ snapshot, assets: job.assetsManager });
  // Images are written as `assets/<file name>`; point them at the blob store
  // so the markdown stands on its own.
  let markdown = result.file || "";
  const assets = job.assetsManager.getAssets();
  for (const blobId of result.assetsIds || []) {
    const name = getAssetName(assets, blobId);
    markdown = markdown.split("](assets/" + name + ")").join("](/note-blobs/" + blobId + ")");
    markdown = markdown.split("](assets/" + encodeURI(name) + ")").join("](/note-blobs/" + blobId + ")");
  }
  return { markdown: markdown.trim(), links, attachments: found.attachments };
}

// ---------------------------------------------------------------------------
// Mount
// ---------------------------------------------------------------------------

/**
 * Mount an editor.
 *
 * options:
 *   docId       the note id ("note:...") or "" for a new note
 *   state       base64 Yjs update, or null
 *   content     markdown to start from when there is no state
 *   mode        "page" | "edgeless"
 *   notes       [{id, title}] the notes that can be linked with @
 *   blobBase    URL prefix of the blob store
 *   onOpenNote(id)            a linked note was clicked
 *   createNote(title)         create a note to link to; resolves to {id, title}
 *   onChange()                the doc changed
 */
export async function mount(host, options) {
  registerElements();
  ensureThemeAttribute();

  const schema = new Schema().register(AffineSchemas);
  const blobSource = new ServerBlobSource(options.blobBase || "/note-blobs");
  const collection = new DocCollection({ schema, blobSources: { main: blobSource } });
  collection.meta.initialize();

  const docId = options.docId || "note:new";
  for (const note of options.notes || []) {
    if (note.id === docId) continue;
    collection.meta.addDocMeta({ id: note.id, title: note.title || UNTITLED, createDate: Date.now(), tags: [] });
  }

  const doc = collection.createDoc({ id: docId });
  if (options.state) {
    try {
      Y.applyUpdate(doc.spaceDoc, fromBase64(options.state));
    } catch (e) {
      console.error("Could not read the saved editor state; rebuilding from the note text", e);
    }
  }
  doc.load();
  if (!doc.root) {
    const noteId = initDoc(doc);
    const markdown = (options.content || "").trim();
    let imported = false;
    if (markdown) {
      try {
        imported = await importMarkdown(doc, noteId, markdown);
      } catch (e) {
        console.error("Markdown import failed; keeping the text as plain paragraphs", e);
        for (const para of markdown.split(/\n{2,}/)) {
          doc.addBlock("affine:paragraph", { text: new Text(para) }, noteId);
        }
        imported = true;
      }
    }
    if (!imported || !doc.getBlock(noteId).model.children.length) doc.addBlock("affine:paragraph", {}, noteId);
  }
  embedsToBookmarks(doc);
  doc.resetHistory();

  const slots = { docLinkClicked: new Slot() };
  slots.docLinkClicked.on(info => {
    if (info && info.pageId && options.onOpenNote) options.onOpenNote(info.pageId);
  });

  const linkedWidget = {
    getMenus(query, abort, editorHost, inlineEditor) {
      const groups = [LinkedWidgetUtils.createLinkedDocMenuGroup(query, abort, editorHost, inlineEditor)];
      groups[0].name = "Link to note";
      if (options.createNote) {
        const name = query || UNTITLED;
        const short = name.length > 16 ? name.slice(0, 16) + "…" : name;
        groups.push({
          name: "New note",
          items: [
            {
              key: "create",
              name: 'Create "' + short + '" note',
              icon: NewDocIcon,
              action: async () => {
                abort();
                try {
                  const created = await options.createNote(name);
                  collection.meta.addDocMeta({ id: created.id, title: created.title || name, createDate: Date.now(), tags: [] });
                  LinkedWidgetUtils.insertLinkedNode({ inlineEditor, docId: created.id });
                } catch (e) {
                  toast(e && e.message ? e.message : "Could not create the note", true);
                }
              },
            },
          ],
        });
      }
      return groups;
    },
  };

  const extra = [
    RefNodeSlotsExtension(slots),
    FontConfigExtension(FONTS),
    ConfigExtension("affine:page", { linkedWidget }),
  ];
  const editor = new AffineEditorContainer();
  editor.pageSpecs = [...withoutRemovedEmbeds(PageEditorBlockSpecs), ...extra];
  editor.edgelessSpecs = [...withoutRemovedEmbeds(EdgelessEditorBlockSpecs), ...extra];
  editor.doc = doc;
  editor.mode = options.mode === "edgeless" ? "edgeless" : "page";
  editor.autofocus = !options.docId;
  host.replaceChildren(editor);
  // BlockSuite's theme observer starts as light and only reacts to changes
  // of <html data-theme>; touch the attribute so it picks up the current one.
  await editor.updateComplete;
  const theme = document.documentElement.getAttribute("data-theme");
  document.documentElement.setAttribute("data-theme", theme === "dark" ? "light" : "dark");
  document.documentElement.setAttribute("data-theme", theme);

  let dirty = false;
  const onUpdate = () => {
    if (dirty) return;
    dirty = true;
    if (options.onChange) options.onChange();
  };
  doc.spaceDoc.on("update", onUpdate);

  return {
    editor,
    get mode() {
      return editor.mode;
    },
    setMode(mode) {
      editor.switchEditor(mode === "edgeless" ? "edgeless" : "page");
    },
    isDirty() {
      return dirty;
    },
    markClean() {
      dirty = false;
    },
    setTitle(title) {
      collection.setDocMeta(docId, { title: title || UNTITLED });
    },
    /** The note as the server stores it. */
    async snapshot(title) {
      const { markdown, links, attachments } = await exportMarkdown(doc);
      let content = markdown;
      if (!content) content = canvasText(doc).trim();
      if (!content) content = (title || "").trim();
      return {
        title: title || "",
        content,
        doc_state: toBase64(Y.encodeStateAsUpdate(doc.spaceDoc)),
        doc_mode: editor.mode,
        links,
        attachments,
      };
    },
    destroy() {
      doc.spaceDoc.off("update", onUpdate);
      editor.remove();
      collection.dispose && collection.dispose();
    },
  };
}

// BlockSuite follows `<html data-theme>`. The app only sets it after the user
// picks a theme; until then, mirror the system scheme the app is showing.
function ensureThemeAttribute() {
  const root = document.documentElement;
  if (root.getAttribute("data-theme")) return;
  const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.setAttribute("data-theme", dark ? "dark" : "light");
}
