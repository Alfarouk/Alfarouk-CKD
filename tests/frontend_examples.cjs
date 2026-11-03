"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const staticDir = path.resolve(__dirname, "../src/ckd_research/static");
const html = fs.readFileSync(path.join(staticDir, "index.html"), "utf8");
const script = fs.readFileSync(path.join(staticDir, "app.js"), "utf8");
class Element {
  constructor(attributes = {}) {
    this.attributes = attributes;
    this.classes = new Set((attributes.class || "").split(/\s+/).filter(Boolean));
    this.classList = {
      toggle: (name, force) => {
        const on = force === undefined ? !this.classes.has(name) : force;
        if (on) this.classes.add(name); else this.classes.delete(name);
        return on;
      },
      contains: name => this.classes.has(name)
    };
    this.style = {};
    this.children = [];
    this.value = "";
    this.disabled = Boolean(attributes.disabled);
    this.textContent = "";
    this.src = attributes.src || "";
    this.href = attributes.href || "";
  }
  setAttribute(name, value) { this.attributes[name] = value; }
  getAttribute(name) { return this.attributes[name] ?? null; }
  replaceChildren(...children) { this.children = children; }
  append(...children) { this.children.push(...children); }
  add(option) { this.children.push(option); }
}
const elements = new Map();
for (const tag of html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) {
  const attrs = {};
  for (const pair of tag[0].matchAll(/([\w-]+)="([^"]*)"/g)) attrs[pair[1]] = pair[2];
  if (/\sdisabled(?:\s|>)/.test(tag[0])) attrs.disabled = true;
  assert(!elements.has(attrs.id), "duplicate HTML id: " + attrs.id);
  elements.set(attrs.id, new Element(attrs));
}
const requests = [];
const context = vm.createContext({
  document: {getElementById: id => elements.get(id) || null, createElement: () => new Element()},
  fetch: async url => {
    requests.push(url);
    return {ok:true, json:async () => ({models:[{id:"test-model", title:"Current checkpoint", installed:true, task:"classification", profiles:[{id:"test-profile", title:"Current input settings"}]}]})};
  },
  Option: function(text, value) { this.text = text; this.value = value; },
  setTimeout, clearTimeout
});
const state = expression => vm.runInContext(expression, context);
(async () => {
  vm.runInContext(script, context, {filename:"app.js"});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(elements.get("examples-tab").getAttribute("aria-selected"), "true");
  assert.equal(elements.get("examples-panel").classList.contains("hidden"), false);
  assert.equal(elements.get("work-panel").classList.contains("hidden"), true);
  assert.equal(state("current"), null);
  assert.deepEqual(requests, ["/api/models"]);
  assert.equal(elements.get("analysis-image").src, "");
  const figures = [
    ["ultrasound", "/examples/ultrasound-output.jpg"],
    ["biopsy-sclerosed", "/examples/biopsy-sclerosed-output.jpg"],
    ["biopsy-normal", "/examples/biopsy-normal-output.jpg"]
  ];
  for (const [id, url] of figures) {
    elements.get("example-" + id).onclick();
    assert.equal(elements.get("example-image").src, url);
    assert.equal(elements.get("example-open").href, url);
    assert.equal(elements.get("example-image-link").href, url);
    assert.equal(elements.get("example-" + id).getAttribute("aria-pressed"), "true");
    assert.equal(state("current"), null);
  }
  elements.get("model-tab").onclick();
  assert.equal(elements.get("analysis-empty").classList.contains("hidden"), false);
  assert.equal(elements.get("analysis-results").classList.contains("hidden"), true);
  elements.get("model-select").value = "test-model";
  elements.get("model-select").onchange();
  state("modelBusy(true)");
  for (const id of ["model-select", "profile-select", "model-image-file", "run-model", "examples-tab", "files-tab", "model-tab"]) assert.equal(elements.get(id).disabled, true, id);
  state("modelBusy(false)");
  for (const id of ["examples-tab", "files-tab", "model-tab"]) assert.equal(elements.get(id).disabled, false, id);
  context.fixture = {image:"data:image/png;base64,AQ==", report:{source:"model", task:"classification", actual_checkpoint_executed:true, model_name:"Current checkpoint", input_profile:{id:"test-profile", title:"Current input settings"}, raw_logits:[0.2,0.8], softmax_index_scores:[0.35,0.65]}};
  state("show(fixture)");
  const report = state("JSON.stringify(current.report)");
  assert.equal(elements.get("analysis-results").classList.contains("hidden"), false);
  elements.get("examples-tab").onclick();
  elements.get("example-biopsy-sclerosed").onclick();
  assert.equal(elements.get("work-panel").classList.contains("hidden"), true);
  assert.equal(state("JSON.stringify(current.report)"), report);
  elements.get("files-tab").onclick();
  assert.equal(elements.get("analysis-results").classList.contains("hidden"), false);
  assert.equal(elements.get("result-title").textContent, "Current result · Current checkpoint");
  assert.equal(elements.get("result-profile").textContent, "Input used: Current input settings");
  assert.equal(elements.get("source-badge").textContent, "MODEL INFERENCE · LOCAL");
  assert.equal(state("JSON.stringify(current.report)"), report);
  process.stdout.write(JSON.stringify({default_view:"examples", figures:3, initial_requests:requests, saved_figures_keep_live_report:true, model_busy_controls:true}) + "\n");
})().catch(error => { console.error(error); process.exitCode = 1; });
