"use strict";
let current = null;
let view = "overlay";
const el = id => document.getElementById(id);
const error = message => { el("error").textContent = message; el("error").classList.toggle("hidden", !message); };
async function request(path, body) {
  error("");
  const response = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Analysis failed");
  return result;
}
function show(result) {
  current = result;
  const r = result.report, m = r.measurements, physical = m.units === "mm";
  renderView();
  el("source-badge").textContent = r.source.startsWith("generated") ? "GENERATED FIXTURE · NO MODEL RUN" : "SUPPLIED LABELS · NO MODEL RUN";
  el("frame-caption").textContent = r.source.startsWith("generated") ? "Synthetic engineering fixture — not a patient scan" : "Your image and labels · processed locally";
  el("units-badge").textContent = physical ? "MILLIMETRES" : "PIXELS";
  el("area").textContent = m.area.toLocaleString(undefined,{maximumFractionDigits:1});
  el("area-unit").textContent = physical ? "mm²" : "pixels²";
  el("major").textContent = m.major_axis === null ? "—" : m.major_axis.toFixed(1) + (physical ? " mm" : " px");
  el("minor").textContent = m.minor_axis === null ? "—" : m.minor_axis.toFixed(1) + (physical ? " mm" : " px");
  el("coverage").textContent = (m.coverage_fraction * 100).toFixed(1) + "%";
  el("coverage-meter").style.width = (m.coverage_fraction * 100) + "%";
  el("calibration-note").replaceChildren();
  const icon=document.createElement("span"), p=document.createElement("p"), strong=document.createElement("strong");
  icon.textContent=physical?"✓":"○";
  strong.textContent=physical?(r.fixture_calibration?"Illustrative sample scale.":"Supplied pixel scale."):"No physical scale supplied.";
  p.append(strong,document.createElement("br"),physical?`${m.calibration.row_mm} × ${m.calibration.column_mm} mm / pixel. No disease classification is inferred.`:"These are pixel measurements. No centimetres or estimated volume are invented.");
  el("calibration-note").append(icon,p);
}
function renderView() {
  if (!current) return;
  const image = el("analysis-image");
  const subject = current.report.source.startsWith("generated") ? "Generated engineering fixture" : "User-supplied image";
  image.src = current[view];
  image.alt = subject + (view === "overlay" ? " with region-label overlay" : " without overlay");
  for (const name of ["image", "overlay"]) {
    el(name + "-view").classList.toggle("active", name === view);
    el(name + "-view").setAttribute("aria-pressed", String(name === view));
  }
}
async function sample() { try { show(await request("/api/demo",{calibrated:el("calibrated").checked})); } catch(e) { error(e.message); } }
function tab(files) {
  for (const [id,on] of [["sample-tab",!files],["files-tab",files]]) {el(id).classList.toggle("active",on);el(id).setAttribute("aria-selected",String(on));}
  el("sample-controls").classList.toggle("hidden",files);el("upload-controls").classList.toggle("hidden",!files);
  if(!files) sample();
}
el("sample-tab").onclick=()=>tab(false);el("files-tab").onclick=()=>tab(true);
el("calibrated").onchange=sample;el("reset").onclick=()=>{el("calibrated").checked=false;sample();};
for (const name of ["image","overlay"]) el(name+"-view").onclick=()=>{view=name;renderView();};
function read(file){return new Promise((resolve,reject)=>{if(!file)return reject(new Error("Choose both an image and a label mask"));if(file.size>1500000)return reject(new Error("Each file must be smaller than 1.5 MB"));const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error("Unable to read file"));reader.readAsDataURL(file);});}
el("upload-controls").onsubmit=async event=>{event.preventDefault();const button=event.submitter;button.disabled=true;try{const [image,mask]=await Promise.all([read(el("image-file").files[0]),read(el("mask-file").files[0])]);let row=el("row-spacing").value,col=el("column-spacing").value;if(Boolean(row)!==Boolean(col))throw new Error("Supply both row and column spacing, or leave both blank");show(await request("/api/analyze",{image,mask,spacing:row?[Number(row),Number(col)]:null}));}catch(e){error(e.message);}finally{button.disabled=false;}};
el("download").onclick=()=>{if(!current)return;const url=URL.createObjectURL(new Blob([JSON.stringify(current.report,null,2)+"\n"],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download="CKD-Analysis.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
sample();
