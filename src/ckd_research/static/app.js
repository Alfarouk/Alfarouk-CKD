"use strict";
let current = null;
let view = "overlay";
let models = [];
let resultRequest = 0;
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
  const r = result.report, classification = r.task === "classification";
  const generated = r.source.startsWith("generated"), model = r.actual_checkpoint_executed === true;
  el("result-title").textContent = model ? "Current result · " + r.model_name : generated ? "Current result · generated engineering fixture" : "Current result · uploaded image and labels";
  el("result-profile").textContent = model ? "Input used: " + r.input_profile.title : generated ? "Generated labels only. No checkpoint ran." : "Label schema: " + (r.label_schema_id === "source-four" ? "four-class model labels" : "five labels") + ". No checkpoint ran.";
  el("geometry-panel").classList.toggle("hidden", classification);
  el("classification-panel").classList.toggle("hidden", !classification);
  el("image-heading").textContent = classification ? "Input image" : "Image & labels";
  el("source-badge").textContent = model ? "MODEL INFERENCE · LOCAL" : generated ? "GENERATED FIXTURE · NO MODEL RUN" : "UPLOADED LABELS · NO MODEL RUN";
  el("frame-caption").textContent = generated ? "Synthetic engineering fixture — not a patient scan" : model ? "Your image · model output · research only" : "Your image and labels · processed locally";
  el("legend").replaceChildren();
  if (classification) {
    el("index-scores").replaceChildren();
    r.raw_logits.forEach((logit, i) => {
      const card=document.createElement("div"), label=document.createElement("span"), score=document.createElement("strong"), raw=document.createElement("small");
      card.className="metric index-score"; label.textContent=`Index ${i} · class meaning unknown`;
      score.textContent=r.softmax_index_scores[i].toFixed(4); raw.textContent=`Raw logit ${logit.toFixed(5)}`;
      card.append(label,score,raw); el("index-scores").append(card);
    });
    el("image-note").textContent="Classifier class meanings and training input settings are undocumented. The downloaded JSON records the chosen research profile and both raw logits.";
    view="image";
  } else {
    const m = r.measurements, physical = m.units === "mm";
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
    strong.textContent=physical?(r.fixture_calibration?"Illustrative sample scale.":"Entered pixel scale."):"No physical scale supplied.";
    p.append(strong,document.createElement("br"),physical?`${m.calibration.row_mm} × ${m.calibration.column_mm} mm / pixel. No disease classification is inferred.`:"These are pixel measurements. No centimetres or estimated volume are invented.");
    el("calibration-note").append(icon,p);
    Object.entries(r.labels).filter(([id])=>id!=="0").forEach(([id,name])=>{
      const item=document.createElement("span"), swatch=document.createElement("i");
      swatch.className="c"+id; item.append(swatch,name.replaceAll("_"," ")); el("legend").append(item);
    });
    el("image-note").textContent=r.label_schema_id==="source-four" ? "Four output IDs. Anatomical meanings are unverified; use the matching label schema." : "Region IDs stay neutral. Confirm the annotation schema before assigning anatomy names.";
  }
  el("overlay-view").disabled=!result.overlay;
  renderView();
}
function renderView() {
  if (!current) return;
  const selected = view === "overlay" && current.overlay ? "overlay" : "image";
  const image = el("analysis-image");
  const subject = current.report.source.startsWith("generated") ? "Generated engineering fixture" : "User-supplied image";
  image.src = current[selected];
  image.alt = subject + (selected === "overlay" ? " with region-label overlay" : " without overlay");
  for (const name of ["image", "overlay"]) {
    el(name + "-view").classList.toggle("active", name === selected);
    el(name + "-view").setAttribute("aria-pressed", String(name === selected));
  }
}
async function sample() { const ticket=++resultRequest; try { const result=await request("/api/demo",{calibrated:el("calibrated").checked}); if(ticket===resultRequest)show(result); } catch(e) { if(ticket===resultRequest)error(e.message); } }
function tab(mode) {
  resultRequest++;
  for (const id of ["sample","files","model"]) {const on=id===mode;el(id+"-tab").classList.toggle("active",on);el(id+"-tab").setAttribute("aria-selected",String(on));}
  el("sample-controls").classList.toggle("hidden",mode!=="sample");
  el("upload-controls").classList.toggle("hidden",mode!=="files");
  el("model-controls").classList.toggle("hidden",mode!=="model");
  error(""); if(mode==="sample") sample();
}
for(const mode of ["sample","files","model"])el(mode+"-tab").onclick=()=>tab(mode);
el("calibrated").onchange=sample;el("reset").onclick=()=>{el("calibrated").checked=false;sample();};
for (const name of ["image","overlay"]) el(name+"-view").onclick=()=>{view=name;renderView();};
function read(file){return new Promise((resolve,reject)=>{if(!file)return reject(new Error("Choose the required image file"));if(file.size>1500000)return reject(new Error("Each image must be smaller than 1.5 MB"));const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error("Unable to read file"));reader.readAsDataURL(file);});}
el("upload-controls").onsubmit=async event=>{event.preventDefault();const ticket=++resultRequest;const button=event.submitter;button.disabled=true;try{const [image,mask]=await Promise.all([read(el("image-file").files[0]),read(el("mask-file").files[0])]);let row=el("row-spacing").value,col=el("column-spacing").value;if(Boolean(row)!==Boolean(col))throw new Error("Supply both row and column spacing, or leave both blank");const result=await request("/api/analyze",{image,mask,label_schema_id:el("mask-schema").value,spacing:row?[Number(row),Number(col)]:null});if(ticket===resultRequest)show(result);}catch(e){if(ticket===resultRequest)error(e.message);}finally{button.disabled=false;}};
function modelChoice(){
  const spec=models.find(x=>x.id===el("model-select").value), select=el("profile-select");
  select.replaceChildren(new Option("Choose an explicit input profile",""));select.disabled=!spec;
  if(spec)for(const p of spec.profiles)select.add(new Option(p.title,p.id));
  el("run-model").disabled=!spec?.installed;
  el("model-note").textContent=!spec?"Choose a model to see its input choices and known limits.":!spec.installed?"These model files are not installed here. Follow the README setup steps first.":spec.task==="segmentation"?"Four classes, separate from five-label training. The two grayscale profiles may produce different outputs.":"Two output indices with undocumented class meanings. Input profiles are explicit research settings.";
}
el("model-select").onchange=modelChoice;
function modelBusy(busy) {
  el("model-controls").setAttribute("aria-busy",String(busy));
  el("model-select").disabled=busy;
  el("profile-select").disabled=busy||!el("model-select").value;
  el("model-image-file").disabled=busy;
  el("run-model").disabled=busy||!models.find(x=>x.id===el("model-select").value)?.installed;
  for(const mode of ["sample","files","model"])el(mode+"-tab").disabled=busy;
}
el("model-controls").onsubmit=async event=>{
  event.preventDefault();const ticket=++resultRequest;const button=el("run-model");
  modelBusy(true);button.textContent="Running on CPU…";
  try{
    const model_id=el("model-select").value,profile_id=el("profile-select").value;
    if(!profile_id)throw new Error("Choose an explicit input profile");
    const image=await read(el("model-image-file").files[0]);
    const result=await request("/api/run-model",{image,model_id,profile_id});
    if(ticket===resultRequest)show(result);
  }catch(e){if(ticket===resultRequest)error(e.message);}
  finally{button.textContent="Run on this image ↗";modelBusy(false);}
};
el("download").onclick=()=>{if(!current)return;const url=URL.createObjectURL(new Blob([JSON.stringify(current.report,null,2)+"\n"],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download="CKD-Analysis.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
async function loadModels(){try{const response=await fetch("/api/models");if(!response.ok)throw new Error("Unable to read local model inventory");models=(await response.json()).models;for(const spec of models)el("model-select").add(new Option(spec.title+(spec.installed?"":" · not installed"),spec.id));el("checkpoint-status").textContent=`${models.filter(x=>x.installed).length} of 3 model checkpoints available locally. Generated samples remain software checks; they do not show model accuracy.`;}catch(e){error(e.message);}}
sample();loadModels();
