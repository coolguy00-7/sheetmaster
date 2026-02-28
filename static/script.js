const form = document.getElementById("analyze-form");
const filesInput = document.getElementById("files");
const dropzone = document.getElementById("dropzone");
const fileSummary = document.getElementById("file-summary");
const fileList = document.getElementById("file-list");
const meta = document.getElementById("meta");
const output = document.getElementById("output");
const generateSheetBtn = document.getElementById("generate-sheet-btn");
const sheetMeta = document.getElementById("sheet-meta");
const sheetPage1 = document.getElementById("sheet-page-1");
const sheetPage2 = document.getElementById("sheet-page-2");

const renderFileSelection = () => {
  const files = filesInput.files;
  if (!files || files.length === 0) {
    fileSummary.textContent = "No files selected.";
    fileList.innerHTML = "";
    return;
  }

  fileSummary.textContent = `${files.length} file(s) selected`;
  fileList.innerHTML = "";
  Array.from(files).forEach((file) => {
    const chip = document.createElement("span");
    chip.className = "file-chip";
    chip.textContent = file.name;
    fileList.appendChild(chip);
  });
};

filesInput.addEventListener("change", renderFileSelection);

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("active");
  });
});

dropzone.addEventListener("drop", (event) => {
  const dropped = event.dataTransfer.files;
  if (!dropped || dropped.length === 0) {
    return;
  }

  filesInput.files = dropped;
  renderFileSelection();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const files = filesInput.files;

  if (!files || files.length === 0) {
    output.textContent = "Please select at least one file.";
    meta.textContent = "";
    return;
  }

  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));

  output.textContent = "Loading...";
  meta.textContent = `Uploading ${files.length} file(s)...`;

  try {
    const res = await fetch("/api/analyze-practice", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) {
      const details =
        typeof data.details === "string"
          ? data.details
          : data.details
            ? JSON.stringify(data.details, null, 2)
            : "";
      output.textContent = details ? `${data.error || "Request failed."}\n\n${details}` : data.error || "Request failed.";
      meta.textContent = "";
      return;
    }

    output.textContent = data.response || "No response text.";
    const modelUsed = data.model_used ? ` using ${data.model_used}` : "";
    meta.textContent = `Analyzed ${data.total_files} file(s)${modelUsed}: ${data.files_analyzed.join(", ")}`;
  } catch (error) {
    output.textContent = `Network error: ${error.message}`;
    meta.textContent = "";
  }
});

const splitSheetText = (text) => {
  const lines = text.split("\n");
  const midpoint = Math.ceil(lines.length / 2);
  const first = lines.slice(0, midpoint).join("\n").trim();
  const second = lines.slice(midpoint).join("\n").trim();
  return [first, second];
};

generateSheetBtn.addEventListener("click", async () => {
  const analysis = output.textContent.trim();
  if (!analysis || analysis === "No response yet." || analysis === "Loading...") {
    sheetMeta.textContent = "Generate analysis first.";
    return;
  }

  sheetMeta.textContent = "Generating 2-page reference sheet...";
  sheetPage1.textContent = "Generating...";
  sheetPage2.textContent = "";

  try {
    const res = await fetch("/api/generate-reference-sheet", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ analysis }),
    });

    const data = await res.json();
    if (!res.ok) {
      const details =
        typeof data.details === "string"
          ? data.details
          : data.details
            ? JSON.stringify(data.details, null, 2)
            : "";
      sheetMeta.textContent = "Sheet generation failed.";
      sheetPage1.textContent = details ? `${data.error || "Request failed."}\n\n${details}` : data.error || "Request failed.";
      sheetPage2.textContent = "";
      return;
    }

    const [pageOne, pageTwo] = splitSheetText(data.reference_sheet || "");
    sheetPage1.textContent = pageOne || "No content generated.";
    sheetPage2.textContent = pageTwo || "No overflow content.";
    sheetMeta.textContent = `Reference sheet generated using ${data.model_used}.`;
  } catch (error) {
    sheetMeta.textContent = "Sheet generation failed.";
    sheetPage1.textContent = `Network error: ${error.message}`;
    sheetPage2.textContent = "";
  }
});
