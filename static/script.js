const form = document.getElementById("analyze-form");
const filesInput = document.getElementById("files");
const meta = document.getElementById("meta");
const output = document.getElementById("output");

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
      output.textContent = data.error || "Request failed.";
      meta.textContent = "";
      return;
    }

    output.textContent = data.response || "No response text.";
    meta.textContent = `Analyzed ${data.total_files} file(s): ${data.files_analyzed.join(", ")}`;
  } catch (error) {
    output.textContent = `Network error: ${error.message}`;
    meta.textContent = "";
  }
});
