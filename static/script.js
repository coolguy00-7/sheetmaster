const form = document.getElementById("chat-form");
const promptInput = document.getElementById("prompt");
const output = document.getElementById("output");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();

  if (!prompt) {
    output.textContent = "Please enter a prompt.";
    return;
  }

  output.textContent = "Loading...";

  try {
    const res = await fetch("/api/gemini", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });

    const data = await res.json();
    if (!res.ok) {
      output.textContent = data.error || "Request failed.";
      return;
    }

    output.textContent = data.response || "No response text.";
  } catch (error) {
    output.textContent = `Network error: ${error.message}`;
  }
});
