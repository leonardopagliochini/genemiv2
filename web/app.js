const form = document.getElementById("pipeline-form");
const runButton = document.getElementById("run-button");
const statusMessage = document.getElementById("status-message");
const commandPreview = document.getElementById("command-preview");
const formErrors = document.getElementById("form-errors");
const stageElements = Array.from(document.querySelectorAll(".stage"));
const stageLookup = new Map(stageElements.map((el) => [el.dataset.stage, el]));
const terminalWrapper = document.getElementById("terminal-wrapper");
const toggleTerminalBtn = document.getElementById("toggle-terminal");
const terminalLog = document.getElementById("terminal-log");

let currentSource = null;
let runInProgress = false;

function resetStages() {
  stageLookup.forEach((el) => {
    el.dataset.status = "pending";
  });
}

function setStageStatus(stageName, status) {
  const el = stageLookup.get(stageName);
  if (!el) return;
  el.dataset.status = status;
}

function appendLog(line) {
  const needsNewline = terminalLog.textContent?.endsWith("\n") ?? true;
  terminalLog.textContent += (needsNewline ? "" : "\n") + line;
  terminalLog.textContent += "\n";
  terminalLog.scrollTop = terminalLog.scrollHeight;
}

function setStatus(text, variant = "") {
  statusMessage.textContent = text || "";
  statusMessage.classList.remove("success", "error");
  if (variant) {
    statusMessage.classList.add(variant);
  }
}

function setCommandPreview(commandParts) {
  if (!Array.isArray(commandParts) || !commandParts.length) {
    commandPreview.textContent = "";
    return;
  }
  commandPreview.textContent = commandParts.join(" ");
}

function clearErrors() {
  formErrors.textContent = "";
}

function renderErrors(errors) {
  if (!errors || Object.keys(errors).length === 0) {
    clearErrors();
    return;
  }
  const items = Object.entries(errors)
    .map(([field, message]) => `<li><strong>${field}</strong>: ${message}</li>`)
    .join("");
  formErrors.innerHTML = `<ul>${items}</ul>`;
}

function closeEventSource() {
  if (currentSource) {
    currentSource.close();
    currentSource = null;
  }
}

function toggleTerminal() {
  const collapsed = terminalWrapper.classList.toggle("collapsed");
  toggleTerminalBtn.textContent = collapsed ? "Show log" : "Hide log";
}

toggleTerminalBtn.addEventListener("click", toggleTerminal);

function handleEvent(data) {
  switch (data.type) {
    case "init":
      resetStages();
      setCommandPreview(data.command);
      appendLog("# Starting pipeline");
      break;
    case "log":
      appendLog(data.message);
      break;
    case "stage":
      setStageStatus(data.stage, data.status);
      break;
    case "error":
      appendLog(`[error] ${data.message}`);
      setStatus(data.message, "error");
      break;
    case "done": {
      const success = Number(data.returncode) === 0;
      setStatus(
        success ? "Pipeline completed successfully." : "Pipeline failed. Inspect the log above.",
        success ? "success" : "error",
      );
      if (success) {
        appendLog("# Pipeline finished ✔");
      } else {
        appendLog("# Pipeline finished with errors ✖");
      }
      runButton.disabled = false;
      runInProgress = false;
      break;
    }
    case "close":
      closeEventSource();
      runButton.disabled = false;
      runInProgress = false;
      break;
    default:
      break;
  }
}

async function startPipeline(event) {
  event.preventDefault();
  if (runInProgress) {
    return;
  }

  clearErrors();
  setStatus("");
  commandPreview.textContent = "";
  terminalLog.textContent = "";
  resetStages();

  const extractTime = form.extractTime.value.trim();
  const simulationTime = form.simulationTime.value.trim();
  const mesh = form.mesh.value.trim();
  const procs = form.procs.value.trim();

  if (!extractTime) {
    renderErrors({ extractTime: "Please provide the timestep to extract." });
    return;
  }

  const payload = { extractTime };
  if (simulationTime) payload.simulationTime = simulationTime;
  if (mesh) payload.mesh = mesh;
  if (procs) payload.procs = procs;

  closeEventSource();
  runButton.disabled = true;
  runInProgress = true;
  appendLog("# Submitting pipeline run");

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      renderErrors(data?.errors);
      if (data?.error) {
        appendLog(`[error] ${data.error}`);
        setStatus(data.error, "error");
      }
      runButton.disabled = false;
      runInProgress = false;
      return;
    }

    if (!data.runId) {
      setStatus("Unexpected server response.", "error");
      runButton.disabled = false;
      runInProgress = false;
      return;
    }

    setStatus("Pipeline started…");

    currentSource = new EventSource(`/api/stream/${data.runId}`);
    currentSource.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        handleEvent(parsed);
      } catch (err) {
        console.error("Failed to parse event", err);
      }
    };

    currentSource.onerror = () => {
      if (!runInProgress) {
        return;
      }
      appendLog("[warn] Connection to server lost.");
      setStatus("Connection lost. Check the server logs.", "error");
      closeEventSource();
      runButton.disabled = false;
      runInProgress = false;
    };
  } catch (error) {
    setStatus(`Network error: ${error}`, "error");
    runButton.disabled = false;
    runInProgress = false;
  }
}

form.addEventListener("submit", startPipeline);

window.addEventListener("beforeunload", closeEventSource);

// Initialise defaults on load.
resetStages();
toggleTerminalBtn.textContent = "Show log";
