const form = document.getElementById("pipeline-form");
const meshInput = document.getElementById("mesh");
const procsInput = document.getElementById("procs");
const runButton = document.getElementById("run-button");
const statusMessage = document.getElementById("status-message");
const commandPreview = document.getElementById("command-preview");
const formErrors = document.getElementById("form-errors");
const stageElements = Array.from(document.querySelectorAll(".stage"));
const stageLookup = new Map(stageElements.map((el) => [el.dataset.stage, el]));
const stageOrder = stageElements.map((el) => el.dataset.stage);
const terminalWrapper = document.getElementById("terminal-wrapper");
const toggleTerminalBtn = document.getElementById("toggle-terminal");
const terminalLog = document.getElementById("terminal-log");
const cancelButton = document.getElementById("cancel-button");
const simulationProgress = document.getElementById("simulation-progress");
const simulationProgressFill = document.getElementById("simulation-progress-fill");
const simulationProgressCount = document.getElementById("simulation-progress-count");
const simulationProgressMeta = document.getElementById("simulation-progress-meta");
const caseCard = document.getElementById("case-card");
const caseOptionsContainer = document.getElementById("case-options");
const formCard = document.getElementById("form-card");
const progressCard = document.getElementById("progress-card");
const logCard = document.getElementById("log-card");
const outputsCard = document.getElementById("outputs-card");
const outputsEmpty = document.getElementById("outputs-empty");
const outputsActions = document.getElementById("outputs-actions");
const outputsList = document.getElementById("outputs-list");
const downloadAllLink = document.getElementById("download-all");
const mobileNav = document.querySelector(".mobile-nav");
const mobileNavButtons = mobileNav ? Array.from(mobileNav.querySelectorAll("button")) : [];
const randomToggleField = document.getElementById("random-toggle-field");
const randomToggleInput = document.getElementById("random-breast");
const randomCountField = document.getElementById("random-count-field");
const randomCountInput = document.getElementById("random-count");
const viewerContainer = document.getElementById("viewer-container");
const viewerCanvas = document.getElementById("preview-canvas");
const viewerHint = document.getElementById("viewer-hint");
const resetViewerButton = document.getElementById("reset-viewer");
const VIEWER_DEFAULT_HINT = "Drag to orbit, right-click to pan, scroll to zoom.";
const VIEWER_LOADING_HINT = "Loading surfaces…";
const VIEWER_LIBRARY_ERROR_HINT = "3D preview unavailable: unable to load Three.js resources.";
const VIEWER_DATA_ERROR_HINT = "3D preview unavailable for the generated files.";

let currentSource = null;
let runInProgress = false;
let activeRunId = null;
let wasCancelled = false;
let lastActiveMobileSection = null;
let surfaceViewer = null;
let viewerResetBound = false;
let currentCase = null;
const caseDefaults = new Map();
const FALLBACK_CASES = [
  {
    key: "brain",
    label: "Brain",
    defaultMesh: "mesh/MNI_with_phys.msh",
    defaultProcs: 4,
  },
  {
    key: "breast",
    label: "Breast",
    defaultMesh: "breast_scripts/mesh/breast_128_augmented.msh",
    defaultProcs: 4,
  },
];

function resolveCaseLabel(caseKey) {
  if (!caseKey) {
    return "";
  }
  const config = caseDefaults.get(caseKey) ?? FALLBACK_CASES.find((entry) => entry.key === caseKey);
  return config?.label ?? caseKey;
}

function resetStages() {
  stageLookup.forEach((el) => {
    el.dataset.status = "pending";
  });
}

function resetSimulationProgress() {
  simulationProgress.hidden = true;
  simulationProgressFill.style.transform = "scaleX(0)";
  simulationProgressFill.parentElement?.setAttribute("aria-valuenow", "0");
  simulationProgressCount.textContent = "";
  simulationProgressMeta.textContent = "";
}

function ensureOutputsHidden() {
  outputsCard.hidden = true;
  outputsActions.hidden = true;
  outputsEmpty.hidden = false;
  outputsEmpty.textContent = "Run the pipeline to generate downloadable surfaces.";
  outputsList.innerHTML = "";
  downloadAllLink.href = "#";
  downloadAllLink.removeAttribute("download");
  destroySurfaceViewer();
  updateNavAvailability();
}

function resetCancellationState() {
  wasCancelled = false;
  cancelButton.hidden = true;
  cancelButton.disabled = false;
}

function updateRandomFieldsVisibility() {
  if (!randomToggleField || !randomCountField) {
    return;
  }
  const isBreastCase = currentCase === "breast";
  randomToggleField.hidden = !isBreastCase;
  if (!isBreastCase) {
    if (randomToggleInput) {
      randomToggleInput.checked = false;
    }
    randomCountField.hidden = true;
    return;
  }
  const enabled = Boolean(randomToggleInput?.checked);
  randomCountField.hidden = !enabled;
}

function updateCaseButtonsState() {
  if (!caseOptionsContainer) {
    return;
  }
  const buttons = Array.from(caseOptionsContainer.querySelectorAll("button[data-case]"));
  buttons.forEach((button) => {
    const isActive = button.dataset.case === currentCase;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", isActive ? "true" : "false");
  });
}

function updateNavAvailability() {
  if (!mobileNavButtons.length) {
    return;
  }
  mobileNavButtons.forEach((button) => {
    const targetId = button.dataset.target;
    if (!targetId) {
      button.disabled = true;
      return;
    }
    const section = document.getElementById(targetId);
    const shouldDisable = !section || (section.hidden && targetId !== "case-card");
    button.disabled = shouldDisable;
    if (shouldDisable) {
      button.classList.remove("active");
    }
  });
}

function applyCaseDefaults(config) {
  if (!config) {
    return;
  }
  if (meshInput) {
    meshInput.value = config.defaultMesh ?? "";
  }
  if (procsInput) {
    if (typeof config.defaultProcs === "number" && Number.isFinite(config.defaultProcs)) {
      procsInput.value = String(config.defaultProcs);
    } else {
      procsInput.value = "";
    }
  }
}

function resetForNewCase() {
  closeEventSource();
  runInProgress = false;
  activeRunId = null;
  resetCancellationState();
  resetStages();
  resetSimulationProgress();
  ensureOutputsHidden();
  clearErrors();
  setStatus("");
  setCommandPreview([]);
  terminalLog.textContent = "";
  if (randomToggleInput) {
    randomToggleInput.checked = false;
  }
  if (randomCountInput) {
    const fallback = Number.isFinite(Number(randomCountInput.defaultValue))
      ? randomCountInput.defaultValue
      : "3";
    randomCountInput.value = fallback || "3";
  }
  updateRandomFieldsVisibility();
}

function renderCaseOptions(casesData) {
  if (!caseOptionsContainer) {
    return;
  }
  caseOptionsContainer.innerHTML = "";
  casesData.forEach((config) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "case-select";
    button.dataset.case = config.key;
    const label = config.label ?? config.key;
    button.textContent = label;
    button.setAttribute("aria-pressed", "false");
    button.setAttribute("aria-label", `${label} scenario`);
    button.addEventListener("click", () => selectCase(config.key));
    caseOptionsContainer.appendChild(button);
  });
  updateCaseButtonsState();
}

function selectCase(caseKey) {
  const config =
    caseDefaults.get(caseKey) ??
    FALLBACK_CASES.find((entry) => entry.key === caseKey);
  if (!config) {
    console.warn("[ui] Unknown case selection:", caseKey);
    return;
  }
  if (runInProgress) {
    setStatus("Stop the current run before changing scenario.", "error");
    return;
  }
  currentCase = caseKey;
  resetForNewCase();
  applyCaseDefaults(config);
  if (formCard) {
    formCard.hidden = false;
  }
  if (progressCard) {
    progressCard.hidden = false;
  }
  if (logCard) {
    logCard.hidden = false;
  }
  updateRandomFieldsVisibility();
  updateCaseButtonsState();
  updateNavAvailability();
}

function showCaseSelection({ scroll = false } = {}) {
  if (caseCard) {
    caseCard.hidden = false;
    if (scroll) {
      caseCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }
  updateCaseButtonsState();
  updateNavAvailability();
}

async function loadCases() {
  let payload;
  try {
    const response = await fetch("/api/cases", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    payload = await response.json();
  } catch (error) {
    console.warn("[ui] Unable to fetch cases from server, using fallback set.", error);
    payload = { cases: FALLBACK_CASES };
  }

  const casesData = Array.isArray(payload?.cases) && payload.cases.length
    ? payload.cases
    : FALLBACK_CASES;

  caseDefaults.clear();
  casesData.forEach((item) => {
    if (!item?.key) {
      return;
    }
    caseDefaults.set(item.key, {
      key: item.key,
      label: item.label ?? item.key,
      defaultMesh: item.defaultMesh ?? "",
      defaultProcs:
        typeof item.defaultProcs === "number" && Number.isFinite(item.defaultProcs)
          ? item.defaultProcs
          : 4,
    });
  });

  renderCaseOptions(casesData);
  updateNavAvailability();
}

function destroySurfaceViewer() {
  if (surfaceViewer && typeof surfaceViewer.dispose === "function") {
    surfaceViewer.dispose();
  }
  surfaceViewer = null;
  if (viewerContainer) {
    viewerContainer.hidden = true;
  }
  if (viewerHint) {
    viewerHint.textContent = VIEWER_DEFAULT_HINT;
  }
  if (resetViewerButton) {
    resetViewerButton.disabled = true;
  }
}

function ensureSurfaceViewer() {
  if (!viewerCanvas || !viewerContainer) {
    return null;
  }
  const threeAvailable =
    typeof window !== "undefined" &&
    window.THREE &&
    typeof THREE.STLLoader === "function" &&
    typeof THREE.OrbitControls === "function";

  if (!threeAvailable) {
    if (viewerHint) {
      viewerHint.textContent = VIEWER_LIBRARY_ERROR_HINT;
    }
    viewerContainer.hidden = true;
    return null;
  }

  if (!surfaceViewer) {
    surfaceViewer = createSurfaceViewer(viewerCanvas);
    if (resetViewerButton && !viewerResetBound) {
      resetViewerButton.addEventListener("click", () => {
        surfaceViewer?.resetView();
      });
      viewerResetBound = true;
    }
  }

  viewerContainer.hidden = false;
  if (viewerHint) {
    viewerHint.textContent = VIEWER_DEFAULT_HINT;
  }
  surfaceViewer.forceResize?.();
  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(() => {
      surfaceViewer?.forceResize?.();
    });
  }
  return surfaceViewer;
}

function createSurfaceViewer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  if ("outputColorSpace" in renderer) {
    renderer.outputColorSpace = THREE.SRGBColorSpace;
  } else if ("outputEncoding" in renderer) {
    renderer.outputEncoding = THREE.sRGBEncoding;
  }
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  const scene = new THREE.Scene();
  const ambient = new THREE.HemisphereLight(0xffffff, 0xb0bec5, 0.7);
  const directional = new THREE.DirectionalLight(0xffffff, 0.75);
  directional.position.set(1, 1.5, 1.2);
  scene.add(ambient);
  scene.add(directional);

  const group = new THREE.Group();
  scene.add(group);

  const baseNear = 0.1;
  const baseFar = 5000;
  const camera = new THREE.PerspectiveCamera(45, 1, baseNear, baseFar);
  camera.position.set(0, 0, 250);

  const controls = new THREE.OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 1;
  controls.maxDistance = Infinity;
  controls.screenSpacePanning = false;

  let userInteracted = false;
  controls.addEventListener("start", () => {
    userInteracted = true;
  });

  const loader = new THREE.STLLoader();
  const surfaces = new Map();

  const bounds = new THREE.Box3();
  const tempBounds = new THREE.Box3();
  const boundingSphere = new THREE.Sphere();
  const direction = new THREE.Vector3(1, 1.2, 1.3).normalize();
  const defaultCameraOffset = new THREE.Vector3(0, 0, 250);
  const colorPalette = [0x6f4bff, 0xff7a45, 0x32c8c0, 0xffc857, 0x6bd4ff, 0xef5da8, 0x7f9cff];

  let disposed = false;
  let generation = 0;
  let animationId = 0;

  const resizeTarget = canvas.parentElement || canvas;
  const resize = () => {
    if (disposed) return;
    const width = Math.max(1, resizeTarget.clientWidth);
    const height = Math.max(1, resizeTarget.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };

  let resizeObserver = null;
  if (typeof ResizeObserver === "function") {
    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(resizeTarget);
  }
  window.addEventListener("resize", resize);
  resize();

  const renderLoop = () => {
    if (disposed) {
      return;
    }
    controls.update();
    renderer.render(scene, camera);
    animationId = requestAnimationFrame(renderLoop);
  };
  animationId = requestAnimationFrame(renderLoop);

  function clearSurfaces() {
    surfaces.forEach(({ mesh, material, geometry }) => {
      group.remove(mesh);
      if (geometry && typeof geometry.dispose === "function") {
        geometry.dispose();
      }
      if (material && typeof material.dispose === "function") {
        material.dispose();
      }
    });
    surfaces.clear();
    bounds.makeEmpty();
    boundingSphere.set(new THREE.Vector3(), 0);
  }

  function computeBounds(visibleOnly) {
    bounds.makeEmpty();
    group.children.forEach((child) => {
      if (!child.isMesh) {
        return;
      }
      if (visibleOnly && !child.visible) {
        return;
      }
      tempBounds.setFromObject(child);
      bounds.union(tempBounds);
    });
    boundingSphere.setFromBox3(bounds);
    return !bounds.isEmpty();
  }

  function frameScene({ visibleOnly = true } = {}) {
    if (!computeBounds(visibleOnly)) {
      const fallback = visibleOnly ? defaultCameraOffset : controls.target;
      controls.target.set(0, 0, 0);
      camera.near = baseNear;
      camera.far = baseFar;
      camera.position.copy(fallback);
      camera.updateProjectionMatrix();
      controls.minDistance = 1;
      controls.maxDistance = 10000;
      controls.update();
      directional.position.set(1, 1.5, 1.2);
      directional.lookAt(controls.target);
      return;
    }

    const center = boundingSphere.center.clone();
    const radius = Math.max(boundingSphere.radius, 1);
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const distance = radius / Math.tan(fov / 2);
    const offset = distance * 1.35;
    const cameraOffset = direction.clone().multiplyScalar(offset);
    camera.position.copy(center).add(cameraOffset);

    const near = Math.max(0.1, radius * 0.02);
    const far = Math.max(offset + radius * 4, near + 1);
    camera.near = near;
    camera.far = far;
    camera.updateProjectionMatrix();

    controls.target.copy(center);
    controls.minDistance = Math.max(near * 0.5, radius * 0.05);
    controls.maxDistance = Math.max(far * 1.5, controls.minDistance * 10);
    controls.update();

    directional.position.copy(center).add(direction.clone().multiplyScalar(radius * 3));
    directional.lookAt(center);
  }

  function setSurfaces(files, hooks = {}) {
    generation += 1;
    const currentGen = generation;
    clearSurfaces();
    userInteracted = false;

    const descriptors = Array.isArray(files) ? files : [];
    if (!descriptors.length) {
      frameScene();
      hooks.onComplete?.();
      return;
    }

    let remaining = descriptors.length;

    descriptors.forEach((file, index) => {
      (async () => {
        try {
          const response = await fetch(file.url, {
            credentials: "same-origin",
            cache: "no-store",
            headers: {
              Accept: "model/stl,application/octet-stream,*/*",
            },
          });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          const buffer = await response.arrayBuffer();
          if (disposed || currentGen !== generation) {
            return;
          }
          let geometry;
          try {
            geometry = loader.parse(buffer);
          } catch (parseError) {
            throw new Error(`Unable to parse STL '${file.name}': ${parseError}`);
          }

          geometry.computeVertexNormals();

          const material = new THREE.MeshStandardMaterial({
            color: colorPalette[index % colorPalette.length],
            metalness: 0.05,
            roughness: 0.7,
            side: THREE.DoubleSide,
          });

          const surfaceKey = typeof file.viewerKey === "string" && file.viewerKey.length
            ? file.viewerKey
            : file.name;
          const mesh = new THREE.Mesh(geometry, material);
          mesh.name = surfaceKey;
          mesh.visible = true;
          group.add(mesh);

          surfaces.set(surfaceKey, { mesh, material, geometry });

          if (!userInteracted) {
            frameScene({ visibleOnly: false });
          }
          hooks.onMeshLoaded?.(surfaceKey);
        } catch (error) {
          if (disposed || currentGen !== generation) {
            return;
          }
          console.error("[viewer] Failed to load surface", file?.name, error);
          const surfaceKey =
            typeof file?.viewerKey === "string" && file.viewerKey.length ? file.viewerKey : file?.name ?? "unknown";
          hooks.onMeshError?.(surfaceKey, error);
        } finally {
          if (disposed || currentGen !== generation) {
            return;
          }
          remaining -= 1;
          if (remaining === 0) {
            hooks.onComplete?.();
            if (!userInteracted) {
              frameScene({ visibleOnly: true });
            }
          }
        }
      })().catch((error) => {
        console.error("[viewer] Unexpected loader error", error);
      });
    });
  }

  function toggleSurfaceVisibility(name, desiredVisibility) {
    const entry = surfaces.get(name);
    if (!entry) {
      return null;
    }
    const targetVisibility =
      typeof desiredVisibility === "boolean" ? desiredVisibility : !entry.mesh.visible;
    entry.mesh.visible = targetVisibility;
    return entry.mesh.visible;
  }

  function dispose() {
    if (disposed) {
      return;
    }
    disposed = true;
    cancelAnimationFrame(animationId);
    window.removeEventListener("resize", resize);
    if (resizeObserver) {
      resizeObserver.disconnect();
    }
    controls.dispose();
    clearSurfaces();
    renderer.dispose();
  }

  return {
    setSurfaces,
    toggleSurfaceVisibility,
    resetView() {
      userInteracted = false;
      frameScene({ visibleOnly: true });
    },
    forceResize: resize,
    dispose,
  };
}

function updateVisibilityButton(button, visible) {
  if (!button) {
    return;
  }
  button.classList.toggle("viewer-hidden", !visible);
  button.textContent = visible ? "Hide in preview" : "Show in preview";
  button.setAttribute("aria-pressed", visible ? "false" : "true");
}

function setStageStatus(stageName, status) {
  const el = stageLookup.get(stageName);
  if (!el) return;
  el.dataset.status = status;
}

function appendLog(line) {
  if (terminalLog.textContent.length > 0 && !terminalLog.textContent.endsWith("\n")) {
    terminalLog.textContent += "\n";
  }
  terminalLog.textContent += line + "\n";
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
    .map(([field, message]) => {
      const label = fieldLabels[field] ?? field;
      return `<li><strong>${label}</strong>: ${message}</li>`;
    })
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

if (randomToggleInput) {
  randomToggleInput.addEventListener("change", () => {
    if (randomToggleInput.checked && randomCountInput && !randomCountInput.value) {
      randomCountInput.value = "1";
    }
    updateRandomFieldsVisibility();
  });
}

function handleEvent(data) {
  switch (data.type) {
    case "init":
      resetStages();
      if (stageOrder.length) {
        setStageStatus(stageOrder[0], "running");
      }
      setCommandPreview(data.command);
      appendLog("# Starting pipeline");
      resetSimulationProgress();
      break;
    case "log":
      appendLog(data.message);
      break;
    case "stage":
      setStageStatus(data.stage, data.status);
      break;
    case "cancelled":
      wasCancelled = true;
      setStatus("Cancellation requested. Waiting for the pipeline to stop…", "error");
      cancelButton.disabled = true;
      appendLog("# Cancellation requested");
      break;
    case "error":
      appendLog(`[error] ${data.message}`);
      setStatus(data.message, "error");
      break;
    case "progress":
      updateSimulationProgress(data);
      break;
    case "done": {
      const success = Number(data.returncode) === 0;
      setStatus(
        wasCancelled
          ? "Pipeline cancelled."
          : success
            ? "Pipeline completed successfully."
            : "Pipeline failed. Inspect the log above.",
        wasCancelled ? "error" : success ? "success" : "error",
      );
      if (wasCancelled) {
        appendLog("# Pipeline cancelled ✖");
      } else if (success) {
        appendLog("# Pipeline finished ✔");
        void loadOutputs(activeRunId);
      } else {
        appendLog("# Pipeline finished with errors ✖");
      }
      runButton.disabled = false;
      runInProgress = false;
      cancelButton.hidden = true;
      cancelButton.disabled = false;
      activeRunId = null;
      break;
    }
    case "close":
      closeEventSource();
      runButton.disabled = false;
      runInProgress = false;
      cancelButton.hidden = true;
      cancelButton.disabled = false;
      activeRunId = null;
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
  if (!currentCase) {
    renderErrors({ case: "Please select a scenario before running the pipeline." });
    setStatus("Please choose a scenario first.", "error");
    return;
  }

  clearErrors();
  setStatus("");
  commandPreview.textContent = "";
  terminalLog.textContent = "";
  resetStages();
  resetSimulationProgress();
  ensureOutputsHidden();
  resetCancellationState();

  const extractRaw = form.extractYears.value.trim();
  const simulationRaw = form.simulationYears.value.trim();
  const mesh = form.mesh.value.trim();
  const procs = form.procs.value.trim();

  if (!extractRaw) {
    renderErrors({ extractYears: "Please provide one or more simulation times in years." });
    return;
  }

  const extractTokens = extractRaw
    .split(/[,\s]+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 0);

  if (!extractTokens.length) {
    renderErrors({ extractYears: "Please provide one or more simulation times in years." });
    return;
  }

  const seenMonths = new Set();
  const extractYearsValues = [];

  for (const token of extractTokens) {
    const numeric = Number(token);
    if (Number.isNaN(numeric) || numeric < 0) {
      renderErrors({ extractYears: "Each time must be a non-negative number." });
      return;
    }
    const monthsValue = numeric * 12;
    const roundedMonths = Math.round(monthsValue);
    if (Math.abs(monthsValue - roundedMonths) > 1e-6) {
      renderErrors({ extractYears: "Each time must align with whole-month steps (increments of 1/12 year)." });
      return;
    }
    if (seenMonths.has(roundedMonths)) {
      continue;
    }
    seenMonths.add(roundedMonths);
    extractYearsValues.push(roundedMonths / 12);
  }

  if (!extractYearsValues.length) {
    renderErrors({ extractYears: "Please provide at least one unique simulation time." });
    return;
  }

  const payload = { extractYears: extractYearsValues, case: currentCase };

  const randomEnabled =
    currentCase === "breast" && Boolean(randomToggleInput?.checked) && !randomToggleField?.hidden;
  if (randomEnabled) {
    const countValue = Number(randomCountInput?.value);
    if (!Number.isFinite(countValue) || countValue <= 0) {
      renderErrors({ randomSimulationCount: "Please provide a positive integer." });
      return;
    }
    if (!Number.isInteger(countValue)) {
      renderErrors({ randomSimulationCount: "Number of simulations must be an integer." });
      return;
    }
    payload.randomBreast = true;
    payload.randomSimulationCount = countValue;
  }

  if (simulationRaw) {
    const simYears = Number(simulationRaw);
    if (Number.isNaN(simYears) || simYears < 0) {
      renderErrors({ simulationYears: "Simulation duration must be a non-negative number." });
      return;
    }
    payload.simulationYears = simYears;
  }
  if (mesh) payload.mesh = mesh;
  if (procs) payload.procs = procs;

  closeEventSource();
  runButton.disabled = true;
  runInProgress = true;
  appendLog("# Submitting pipeline run");
  const caseLabel = resolveCaseLabel(currentCase) || currentCase;
  appendLog(`# Scenario: ${caseLabel}`);
  appendLog(`# Requested times (years): ${extractYearsValues.map(formatYearsValue).join(", ")}`);

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

    activeRunId = data.runId;
    wasCancelled = false;
    cancelButton.hidden = false;
    cancelButton.disabled = false;
    setStatus("Pipeline launched. Tracking live progress…");

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
cancelButton.addEventListener("click", cancelPipeline);

window.addEventListener("beforeunload", closeEventSource);

// Initialise defaults on load.
resetStages();
toggleTerminalBtn.textContent = "Show log";
const fieldLabels = {
  case: "Scenario",
  extractYears: "Simulation time(s) (years)",
  simulationYears: "Simulation horizon (years)",
  mesh: "Mesh path",
  procs: "MPI processes",
  randomBreast: "Random tumour placement",
  randomSimulationCount: "Number of simulations",
  lobulePoints: "Lobule coordinates file",
  randomSeed: "Random seed",
};
resetSimulationProgress();
ensureOutputsHidden();
resetCancellationState();
initMobileNavigation();
updateNavAvailability();
updateRandomFieldsVisibility();
void loadCases();
showCaseSelection();

const percentFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 0,
});

const numberFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 2,
});

function formatBytes(bytes) {
  if (typeof bytes !== "number" || Number.isNaN(bytes) || bytes < 0) {
    return "";
  }
  const UNITS = ["B", "KB", "MB", "GB"];
  let idx = 0;
  let value = bytes;
  while (value >= 1024 && idx < UNITS.length - 1) {
    value /= 1024;
    idx += 1;
  }
  return `${numberFormatter.format(value)} ${UNITS[idx]}`;
}

function formatYearsValue(years) {
  if (typeof years !== "number" || Number.isNaN(years)) {
    return "";
  }
  const text = years.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
  return text || "0";
}

function updateSimulationProgress(data) {
  if (!data || data.stage !== "Running simulation") {
    return;
  }

  const fractionRaw = typeof data.fraction === "number" ? data.fraction : null;
  let fraction = fractionRaw;
  if (fraction === null && typeof data.currentStep === "number" && typeof data.totalSteps === "number" && data.totalSteps > 0) {
    fraction = data.currentStep / data.totalSteps;
  }
  if (fraction !== null) {
    const clamped = Math.max(0, Math.min(fraction, 1));
    simulationProgress.hidden = false;
    simulationProgressFill.style.transform = `scaleX(${clamped})`;
    simulationProgressFill.parentElement?.setAttribute("aria-valuenow", percentFormatter.format(clamped * 100));
  } else if (simulationProgress.hidden) {
    simulationProgress.hidden = false;
  }

  if (typeof data.currentStep === "number" && typeof data.totalSteps === "number") {
    simulationProgressCount.textContent = `Step ${data.currentStep} / ${data.totalSteps}`;
  } else if (typeof data.currentStep === "number") {
    simulationProgressCount.textContent = `Step ${data.currentStep}`;
  } else {
    simulationProgressCount.textContent = "";
  }

  if (typeof data.currentTime === "number") {
    const formattedCurrent = numberFormatter.format(data.currentTime);
    if (typeof data.targetTime === "number" && data.targetTime > 0) {
      simulationProgressMeta.textContent = `Time ${formattedCurrent} / ${numberFormatter.format(data.targetTime)} years`;
    } else {
      simulationProgressMeta.textContent = `Time ${formattedCurrent} years`;
    }
  } else {
    simulationProgressMeta.textContent = "";
  }
}

async function loadOutputs(runId) {
  if (!runId) {
    return;
  }

  try {
    const response = await fetch(`/api/run/${runId}/outputs`);
    if (!response.ok) {
      if (response.status === 404) {
        return;
      }
      throw new Error(`Failed to fetch outputs: ${response.status}`);
    }
    const payload = await response.json();
    const files = Array.isArray(payload?.files) ? payload.files : [];
    outputsCard.hidden = false;

    outputsList.innerHTML = "";
    if (!files.length) {
      outputsActions.hidden = true;
      outputsEmpty.hidden = false;
      outputsEmpty.textContent = "No STL files were produced for this run.";
      destroySurfaceViewer();
      updateNavAvailability();
      return;
    }

    outputsEmpty.hidden = true;
    outputsActions.hidden = false;

    if (payload.downloadAllUrl) {
      downloadAllLink.href = payload.downloadAllUrl;
      downloadAllLink.download = `${runId}_outputs.zip`;
    } else {
      downloadAllLink.href = "#";
      downloadAllLink.removeAttribute("download");
    }
    updateNavAvailability();

    const viewer = ensureSurfaceViewer();
    const viewerAvailable = Boolean(viewer);
    if (viewerAvailable && viewerHint) {
      viewerHint.textContent = VIEWER_LOADING_HINT;
    }
    if (resetViewerButton) {
      resetViewerButton.disabled = !viewerAvailable;
    }

    const toggleButtons = new Map();
    let loadedSurfaces = 0;
    let failedSurfaces = 0;

    files.forEach((file) => {
      const displayName =
        typeof file.displayName === "string" && file.displayName.length ? file.displayName : file.name;
      const viewerKey =
        typeof file.viewerKey === "string" && file.viewerKey.length ? file.viewerKey : displayName;
      const downloadName =
        typeof file.downloadName === "string" && file.downloadName.length ? file.downloadName : file.name;
      const item = document.createElement("li");
      item.className = "output-item";

      const details = document.createElement("div");
      details.className = "output-details";

      const name = document.createElement("span");
      name.className = "output-name";
      name.textContent = displayName;
      details.appendChild(name);

      if (typeof file.simulation === "number" && !Number.isNaN(file.simulation)) {
        const simMeta = document.createElement("span");
        simMeta.className = "output-meta";
        simMeta.textContent = `Simulation ${file.simulation}`;
        details.appendChild(simMeta);
      }

      if (typeof file.year === "number" && !Number.isNaN(file.year)) {
        const yearMeta = document.createElement("span");
        yearMeta.className = "output-meta";
        yearMeta.textContent = `${formatYearsValue(file.year)} years`;
        details.appendChild(yearMeta);
      }

      if (typeof file.size === "number" && file.size >= 0) {
        const meta = document.createElement("span");
        meta.className = "output-meta";
        meta.textContent = formatBytes(file.size);
        details.appendChild(meta);
      }

      const actions = document.createElement("div");
      actions.className = "output-actions";

      const link = document.createElement("a");
      link.className = "ghost-button";
      link.href = file.url;
      link.textContent = "Download";
      link.setAttribute("download", downloadName ?? "");
      actions.appendChild(link);

      if (viewerAvailable) {
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "ghost-button visibility-toggle";
        toggle.disabled = true;
        toggle.textContent = "Hide in preview";
        toggle.setAttribute("aria-label", `Toggle ${displayName} visibility in preview`);
        toggleButtons.set(viewerKey, toggle);
        toggle.addEventListener("click", () => {
          if (!surfaceViewer) {
            return;
          }
          const visible = surfaceViewer.toggleSurfaceVisibility(viewerKey);
          if (visible === null) {
            return;
          }
          updateVisibilityButton(toggle, visible);
        });
        actions.appendChild(toggle);
      }

      item.append(details, actions);
      outputsList.appendChild(item);
    });

    if (viewerAvailable && viewer) {
      viewer.forceResize?.();
      viewer.setSurfaces(files, {
        onMeshLoaded(name) {
          loadedSurfaces += 1;
          const button = toggleButtons.get(name);
          if (button) {
            button.disabled = false;
            updateVisibilityButton(button, true);
          }
        },
        onMeshError(name) {
          failedSurfaces += 1;
          const button = toggleButtons.get(name);
          if (button) {
            button.disabled = true;
            button.textContent = "Preview unavailable";
            button.classList.add("viewer-error");
            button.removeAttribute("aria-pressed");
          }
        },
        onComplete() {
          viewer.forceResize?.();
          if (viewerHint) {
            if (loadedSurfaces > 0) {
              viewerHint.textContent = VIEWER_DEFAULT_HINT;
            } else if (failedSurfaces > 0) {
              viewerHint.textContent = VIEWER_DATA_ERROR_HINT;
            } else {
              viewerHint.textContent = VIEWER_DEFAULT_HINT;
            }
          }
          if (resetViewerButton) {
            resetViewerButton.disabled = loadedSurfaces === 0;
          }
        },
      });
    } else {
      destroySurfaceViewer();
      if (viewerHint) {
        viewerHint.textContent = VIEWER_LIBRARY_ERROR_HINT;
      }
    }
  } catch (error) {
    console.error(error);
    outputsCard.hidden = false;
    outputsActions.hidden = true;
    outputsEmpty.hidden = false;
    outputsEmpty.textContent = "Unable to load outputs. Check the server logs.";
    destroySurfaceViewer();
    updateNavAvailability();
  }
}

async function cancelPipeline() {
  if (!activeRunId || !runInProgress || wasCancelled) {
    return;
  }

  cancelButton.disabled = true;

  try {
    const response = await fetch(`/api/run/${activeRunId}/cancel`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`Cancel request failed with status ${response.status}`);
    }
  } catch (error) {
    console.error(error);
    setStatus("Unable to cancel the pipeline. Check the server logs.", "error");
    cancelButton.disabled = false;
  }
}

function initMobileNavigation() {
  if (!mobileNavButtons.length) {
    return;
  }

  function setActiveMobileNav(targetId) {
    if (!targetId) {
      return;
    }
    if (lastActiveMobileSection === targetId) {
      return;
    }
    lastActiveMobileSection = targetId;
    mobileNavButtons.forEach((button) => {
      button.classList.toggle("active", button.dataset.target === targetId);
    });
  }

  mobileNavButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const targetId = button.dataset.target;
      if (!targetId) {
        return;
      }
      if (targetId === "case-card") {
        showCaseSelection({ scroll: true });
        setActiveMobileNav(targetId);
        return;
      }
      const section = document.getElementById(targetId);
      if (!section) {
        return;
      }
      const offset = 20;
      const yPosition = section.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({
        top: Math.max(0, yPosition),
        behavior: "smooth",
      });
      setActiveMobileNav(targetId);
    });
  });

  if (!("IntersectionObserver" in window)) {
    const defaultTarget = mobileNavButtons[0]?.dataset.target;
    if (defaultTarget) {
      setActiveMobileNav(defaultTarget);
    }
    return;
  }

  const sections = mobileNavButtons
    .map((button) => document.getElementById(button.dataset.target ?? ""))
    .filter(Boolean);

  if (!sections.length) {
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting && !entry.target.hidden)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
      if (!visible.length) {
        return;
      }
      const topSection = visible[0].target;
      if (topSection.id) {
        setActiveMobileNav(topSection.id);
      }
    },
    {
      threshold: [0.25, 0.5, 0.75],
      rootMargin: "-35% 0px -45% 0px",
    },
  );

  sections.forEach((section) => observer.observe(section));
  setActiveMobileNav(sections[0].id);
}
