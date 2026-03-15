let editor = null;
let tutorialViewer = null;
let playgroundEditor = null;
let questions = [];
let currentIndex = 0;
let hintCounts = {};
let lastRunPassed = false;
let lastRunQuestionId = null;

function chapterUi() {
  return (window.__CHAPTER_DATA__ && window.__CHAPTER_DATA__.ui) || {};
}

function findActionElements(action) {
  return Array.from(document.querySelectorAll(`[data-action="${action}"]`));
}

function triggerRun() {
  return sendCode("run");
}

function triggerSubmit() {
  return sendCode("submit");
}

function triggerPrimaryShortcut() {
  const question = currentQuestion();
  if (question && lastRunPassed && lastRunQuestionId === question.id) {
    return triggerSubmit();
  }
  return triggerRun();
}

function updateNavigationState() {
  const prevButton = document.getElementById("prev-question");
  const nextButtons = findActionElements("next-question");
  const nextInlineButton = document.getElementById("next-question-inline");
  const progress = document.getElementById("question-progress");
  const question = currentQuestion();

  if (!prevButton || !nextInlineButton || !progress || !question) {
    return;
  }

  prevButton.disabled = currentIndex === 0;
  nextButtons.forEach((button) => { button.disabled = questions.length === 0; });
  nextInlineButton.disabled = questions.length === 0;
  progress.textContent = `${currentIndex + 1} / ${questions.length}`;
  prevButton.textContent = "Previous";
  const nextLabel = currentIndex === questions.length - 1
    ? "Finish Chapter"
    : "Next";
  nextButtons.forEach((button) => { button.textContent = nextLabel; });
}

function hideInlineNext() {
  const button = document.getElementById("next-question-inline");
  if (button) {
    button.classList.add("is-hidden");
  }
}

function showInlineNext() {
  const button = document.getElementById("next-question-inline");
  if (button) {
    button.classList.remove("is-hidden");
  }
}

function goToNextQuestion() {
  if (currentIndex < questions.length - 1) {
    currentIndex += 1;
    renderCurrentQuestion(true);
  } else {
    setSummaryHtml('<p class="status-pass">You finished this chapter quiz. Use the next chapter link in the sidebar to continue.</p>');
    updateNavigationState();
    showInlineNext();
  }
}

function openPdfDrawer(url) {
  const drawer = document.getElementById("pdf-drawer");
  const frame = document.getElementById("pdf-drawer-frame");
  if (!drawer || !frame) {
    return;
  }

  frame.src = url;
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
}

function closePdfDrawer() {
  const drawer = document.getElementById("pdf-drawer");
  const frame = document.getElementById("pdf-drawer-frame");
  if (!drawer || !frame) {
    return;
  }

  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  frame.src = "about:blank";
}

function openTutorialDrawer() {
  const drawer = document.getElementById("tutorial-drawer");
  if (!drawer) {
    return;
  }

  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  window.requestAnimationFrame(() => {
    if (tutorialViewer && tutorialViewer.refresh) tutorialViewer.refresh();
    if (playgroundEditor && playgroundEditor.refresh) playgroundEditor.refresh();
  });
  window.setTimeout(() => {
    if (tutorialViewer && tutorialViewer.refresh) tutorialViewer.refresh();
    if (playgroundEditor && playgroundEditor.refresh) playgroundEditor.refresh();
  }, 220);
}

function closeTutorialDrawer() {
  const drawer = document.getElementById("tutorial-drawer");
  if (!drawer) {
    return;
  }

  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
}

function currentQuestion() {
  return questions[currentIndex];
}

function setSummaryHtml(html) {
  const element = document.getElementById("summary-output");
  if (element) {
    element.innerHTML = html;
  }
}

function setPlainOutput(id, text) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = text || "";
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatSnapshot(value) {
  if (value === null || value === undefined) {
    return "Not available";
  }
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value, null, 2);
}

function renderResultDetails(result) {
  const inputSnapshot = escapeHtml(formatSnapshot(result.input_snapshot));
  const expectedOutput = escapeHtml(formatSnapshot(result.expected_output));
  const actualOutput = escapeHtml(formatSnapshot(result.actual_output));

  return [
    '<div class="test-details">',
    '<div class="test-detail-block"><strong>Input</strong><pre class="test-detail-pre">', inputSnapshot, "</pre></div>",
    '<div class="test-detail-block"><strong>Expected Output</strong><pre class="test-detail-pre">', expectedOutput, "</pre></div>",
    '<div class="test-detail-block"><strong>Your Output</strong><pre class="test-detail-pre">', actualOutput, "</pre></div>",
    "</div>",
  ].join("");
}

function bindHintButtons() {
  return;
}

function applyHint(key) {
  if (!key) {
    return;
  }
  const detail = document.querySelector(`[data-result-detail="${key}"]`);
  const counter = document.querySelector(`[data-hint-counter="${key}"]`);
  hintCounts[key] = (hintCounts[key] || 0) + 1;

  if (counter) {
    const remaining = Math.max(0, 3 - hintCounts[key]);
    counter.textContent = remaining > 0
      ? `Hint used ${hintCounts[key]} time(s). Reveal after ${remaining} more.`
      : "Input and expected output revealed below.";
  }

  if (detail && hintCounts[key] >= 3) {
    detail.hidden = false;
  }
}

function applyGlobalHint() {
  const detailShells = Array.from(document.querySelectorAll("[data-result-detail]"));
  if (detailShells.length === 0) {
    setSummaryHtml('<p class="status-fail">Run or submit the current question first, then use Hint.</p>');
    return;
  }

  detailShells.forEach((detail) => applyHint(detail.dataset.resultDetail));
}

function toggleEditorComment() {
  if (!editor) {
    return;
  }
  if (typeof editor.execCommand === "function") {
    editor.execCommand("toggleComment");
  }
}

function buildEditor() {
  const textarea = document.getElementById("code-editor");
  const tutorialTextarea = document.getElementById("tutorial-script-editor");
  const playgroundTextarea = document.getElementById("playground-editor");
  if (!textarea) {
    return;
  }

  if (window.CodeMirror) {
    if (tutorialTextarea) {
      tutorialViewer = window.CodeMirror.fromTextArea(tutorialTextarea, {
        mode: "python",
        theme: "material",
        lineNumbers: true,
        readOnly: "nocursor",
        lineWrapping: false,
      });
    }
    if (playgroundTextarea) {
      playgroundEditor = window.CodeMirror.fromTextArea(playgroundTextarea, {
        mode: "python",
        theme: "material",
        lineNumbers: true,
        indentUnit: 4,
        tabSize: 4,
      });
    }
    editor = window.CodeMirror.fromTextArea(textarea, {
      mode: "python",
      theme: "material",
      lineNumbers: true,
      indentUnit: 4,
      tabSize: 4,
    });
    editor.addKeyMap({
      "Cmd-Enter": () => { triggerPrimaryShortcut(); },
      "Ctrl-Enter": () => { triggerPrimaryShortcut(); },
      "Shift-Enter": () => { triggerSubmit(); },
      "Cmd-/": () => { toggleEditorComment(); },
      "Ctrl-/": () => { toggleEditorComment(); },
    });
    editor.on("change", () => {
      const question = currentQuestion();
      lastRunPassed = false;
      lastRunQuestionId = question ? question.id : null;
    });
  } else {
    tutorialViewer = tutorialTextarea ? { refresh: () => {} } : null;
    playgroundEditor = playgroundTextarea ? {
      getValue: () => playgroundTextarea.value,
      setValue: (value) => { playgroundTextarea.value = value; },
      refresh: () => {},
    } : null;
    editor = {
      getValue: () => textarea.value,
      setValue: (value) => { textarea.value = value; },
      refresh: () => {},
    };
  }
}

function renderCurrentQuestion(resetCode = false) {
  const question = currentQuestion();
  if (!question) {
    setSummaryHtml("<p>No questions configured for this chapter yet.</p>");
    updateNavigationState();
    return;
  }

  document.getElementById("question-title").textContent = question.title;
  document.getElementById("question-prompt").textContent = question.prompt;
  document.getElementById("visible-examples-title").textContent = question.visible_examples_title || chapterUi().question_prompt_title || "Visible Checks";
  document.getElementById("answer-editor-title").textContent = question.answer_editor_title || chapterUi().answer_editor_title || "Your Answer";
  document.getElementById("answer-editor-note").textContent = question.answer_editor_note || chapterUi().answer_editor_note || "Write the code for this question here";

  const examples = document.getElementById("visible-examples");
  examples.innerHTML = "";
  question.visible_examples.forEach((example) => {
    const item = document.createElement("li");
    item.textContent = example;
    examples.appendChild(item);
  });

  if (resetCode && editor) {
    editor.setValue(question.starter_code);
    if (editor.refresh) editor.refresh();
  }

  if (tutorialViewer && tutorialViewer.refresh) tutorialViewer.refresh();
  if (playgroundEditor && playgroundEditor.refresh) playgroundEditor.refresh();

  setSummaryHtml(`Question ${currentIndex + 1} of ${questions.length}<br><strong>${question.title}</strong>`);
  setPlainOutput("stdout-output", "");
  setPlainOutput("stderr-output", "");
  hintCounts = {};
  lastRunPassed = false;
  lastRunQuestionId = question.id;
  hideInlineNext();
  updateNavigationState();
}

function formatResults(payload) {
  if (payload.error) {
    return `<p class="status-fail">${payload.error}</p>`;
  }

  hintCounts = {};
  const statusClass = payload.ok ? "status-pass" : "status-fail";
  const label = payload.mode === "submit" ? "Hidden tests" : "Visible tests";
  const items = payload.results
    .map((result, index) => {
      const resultKey = `${payload.mode}-${currentIndex}-${index}`;
      const status = result.passed ? "PASS" : "FAIL";
      const message = result.message ? ` - ${result.message}` : "";
      const showDetailsImmediately = payload.mode === "submit" && result.passed;
      const safeHint = escapeHtml(result.hint || "Compare your output to the expected behavior.");
      return `
        <li class="test-result-item">
          <div><strong>${status}</strong> ${escapeHtml(result.name)}${escapeHtml(message)}</div>
          <div class="test-hint-row">
            <span class="hint-text">${safeHint}</span>
          </div>
          <div class="hint-counter" data-hint-counter="${resultKey}">Press hint 3 times to reveal the test input and expected output.</div>
          <div class="test-detail-shell" data-result-detail="${resultKey}" ${showDetailsImmediately ? "" : "hidden"}>
            ${renderResultDetails(result)}
          </div>
        </li>
      `;
    })
    .join("");

  return `<p class="${statusClass}">${label}: ${payload.passed}/${payload.total} passed</p><ul>${items}</ul>`;
}

function isSuccessfulSubmit(payload, mode) {
  if (mode !== "submit" || !payload || payload.error) {
    return false;
  }
  if (payload.ok === true) {
    return true;
  }
  if (typeof payload.passed === "number" && typeof payload.total === "number" && payload.total > 0) {
    return payload.passed === payload.total;
  }
  return false;
}

async function sendCode(mode) {
  const chapter = window.__CHAPTER_DATA__;
  const question = currentQuestion();
  const response = await fetch(`/api/${mode}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      chapter_id: chapter.id,
      question_id: question.id,
      code: editor.getValue(),
    }),
  });

  const payload = await response.json();
  setSummaryHtml(formatResults(payload));
  bindHintButtons();
  setPlainOutput("stdout-output", payload.stdout);
  setPlainOutput("stderr-output", payload.stderr || payload.error || "");
  if (mode === "run") {
    lastRunPassed = Boolean(payload.ok);
    lastRunQuestionId = question.id;
  } else if (mode === "submit") {
    lastRunQuestionId = question.id;
  }
  if (isSuccessfulSubmit(payload, mode)) {
    showInlineNext();
  } else {
    hideInlineNext();
  }
}

async function runPlayground() {
  if (!playgroundEditor) {
    return;
  }
  const response = await fetch("/api/playground", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: playgroundEditor.getValue() }),
  });

  const payload = await response.json();
  setPlainOutput("playground-stdout", payload.stdout);
  setPlainOutput("playground-stderr", payload.stderr || payload.error || "");
}

function bindQuizUi() {
  const actionHandlers = {
    "run-code": triggerRun,
    "submit-code": triggerSubmit,
    "show-hint": applyGlobalHint,
    "reset-question": () => renderCurrentQuestion(true),
    "previous-question": () => {
      if (currentIndex > 0) {
        currentIndex -= 1;
        renderCurrentQuestion(true);
      }
    },
    "next-question": goToNextQuestion,
    "open-tutorial-drawer": openTutorialDrawer,
    "open-pdf-drawer": (event) => openPdfDrawer(event.currentTarget.dataset.pdfUrl),
    "run-playground": runPlayground,
    "reset-playground": () => {
      if (!playgroundEditor) {
        return;
      }
      playgroundEditor.setValue(window.__CHAPTER_DATA__.playground_starter || "");
      setPlainOutput("playground-stdout", "");
      setPlainOutput("playground-stderr", "");
    },
  };

  Object.entries(actionHandlers).forEach(([action, handler]) => {
    findActionElements(action).forEach((element) => {
      element.addEventListener("click", handler);
    });
  });

  const closeButton = document.getElementById("tutorial-drawer-close");
  if (closeButton) closeButton.addEventListener("click", closeTutorialDrawer);

  const pdfCloseButton = document.getElementById("pdf-drawer-close");
  if (pdfCloseButton) pdfCloseButton.addEventListener("click", closePdfDrawer);

  window.addEventListener("keydown", (event) => {
    const activeElement = document.activeElement;
    const codeMirrorActive = activeElement && activeElement.closest && activeElement.closest(".CodeMirror");
    if (!codeMirrorActive) {
      return;
    }

    if (event.key === "Enter") {
      if (event.shiftKey) {
        event.preventDefault();
        triggerSubmit();
        return;
      }
      if (event.metaKey || event.ctrlKey) {
        event.preventDefault();
        triggerPrimaryShortcut();
      }
      return;
    }

  });
}

window.addEventListener("DOMContentLoaded", () => {
  if (!window.__CHAPTER_DATA__) {
    return;
  }

  questions = window.__CHAPTER_DATA__.questions || [];
  buildEditor();
  bindQuizUi();
  renderCurrentQuestion(true);
});
