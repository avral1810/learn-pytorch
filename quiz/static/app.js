let editor = null;
let questions = [];
let currentIndex = 0;

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

function buildEditor() {
  const textarea = document.getElementById("code-editor");
  if (!textarea) {
    return;
  }

  if (window.CodeMirror) {
    editor = window.CodeMirror.fromTextArea(textarea, {
      mode: "python",
      theme: "material",
      lineNumbers: true,
      indentUnit: 4,
      tabSize: 4,
    });
  } else {
    editor = {
      getValue: () => textarea.value,
      setValue: (value) => {
        textarea.value = value;
      },
      refresh: () => {},
    };
  }
}

function renderCurrentQuestion(resetCode = false) {
  const question = currentQuestion();
  if (!question) {
    setSummaryHtml("<p>No questions configured for this chapter yet.</p>");
    return;
  }

  document.getElementById("question-title").textContent = question.title;
  document.getElementById("question-prompt").textContent = question.prompt;

  const examples = document.getElementById("visible-examples");
  examples.innerHTML = "";
  question.visible_examples.forEach((example) => {
    const item = document.createElement("li");
    item.textContent = example;
    examples.appendChild(item);
  });

  if (resetCode && editor) {
    editor.setValue(question.starter_code);
    if (editor.refresh) {
      editor.refresh();
    }
  }

  setSummaryHtml(`Question ${currentIndex + 1} of ${questions.length}`);
  setPlainOutput("stdout-output", "");
  setPlainOutput("stderr-output", "");
}

function formatResults(payload) {
  if (payload.error) {
    return `<p class="status-fail">${payload.error}</p>`;
  }

  const statusClass = payload.ok ? "status-pass" : "status-fail";
  const label = payload.mode === "submit" ? "Hidden tests" : "Visible tests";
  const items = payload.results
    .map((result) => {
      const status = result.passed ? "PASS" : "FAIL";
      const message = result.message ? ` - ${result.message}` : "";
      return `<li><strong>${status}</strong> ${result.name}${message}</li>`;
    })
    .join("");

  return `<p class="${statusClass}">${label}: ${payload.passed}/${payload.total} passed</p><ul>${items}</ul>`;
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
  setPlainOutput("stdout-output", payload.stdout);
  setPlainOutput("stderr-output", payload.stderr || payload.error || "");
}

function bindQuizUi() {
  document.getElementById("run-button").addEventListener("click", () => sendCode("run"));
  document.getElementById("submit-button").addEventListener("click", () => sendCode("submit"));
  document.getElementById("reset-button").addEventListener("click", () => renderCurrentQuestion(true));
  document.getElementById("prev-question").addEventListener("click", () => {
    if (currentIndex > 0) {
      currentIndex -= 1;
      renderCurrentQuestion(true);
    }
  });
  document.getElementById("next-question").addEventListener("click", () => {
    if (currentIndex < questions.length - 1) {
      currentIndex += 1;
      renderCurrentQuestion(true);
    } else {
      setSummaryHtml('<p class="status-pass">You finished this chapter quiz. Use the next chapter link in the sidebar to continue.</p>');
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
