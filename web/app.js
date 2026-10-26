const form = document.querySelector("#chat-form");
const promptField = document.querySelector("#prompt");
const sendButton = document.querySelector("#send");
const endSessionButton = document.querySelector("#end-session");
const messagesElement = document.querySelector("#messages");
const notice = document.querySelector("#notice");
const proposalSection = document.querySelector("#proposals");
const proposalList = document.querySelector("#proposal-list");
const proposalNotice = document.querySelector("#proposal-notice");
const saveSelectedButton = document.querySelector("#save-selected");
const memoryAddFileSelect = document.querySelector("#memory-add-file");
const memoryLines = document.querySelector("#memory-lines");
const memoryNotice = document.querySelector("#memory-notice");
const addMemoryForm = document.querySelector("#add-memory-form");
const newMemoryField = document.querySelector("#new-memory");
const activityEntries = document.querySelector("#activity-entries");
const activityNotice = document.querySelector("#activity-notice");
const conversation = [];
let proposals = [];
let memoryData = [];
let approvalPending = false;

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  let result;
  try {
    result = await response.json();
  } catch {
    throw new Error(`Request failed (${response.status}).`);
  }
  if (!response.ok) {
    throw new Error(result.detail || `Request failed (${response.status}).`);
  }
  return result;
}

function addMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role === "assistant" ? "assistant-message" : "user-message"}`;
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "assistant" ? "LIFEOS" : "YOU";
  const body = document.createElement("p");
  body.textContent = text;
  article.append(label, body);
  messagesElement.append(article);
  messagesElement.scrollTop = messagesElement.scrollHeight;
  return body;
}

async function readAgentStream(response, answerElement) {
  if (!response.ok || !response.body) {
    throw new Error(`Request failed (${response.status}).`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  let completed = false;
  let approval = null;
  let answer = "";
  while (true) {
    const { value, done } = await reader.read();
    pending += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = pending.split("\n");
    pending = lines.pop() || "";
    for (const line of lines) {
      if (!line) continue;
      const eventData = JSON.parse(line);
      if (eventData.type === "delta") {
        answer += eventData.text;
        answerElement.textContent = answer;
        messagesElement.scrollTop = messagesElement.scrollHeight;
      } else if (eventData.type === "error") {
        throw new Error(eventData.message);
      } else if (eventData.type === "approval_required") {
        approval = eventData;
      } else if (eventData.type === "done") {
        completed = true;
      }
    }
    if (done) break;
  }
  if (pending) {
    const eventData = JSON.parse(pending);
    if (eventData.type === "error") throw new Error(eventData.message);
    if (eventData.type === "approval_required") approval = eventData;
    if (eventData.type === "done") completed = true;
  }
  return { completed, approval, answer };
}

function addApprovalCard(approval, answerElement) {
  const card = document.createElement("article");
  card.className = "approval-card";
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = "YOUR APPROVAL NEEDED";
  const description = document.createElement("p");
  description.className = "approval-description";
  description.textContent = approval.description;
  const toolName = document.createElement("p");
  toolName.className = "approval-tool";
  toolName.textContent = `${approval.tool} · ${JSON.stringify(approval.input)}`;
  const buttons = document.createElement("div");
  buttons.className = "proposal-actions";
  const approveButton = document.createElement("button");
  approveButton.type = "button";
  approveButton.textContent = "Approve";
  const denyButton = document.createElement("button");
  denyButton.type = "button";
  denyButton.className = "secondary-button";
  denyButton.textContent = "Deny";
  buttons.append(approveButton, denyButton);
  card.append(label, description, toolName, buttons);
  messagesElement.append(card);
  messagesElement.scrollTop = messagesElement.scrollHeight;

  async function decide(approved) {
    approveButton.disabled = true;
    denyButton.disabled = true;
    try {
      const response = await fetch("/api/approval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approval_id: approval.approval_id, approved }),
      });
      const result = await readAgentStream(response, answerElement);
      if (result.approval) {
        addApprovalCard(result.approval, answerElement);
        return;
      }
      if (!result.completed) throw new Error("The response stream ended unexpectedly.");
      conversation.push({ role: "assistant", content: answerElement.textContent || "" });
      approvalPending = false;
      sendButton.disabled = false;
      endSessionButton.disabled = false;
      card.remove();
      await loadActivity();
      notice.textContent = "Tool action finished. Your decision and the result were added to the activity log.";
      promptField.focus();
    } catch (error) {
      description.textContent = `Could not complete this approval: ${error.message}`;
      approveButton.disabled = false;
      denyButton.disabled = false;
    }
  }
  approveButton.addEventListener("click", () => decide(true));
  denyButton.addEventListener("click", () => decide(false));
}

function renderActivity(entries) {
  activityEntries.replaceChildren();
  if (entries.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No tool calls yet.";
    activityEntries.append(empty);
    return;
  }
  for (const entry of [...entries].reverse()) {
    const card = document.createElement("article");
    card.className = "activity-entry";
    const heading = document.createElement("div");
    heading.className = "activity-heading";
    const tool = document.createElement("strong");
    tool.textContent = entry.tool;
    const decision = document.createElement("span");
    decision.className = entry.risk
      ? entry.approved ? "activity-decision approved" : "activity-decision denied"
      : "activity-decision";
    decision.textContent = entry.risk
      ? entry.approved ? "Approved" : "Denied"
      : "No approval needed";
    const input = document.createElement("pre");
    input.textContent = JSON.stringify(entry.input, null, 2);
    const result = document.createElement("p");
    result.className = "activity-result";
    result.textContent = entry.result;
    const timestamp = document.createElement("time");
    timestamp.dateTime = entry.timestamp;
    timestamp.textContent = new Date(entry.timestamp).toLocaleString();
    heading.append(tool, decision);
    card.append(heading, input, result, timestamp);
    activityEntries.append(card);
  }
}

async function loadActivity() {
  try {
    const result = await api("/api/activity");
    renderActivity(result.entries);
  } catch (error) {
    activityNotice.textContent = error.message;
  }
}

function renderMemory() {
  memoryLines.replaceChildren();
  for (const file of memoryData) {
    const section = document.createElement("section");
    section.className = "memory-file";
    const heading = document.createElement("h3");
    heading.textContent = file.name;
    section.append(heading);
    if (file.lines.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "Nothing saved here yet.";
      section.append(empty);
      memoryLines.append(section);
      continue;
    }

    for (const line of file.lines) {
      const card = document.createElement("article");
      card.className = "memory-line";
      const number = document.createElement("span");
      number.className = "line-number";
      number.textContent = String(line.line_number).padStart(2, "0");
      const content = document.createElement("p");
      content.className = "memory-text";
      content.textContent = line.text;
      const actions = document.createElement("div");
      actions.className = "line-actions";
      const editButton = document.createElement("button");
      editButton.className = "text-button";
      editButton.type = "button";
      editButton.textContent = "Edit";
      const deleteButton = document.createElement("button");
      deleteButton.className = "text-button delete-button";
      deleteButton.type = "button";
      deleteButton.textContent = "Delete";
      actions.append(editButton, deleteButton);
      card.append(number, content, actions);

      editButton.addEventListener("click", () => {
        const editor = document.createElement("textarea");
        editor.className = "memory-editor";
        editor.maxLength = 1000;
        editor.value = line.text;
        const editorActions = document.createElement("div");
        editorActions.className = "line-actions";
        const saveButton = document.createElement("button");
        saveButton.type = "button";
        saveButton.textContent = "Save change";
        const cancelButton = document.createElement("button");
        cancelButton.className = "text-button";
        cancelButton.type = "button";
        cancelButton.textContent = "Cancel";
        editorActions.append(saveButton, cancelButton);
        content.replaceWith(editor);
        actions.replaceChildren(editorActions);
        saveButton.addEventListener("click", async () => {
          try {
            await api("/api/memory/update", {
              method: "PUT",
              body: JSON.stringify({
                file: file.name,
                line_number: line.line_number,
                text: editor.value,
              }),
            });
            memoryNotice.textContent = "Memory updated.";
            await loadMemory();
          } catch (error) {
            memoryNotice.textContent = error.message;
          }
        });
        cancelButton.addEventListener("click", renderMemory);
      });

      deleteButton.addEventListener("click", async () => {
        if (!window.confirm("Delete this memory? This cannot be undone.")) return;
        try {
          await api("/api/memory/delete", {
            method: "DELETE",
            body: JSON.stringify({
              file: file.name,
              line_number: line.line_number,
            }),
          });
          memoryNotice.textContent = "Memory deleted.";
          await loadMemory();
        } catch (error) {
          memoryNotice.textContent = error.message;
        }
      });

      section.append(card);
    }
    memoryLines.append(section);
  }
}

async function loadMemory() {
  try {
    const result = await api("/api/memory");
    memoryData = result.files;
    renderMemory();
  } catch (error) {
    memoryNotice.textContent = error.message;
  }
}

function renderProposals() {
  proposalList.replaceChildren();
  saveSelectedButton.disabled = !proposals.some((proposal) => proposal.selected);
  if (proposals.length === 0) {
    proposalNotice.textContent = "Nothing to save.";
    return;
  }

  for (const [index, proposal] of proposals.entries()) {
    const card = document.createElement("article");
    card.className = "proposal-card";
    const heading = document.createElement("div");
    heading.className = "proposal-heading";
    const fileLabel = document.createElement("span");
    fileLabel.className = "file-label";
    fileLabel.textContent = proposal.file.replace(".md", "");
    const reason = document.createElement("p");
    reason.className = "proposal-reason";
    reason.textContent = proposal.reason;
    const text = document.createElement("p");
    text.className = "proposal-text";
    text.textContent = proposal.text;
    const editField = document.createElement("textarea");
    editField.className = "proposal-editor";
    editField.maxLength = 1000;
    editField.value = proposal.text;
    editField.hidden = true;
    const actions = document.createElement("div");
    actions.className = "proposal-actions";
    const confirmButton = document.createElement("button");
    confirmButton.type = "button";
    confirmButton.className = proposal.selected ? "confirm-button selected" : "confirm-button";
    confirmButton.textContent = proposal.selected ? "Confirmed" : "Confirm";
    const editButton = document.createElement("button");
    editButton.type = "button";
    editButton.className = "text-button";
    editButton.textContent = "Edit";
    const rejectButton = document.createElement("button");
    rejectButton.type = "button";
    rejectButton.className = "text-button delete-button";
    rejectButton.textContent = "Reject";
    actions.append(confirmButton, editButton, rejectButton);
    heading.append(fileLabel, reason);
    card.append(heading, text, editField, actions);

    confirmButton.addEventListener("click", () => {
      proposal.selected = !proposal.selected;
      renderProposals();
    });
    editButton.addEventListener("click", () => {
      if (editField.hidden) {
        editField.hidden = false;
        text.hidden = true;
        editButton.textContent = "Keep edit";
        editField.focus();
        return;
      }
      proposal.text = editField.value.trim();
      editField.hidden = true;
      text.hidden = false;
      text.textContent = proposal.text;
      editButton.textContent = "Edit";
    });
    rejectButton.addEventListener("click", () => {
      proposals.splice(index, 1);
      renderProposals();
    });
    proposalList.append(card);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = promptField.value.trim();
  if (!text || sendButton.disabled) return;

  conversation.push({ role: "user", content: text });
  addMessage("user", text);
  promptField.value = "";
  sendButton.disabled = true;
  endSessionButton.disabled = true;
  notice.textContent = "Thinking…";
  const answerElement = addMessage("assistant", "");
  let answer = "";
  let waitingForApproval = false;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: conversation }),
    });
    const result = await readAgentStream(response, answerElement);
    answer = result.answer;
    if (result.approval) {
      waitingForApproval = true;
      approvalPending = true;
      addApprovalCard(result.approval, answerElement);
      notice.textContent = "A task change is paused until you approve or deny it.";
      return;
    }
    if (!result.completed) throw new Error("The response stream ended unexpectedly.");
    conversation.push({ role: "assistant", content: answer });
    await loadActivity();
    notice.textContent = "Sent to the configured model provider (Nebius by default). This build does not redact personal details.";
  } catch (error) {
    answerElement.textContent = error.message;
    answerElement.classList.add("error-message");
    conversation.pop();
    notice.textContent = "Could not reach the model. Check your connection and server settings.";
  } finally {
    if (!waitingForApproval) {
      sendButton.disabled = false;
      endSessionButton.disabled = false;
    }
    promptField.focus();
  }
});

endSessionButton.addEventListener("click", async () => {
  if (sendButton.disabled || endSessionButton.disabled) return;
  proposalSection.hidden = false;
  proposalNotice.textContent = "";
  proposals = [];
  renderProposals();
  if (conversation.length === 0) return;

  endSessionButton.disabled = true;
  sendButton.disabled = true;
  proposalNotice.textContent = "Reviewing this conversation…";
  try {
    const result = await api("/api/session/end", {
      method: "POST",
      body: JSON.stringify({ messages: conversation }),
    });
    proposals = result.proposals.map((proposal) => ({ ...proposal, selected: false }));
    proposalNotice.textContent = result.message;
    renderProposals();
    conversation.length = 0;
    messagesElement.replaceChildren();
    addMessage("assistant", "Session ended. Start a new conversation whenever you’re ready.");
  } catch (error) {
    proposalNotice.textContent = `Could not review this session: ${error.message}`;
  } finally {
    endSessionButton.disabled = false;
    sendButton.disabled = false;
  }
});

saveSelectedButton.addEventListener("click", async () => {
  const selected = proposals.filter((proposal) => proposal.selected);
  if (selected.length === 0) return;
  saveSelectedButton.disabled = true;
  try {
    const result = await api("/api/memory/confirm", {
      method: "POST",
      body: JSON.stringify({
        proposals: selected.map(({ file, text, reason }) => ({ file, text, reason })),
      }),
    });
    proposalNotice.textContent =
      `Saved ${result.saved} ${result.saved === 1 ? "memory" : "memories"}; skipped ${result.skipped_duplicates} duplicate(s).`;
    proposals = proposals.filter((proposal) => !proposal.selected);
    renderProposals();
    await loadMemory();
  } catch (error) {
    proposalNotice.textContent = `Could not save selected memories: ${error.message}`;
    saveSelectedButton.disabled = false;
  }
});

document.querySelector("#refresh-memory").addEventListener("click", loadMemory);
document.querySelector("#refresh-activity").addEventListener("click", loadActivity);
addMemoryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = newMemoryField.value.trim();
  if (!text) return;
  try {
    const result = await api("/api/memory/add", {
      method: "POST",
      body: JSON.stringify({ file: memoryAddFileSelect.value, text }),
    });
    memoryNotice.textContent = result.added ? "Memory added." : "That memory is already saved.";
    newMemoryField.value = "";
    await loadMemory();
  } catch (error) {
    memoryNotice.textContent = error.message;
  }
});

loadMemory();
loadActivity();
