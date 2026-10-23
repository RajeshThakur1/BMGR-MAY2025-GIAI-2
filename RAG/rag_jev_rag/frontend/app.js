let conversationId = null;

const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");

function appendMsg(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  appendMsg("user", message);
  chatInput.value = "";

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: conversationId, message }),
  });
  const data = await res.json();
  conversationId = data.conversation_id;

  if (data.type === "answer") {
    appendMsg("assistant", data.answer);
  } else {
    appendMsg("system", data.message || "Escalated to support.");
    loadTickets();
  }
});

document.getElementById("upload-btn").addEventListener("click", async () => {
  const input = document.getElementById("pdf-file");
  const status = document.getElementById("upload-status");
  if (!input.files?.length) {
    status.textContent = "Choose a PDF first.";
    return;
  }
  status.textContent = "Indexing…";
  const fd = new FormData();
  fd.append("file", input.files[0]);
  const res = await fetch("/api/documents/upload", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) {
    status.textContent = JSON.stringify(data);
    return;
  }
  status.textContent = `Indexed ${data.filename}: ${data.pages} pages, ${data.chunks} vectors.`;
});

async function loadTickets() {
  const res = await fetch("/api/tickets");
  const data = await res.json();
  const root = document.getElementById("tickets");
  root.innerHTML = "";
  for (const t of data.tickets) {
    const el = document.createElement("div");
    el.className = "ticket";
    el.innerHTML = `
      <strong>Ticket ${t.id.slice(0, 8)}</strong>
      <p>${t.question}</p>
      <pre>${t.rejected_json || "[]"}</pre>
      <textarea rows="3" placeholder="Rep reply…"></textarea>
      <br/>
      <label><input type="checkbox" checked /> Save to knowledge base</label>
      <br/>
      <button type="button">Send reply & resolve</button>
    `;
    const btn = el.querySelector("button");
    btn.addEventListener("click", async () => {
      const reply = el.querySelector("textarea").value.trim();
      const save = el.querySelector("input").checked;
      await fetch(`/api/tickets/${t.id}/reply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reply, save_to_knowledge: save }),
      });
      loadTickets();
    });
    root.appendChild(el);
  }
}

document.getElementById("refresh-tickets").addEventListener("click", loadTickets);
loadTickets();
