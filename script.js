const chat = document.getElementById('chat');
const emptyState = document.getElementById('emptyState');
const chatForm = document.getElementById('chatForm');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const moduleFilter = document.getElementById('moduleFilter');
const clearBtn = document.getElementById('clearBtn');

const apiBase = window.location.port === '8000' ? '' : 'http://127.0.0.1:8000';

function askSuggestion(text) {
  queryInput.value = text;
  chatForm.dispatchEvent(new Event('submit'));
}

clearBtn.addEventListener('click', () => {
  chat.innerHTML = '';
  chat.appendChild(emptyState);
  emptyState.style.display = 'block';
});

function appendMessage(role, htmlContent, sources = [], diagrams = []) {
  if (emptyState) emptyState.style.display = 'none';
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = htmlContent;

  if (diagrams.length > 0) {
    const grid = document.createElement('div');
    grid.className = 'diagrams-grid';
    diagrams.forEach(d => {
      const card = document.createElement('div');
      card.className = 'diagram-card';
      const imgUrl = `${apiBase}${d.url}`;
      card.innerHTML = `
        <a href="${imgUrl}" target="_blank" title="Click to open full slide diagram">
          <img src="${imgUrl}" alt="${d.source} page ${d.page}" loading="lazy" />
        </a>
        <div class="diagram-caption">📊 ${d.source} · Slide ${d.page}</div>
      `;
      grid.appendChild(card);
    });
    div.appendChild(grid);
  }

  if (sources.length > 0) {
    const srcContainer = document.createElement('div');
    srcContainer.className = 'sources';
    sources.forEach(s => {
      const tag = document.createElement('span');
      tag.className = 'source-tag';
      tag.textContent = `${s.source} · p.${s.page}`;
      srcContainer.appendChild(tag);
    });
    div.appendChild(srcContainer);
  }

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  queryInput.value = '';
  sendBtn.disabled = true;

  appendMessage('user', query.replace(/</g, '&lt;').replace(/>/g, '&gt;'));
  const loadingDiv = appendMessage('assistant', '<span style="color:#6b7280">Thinking & reading slides...</span>');

  try {
    const res = await fetch(`${apiBase}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        source: moduleFilter.value || null
      })
    });
    const data = await res.json();
    chat.removeChild(loadingDiv);

    if (!res.ok) {
      appendMessage('assistant', `<p>Error: ${data.detail || 'Request failed'}</p>`);
    } else {
      appendMessage(
        'assistant',
        marked.parse(data.answer),
        data.sources || [],
        data.diagrams || []
      );
    }
  } catch (err) {
    chat.removeChild(loadingDiv);
    appendMessage('assistant', `<p>Failed to connect to server.</p>`);
  } finally {
    sendBtn.disabled = false;
    queryInput.focus();
  }
});
