/**
 * CodeQuery AI — Frontend Logic
 * Handles API calls, chat rendering, repo management, and UI interactions.
 */

// ── State ────────────────────────────────────────────────────
let activeRepo = null;
let chatHistory = [];
let isLoading = false;

const API_BASE = '';

// ── DOM Elements ─────────────────────────────────────────────
const chatContainer = document.getElementById('chatContainer');
const welcomeMessage = document.getElementById('welcomeMessage');
const chatForm = document.getElementById('chatForm');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const indexForm = document.getElementById('indexForm');
const sourceInput = document.getElementById('sourceInput');
const repoNameInput = document.getElementById('repoNameInput');
const indexBtn = document.getElementById('indexBtn');
const progressContainer = document.getElementById('progressContainer');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const reposList = document.getElementById('reposList');
const activeRepoEl = document.getElementById('activeRepo');
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const sidebar = document.getElementById('sidebar');

// ── Initialize ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadRepos();
    setupEventListeners();
    autoResizeTextarea();
});

function setupEventListeners() {
    // Chat form
    chatForm.addEventListener('submit', handleAskQuestion);

    // Index form
    indexForm.addEventListener('submit', handleIndexRepo);

    // Textarea auto-resize and enter handling
    questionInput.addEventListener('input', autoResizeTextarea);
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Mobile menu
    mobileMenuBtn.addEventListener('click', toggleSidebar);

    // Close sidebar on outside click (mobile)
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && sidebar.classList.contains('open')) {
            if (!sidebar.contains(e.target) && e.target !== mobileMenuBtn) {
                sidebar.classList.remove('open');
                removeOverlay();
            }
        }
    });
}

// ── Textarea Auto-Resize ─────────────────────────────────────
function autoResizeTextarea() {
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 150) + 'px';
}

// ── Mobile Sidebar ───────────────────────────────────────────
function toggleSidebar() {
    sidebar.classList.toggle('open');
    if (sidebar.classList.contains('open')) {
        addOverlay();
    } else {
        removeOverlay();
    }
}

function addOverlay() {
    let overlay = document.querySelector('.sidebar-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay active';
        overlay.onclick = () => {
            sidebar.classList.remove('open');
            removeOverlay();
        };
        document.body.appendChild(overlay);
    }
    overlay.classList.add('active');
}

function removeOverlay() {
    const overlay = document.querySelector('.sidebar-overlay');
    if (overlay) overlay.remove();
}

// ── Load Repositories ────────────────────────────────────────
async function loadRepos() {
    try {
        const res = await fetch(`${API_BASE}/api/repos`);
        const repos = await res.json();
        renderRepos(repos);
    } catch (err) {
        console.error('Failed to load repos:', err);
    }
}

function renderRepos(repos) {
    if (!repos || repos.length === 0) {
        reposList.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">📭</span>
                <p>No repositories indexed yet</p>
            </div>
        `;
        return;
    }

    reposList.innerHTML = repos.map(repo => `
        <div class="repo-item ${activeRepo === repo.name ? 'active' : ''}" 
             onclick="selectRepo('${repo.name}')" data-repo="${repo.name}">
            <div class="repo-item-info">
                <span class="repo-icon">📁</span>
                <span class="repo-name">${escapeHtml(repo.name)}</span>
            </div>
            <span class="repo-chunks">${repo.num_chunks} chunks</span>
            <button class="repo-delete" onclick="event.stopPropagation(); deleteRepo('${repo.name}')" title="Delete">🗑️</button>
        </div>
    `).join('');
}

// ── Select Repository ────────────────────────────────────────
function selectRepo(name) {
    activeRepo = name;

    // Update sidebar
    document.querySelectorAll('.repo-item').forEach(el => {
        el.classList.toggle('active', el.dataset.repo === name);
    });

    // Update header badge
    activeRepoEl.innerHTML = `<span class="repo-badge active">📁 ${escapeHtml(name)}</span>`;

    // Close mobile sidebar
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('open');
        removeOverlay();
    }

    showToast(`Selected: ${name}`, 'info');
}

// ── Delete Repository ────────────────────────────────────────
async function deleteRepo(name) {
    if (!confirm(`Delete "${name}" from the index?`)) return;

    try {
        await fetch(`${API_BASE}/api/repos/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (activeRepo === name) {
            activeRepo = null;
            activeRepoEl.innerHTML = '<span class="repo-badge">No repo selected</span>';
        }
        showToast(`Deleted: ${name}`, 'success');
        loadRepos();
    } catch (err) {
        showToast('Failed to delete repository', 'error');
    }
}

// ── Index Repository ─────────────────────────────────────────
async function handleIndexRepo(e) {
    e.preventDefault();

    const source = sourceInput.value.trim();
    if (!source) return;

    const repoName = repoNameInput.value.trim() || null;

    // Show progress
    indexBtn.disabled = true;
    indexBtn.innerHTML = '<span class="btn-icon">⏳</span><span class="btn-text">Indexing...</span>';
    progressContainer.style.display = 'block';
    progressFill.style.width = '10%';
    progressText.textContent = 'Starting indexing...';

    try {
        const body = { source };
        if (repoName) body.repo_name = repoName;

        const res = await fetch(`${API_BASE}/api/index`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const result = await res.json();

        if (res.ok && result.status === 'success') {
            progressFill.style.width = '100%';
            progressText.textContent = `✅ ${result.message}`;
            showToast(`Indexed: ${result.repo_name}`, 'success');

            // Auto-select the newly indexed repo
            selectRepo(result.repo_name);
            loadRepos();

            // Clear form
            sourceInput.value = '';
            repoNameInput.value = '';
        } else {
            const msg = extractErrorMessage(result);
            progressText.textContent = `❌ ${msg}`;
            showToast(msg, 'error');
        }
    } catch (err) {
        progressText.textContent = `❌ Error: ${err.message}`;
        showToast('Indexing failed. Is the server running?', 'error');
    } finally {
        indexBtn.disabled = false;
        indexBtn.innerHTML = '<span class="btn-icon">📥</span><span class="btn-text">Index Repository</span>';

        // Hide progress after delay
        setTimeout(() => {
            progressContainer.style.display = 'none';
            progressFill.style.width = '0%';
        }, 5000);
    }
}

// ── Ask Question ─────────────────────────────────────────────
async function handleAskQuestion(e) {
    e.preventDefault();

    const question = questionInput.value.trim();
    if (!question || isLoading) return;

    if (!activeRepo) {
        showToast('Please select a repository first', 'error');
        return;
    }

    // Hide welcome message
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }

    // Add user message
    addMessage('user', question);
    questionInput.value = '';
    autoResizeTextarea();

    // Show thinking indicator
    isLoading = true;
    sendBtn.disabled = true;
    const thinkingEl = addThinking();

    try {
        const res = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                repo_name: activeRepo,
                top_k: 5,
                mode: getMode(),
            }),
        });

        const result = await res.json();

        // Remove thinking indicator
        thinkingEl.remove();

        if (res.ok) {
            addMessage('assistant', result.answer, result.sources, false, result.trace);
        } else {
            const msg = result.detail || 'Failed to get an answer';
            addMessage('assistant', `⚠️ Error: ${msg}`, [], true);
        }
    } catch (err) {
        thinkingEl.remove();
        addMessage('assistant', `⚠️ Could not reach the server. Make sure it's running on port 8000.`, [], true);
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        questionInput.focus();
    }
}

// ── Answer Mode ──────────────────────────────────────────────
function getMode() {
    const checked = document.querySelector('input[name="mode"]:checked');
    return checked ? checked.value : 'rag';
}

// The agent's steps: which tool it called, with what, and how long it took
function renderTrace(trace) {
    if (!trace || trace.length === 0) return '';
    const toolSteps = trace.filter(s => s.kind === 'tool_call' || s.kind === 'tool_error').length;
    const steps = trace.map(s => {
        const model = s.model_ms ? ` · model ${(s.model_ms / 1000).toFixed(1)}s` : '';
        if (s.kind === 'answer') {
            return `<div class="trace-step">round ${s.iteration}: ✅ answer <span class="trace-meta">${model}</span></div>`;
        }
        if (s.kind === 'model_error') {
            return `<div class="trace-step trace-error">round ${s.iteration}: ⚠️ ${escapeHtml(s.error || 'model error')}</div>`;
        }
        const result = s.error ? `⚠️ ${escapeHtml(s.error)}` : `${s.result_chars.toLocaleString()} chars`;
        return `
            <div class="trace-step ${s.error ? 'trace-error' : ''}">
                round ${s.iteration}: 🔧 ${escapeHtml(s.tool)}(${escapeHtml(JSON.stringify(s.arguments))})
                <span class="trace-meta">→ ${result} · tool ${Math.round(s.latency_ms)}ms${model}</span>
            </div>`;
    }).join('');
    return `
        <div class="sources-container">
            <button class="sources-toggle" onclick="this.nextElementSibling.classList.toggle('expanded')">
                🧠 Agent trace: ${toolSteps} tool call${toolSteps === 1 ? '' : 's'} ▾
            </button>
            <div class="trace-list">${steps}</div>
        </div>
    `;
}

// ── Render Messages ──────────────────────────────────────────
function addMessage(role, content, sources = [], isError = false, trace = []) {
    const msgEl = document.createElement('div');
    msgEl.className = `message ${role}`;

    const avatar = role === 'assistant' ? '⚡' : '👤';

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="sources-container">
                <button class="sources-toggle" onclick="this.nextElementSibling.classList.toggle('expanded')">
                    📎 ${sources.length} source${sources.length > 1 ? 's' : ''} referenced ▾
                </button>
                <div class="sources-list">
                    ${sources.map(s => `
                        <div class="source-item">
                            <span class="source-file">${escapeHtml(s.file_path)}</span>
                            <span class="source-lines">L${s.start_line}–${s.end_line}</span>
                            ${s.relevance ? `<span class="source-relevance">${(s.relevance * 100).toFixed(0)}%</span>` : ''}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    msgEl.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-bubble ${isError ? 'error-bubble' : ''}">
                ${role === 'assistant' ? renderMarkdown(content) : `<p>${escapeHtml(content)}</p>`}
            </div>
            ${sourcesHtml}
            ${renderTrace(trace)}
        </div>
    `;

    chatContainer.appendChild(msgEl);
    scrollToBottom();
}

function addThinking() {
    const el = document.createElement('div');
    el.className = 'message assistant';
    el.innerHTML = `
        <div class="message-avatar">⚡</div>
        <div class="message-content">
            <div class="thinking-indicator">
                <div class="thinking-dots">
                    <span></span><span></span><span></span>
                </div>
                Analyzing codebase...
            </div>
        </div>
    `;
    chatContainer.appendChild(el);
    scrollToBottom();
    return el;
}

// ── Simple Markdown Renderer ─────────────────────────────────
function renderMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Code blocks (```language\n...\n```)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Unordered lists
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

    // Ordered lists
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    // Line breaks → paragraphs
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');

    // Wrap in paragraph if not already wrapped in block elements
    if (!html.startsWith('<h') && !html.startsWith('<ul') && !html.startsWith('<pre')) {
        html = `<p>${html}</p>`;
    }

    return html;
}

// ── Helper Functions ─────────────────────────────────────────
function extractErrorMessage(result) {
    // FastAPI 422 returns detail as an array of objects
    if (Array.isArray(result.detail)) {
        return result.detail.map(d => d.msg || JSON.stringify(d)).join('; ');
    }
    if (typeof result.detail === 'string') return result.detail;
    if (typeof result.message === 'string') return result.message;
    return 'An unexpected error occurred';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function fillQuestion(el) {
    questionInput.value = el.textContent;
    questionInput.focus();
    autoResizeTextarea();
}

// ── Toast Notifications ──────────────────────────────────────
function showToast(message, type = 'info') {
    // Remove existing toasts
    document.querySelectorAll('.toast').forEach(t => t.remove());

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
