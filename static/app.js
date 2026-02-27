// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const form = document.getElementById('question-form');
    const questionInput = document.getElementById('question-input');
    const topKInput = document.getElementById('top-k');
    const submitBtn = document.getElementById('submit-btn');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const results = document.getElementById('results');
    const answerWithRag = document.getElementById('answer-with-rag');
    const answerWithoutRag = document.getElementById('answer-without-rag');
    const contextsContainer = document.getElementById('contexts-container');
    const kbContext = document.getElementById('kb-context');
    const llmContext = document.getElementById('llm-context');
    const themeToggle = document.getElementById('theme-toggle');
    const docCount = document.getElementById('doc-count');

    // Dark mode functionality
    function initTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeIcon(savedTheme);
    }

    function toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme);
    }

    function updateThemeIcon(theme) {
        if (themeToggle) {
            themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
            themeToggle.setAttribute('title', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
        }
    }

    // Add theme toggle click handler
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }

    // Load system stats on page load
    async function loadSystemStats() {
        try {
            const response = await fetch('/health');
            const data = await response.json();

            if (data.knowledge_base_documents > 0) {
                docCount.textContent = `${data.knowledge_base_documents.toLocaleString()} documents ready`;
            } else {
                docCount.textContent = 'No documents loaded';
                docCount.style.color = '#f59e0b'; // Warning color
            }
        } catch (err) {
            console.error('Failed to load system stats:', err);
            if (docCount) {
                docCount.textContent = 'Status unavailable';
            }
        }
    }

    // Form submission handler
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const question = questionInput.value.trim();
            const topK = parseInt(topKInput.value);

            if (!question) return;

            // UI state: loading
            setLoadingState(true);
            hideError();
            hideResults();

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        question: question,
                        top_k: topK
                    })
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Failed to get response');
                }

                const data = await response.json();
                displayResults(data);

            } catch (err) {
                showError(err.message);
            } finally {
                setLoadingState(false);
            }
        });
    }

    function setLoadingState(isLoading) {
        if (submitBtn) {
            submitBtn.disabled = isLoading;
            submitBtn.textContent = isLoading ? 'Processing...' : 'Ask Question';
        }
        if (loading) {
            loading.classList.toggle('hidden', !isLoading);
        }
    }

    function showError(message) {
        if (error) {
            error.textContent = `Error: ${message}`;
            error.classList.remove('hidden');
        }
    }

    function hideError() {
        if (error) {
            error.classList.add('hidden');
        }
    }

    function hideResults() {
        if (results) {
            results.classList.add('hidden');
        }
    }

    function displayResults(data) {
        // Display both answers for comparison
        if (answerWithRag) {
            answerWithRag.textContent = data.answer_with_rag;
        }
        if (answerWithoutRag) {
            answerWithoutRag.textContent = data.answer_without_rag;
        }

        // Display contexts
        if (contextsContainer) {
            contextsContainer.innerHTML = '';

            if (data.contexts && data.contexts.length > 0) {
                data.contexts.forEach((ctx, index) => {
                    const contextCard = createContextCard(ctx, index + 1);
                    contextsContainer.appendChild(contextCard);
                });
            } else {
                contextsContainer.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">No contexts retrieved from Canvas materials</p>';
            }
        }

        // Display additional info
        if (kbContext) {
            kbContext.textContent = data.kb_context;
        }
        if (llmContext) {
            llmContext.textContent = data.llm_context;
        }

        // Show results
        if (results) {
            results.classList.remove('hidden');
            // Smooth scroll to results
            results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    function createContextCard(ctx, index) {
        const card = document.createElement('div');
        card.className = 'context-card';

        const similarityPercent = (ctx.similarity * 100).toFixed(1);

        // Get content type icon
        const contentType = ctx.title.toLowerCase();
        let icon = '📄';
        if (contentType.includes('pdf')) icon = '📕';
        else if (contentType.includes('ppt') || contentType.includes('slide')) icon = '📊';
        else if (contentType.includes('assignment')) icon = '📝';
        else if (contentType.includes('discussion')) icon = '💬';
        else if (contentType.includes('announcement')) icon = '📢';
        else if (contentType.includes('module')) icon = '📦';

        // Preview text (first 150 characters)
        const previewText = ctx.text.length > 150
            ? ctx.text.substring(0, 150) + '...'
            : ctx.text;

        card.innerHTML = `
            <div class="context-header-clickable">
                <div class="context-header-left">
                    <span class="context-title">${icon} ${index}. ${escapeHtml(ctx.title)}</span>
                    <span class="similarity-badge">${similarityPercent}% match</span>
                </div>
                <button class="toggle-btn" aria-label="Toggle content">
                    <span class="toggle-icon">▼</span>
                </button>
            </div>
            <div class="context-source">Source: ${escapeHtml(ctx.source)}</div>
            <div class="context-preview">${escapeHtml(previewText)}</div>
            <div class="context-text-full" style="display: none;">
                ${escapeHtml(ctx.text)}
            </div>
        `;

        // Add click handler for toggle
        const toggleBtn = card.querySelector('.toggle-btn');
        const preview = card.querySelector('.context-preview');
        const fullText = card.querySelector('.context-text-full');
        const toggleIcon = card.querySelector('.toggle-icon');

        toggleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isExpanded = fullText.style.display !== 'none';

            if (isExpanded) {
                // Collapse
                fullText.style.display = 'none';
                preview.style.display = 'block';
                toggleIcon.textContent = '▼';
                toggleBtn.setAttribute('aria-label', 'Expand content');
            } else {
                // Expand
                fullText.style.display = 'block';
                preview.style.display = 'none';
                toggleIcon.textContent = '▲';
                toggleBtn.setAttribute('aria-label', 'Collapse content');
            }
        });

        return card;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Initialize theme and load stats
    initTheme();
    loadSystemStats();
});
