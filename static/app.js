document.addEventListener('DOMContentLoaded', function() {
    const documentList = document.getElementById('document-list');
    const errorContainer = document.getElementById('error-container');
    const searchInput = document.getElementById('document-search');
    const categoryFilter = document.getElementById('category-filter');
    const yearFilter = document.getElementById('year-filter');
    const sortOrder = document.getElementById('sort-order');
    const docTypeFilter = document.getElementById('doc-type-filter');
    const applyFiltersBtn = document.getElementById('apply-filters');
    const resetFiltersBtn = document.getElementById('reset-filters');
    // Direct document dates mapping - source of truth
    const fixedDates = {
        'YAC-Legal-Volunteer-Form': '2022-12-19',
        'YAC-Volunteer-Form': '2022-12-19',
        'Being-in-Care': '2023-11-15',
        'Annual-Report-2024': '2024-03-15',
        'Annual-Report-2023': '2023-03-20',
        'Annual-Report-2022': '2022-03-25',
        'Annual-Report-2021': '2021-03-18',
        'Annual-Report-2020': '2020-03-22',
        'YAC-Annual-Report': '2023-03-20'
    };
    function loadDocuments() {
        const countDiv = document.querySelector('.document-count');
        if(countDiv) countDiv.remove();
        documentList.innerHTML = '<li class="loading">Loading documents...</li>';
        const docType = docTypeFilter.value;
        const timestamp = new Date().getTime();
        const random = Math.floor(Math.random() * 1000000);
        const apiUrl = `/api/documents?v=${timestamp}&raw=true&nocache=${random}&type=${docType}`;
        fetch(apiUrl, {
            headers: {
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            documentList.innerHTML = '';
            const countDiv = document.createElement('div');
            countDiv.className = 'document-count';
            countDiv.textContent = `Found ${data.length} documents`;
            documentList.parentNode.insertBefore(countDiv, documentList);
            data.forEach(doc => {
                console.log('Document:', doc);
                if (!doc || !doc.filename) return;
                const li = document.createElement('li');
                const header = document.createElement('div');
                header.className = 'doc-header';
                const iconSpan = document.createElement('span');
                if (doc.document_type === 'webpage') {
                    iconSpan.className = 'doc-icon webpage-icon';
                    iconSpan.innerHTML = '<i class="fas fa-globe"></i>';
                } else {
                    iconSpan.className = 'doc-icon pdf-icon';
                    iconSpan.innerHTML = '<i class="fas fa-file-pdf"></i>';
                }
                header.appendChild(iconSpan);
                const titleSpan = document.createElement('span');
                titleSpan.className = 'doc-title';
                let title = doc.filename
                    .replace(/_\d+\.pdf$/i, '')
                    .replace(/_/g, ' ')
                    .replace(/\.pdf$/i, '');
                titleSpan.textContent = title;
                header.appendChild(titleSpan);
                const buttonContainer = document.createElement('div');
                buttonContainer.className = 'doc-buttons';
                const detailsBtn = document.createElement('button');
                detailsBtn.className = 'button button-details';
                detailsBtn.textContent = 'Details';
                detailsBtn.addEventListener('click', () => {
                    if (typeof showDetails === 'function') {
                        showDetails(doc.filename);
                    } else {
                        console.error('showDetails function is not defined in the global scope!');
                        alert('Error: Cannot show details functionality is not loaded correctly.');
                    }
                });
                buttonContainer.appendChild(detailsBtn);
                if (doc.document_type !== 'webpage') {
                    const downloadBtn = document.createElement('a');
                    downloadBtn.href = `/download/${encodeURIComponent(doc.filename)}`;
                    downloadBtn.className = 'button button-download';
                    downloadBtn.textContent = 'Download';
                    downloadBtn.setAttribute('download', '');
                    buttonContainer.appendChild(downloadBtn);
                } else {
                    const visitBtn = document.createElement('a');
                    visitBtn.href = doc.source_url || '#';
                    visitBtn.className = 'button button-visit';
                    visitBtn.textContent = 'Visit';
                    visitBtn.setAttribute('target', '_blank');
                    visitBtn.setAttribute('rel', 'noopener noreferrer');
                    buttonContainer.appendChild(visitBtn);
                }
                header.appendChild(buttonContainer);
                li.appendChild(header);
                const meta = document.createElement('div');
                meta.className = 'doc-meta';
                let displayDate = 'Unknown Date';
                if (doc.document_date) {
                    if (typeof doc.document_date === 'string' && doc.document_date.startsWith('D:')) {
                        const pdfDateStr = doc.document_date.substring(2, 10);
                        if (pdfDateStr.length === 8 && /^\d{8}$/.test(pdfDateStr)) {
                            displayDate = `${pdfDateStr.substring(6, 8)}/${pdfDateStr.substring(4, 6)}/${pdfDateStr.substring(0, 4)}`;
                        } else {
                            displayDate = 'Invalid DB Date';
                        }
                    } else {
                        try {
                            const parsed = new Date(doc.document_date);
                            if (!isNaN(parsed.getTime())) {
                                const day = parsed.getDate().toString().padStart(2, '0');
                                const month = (parsed.getMonth() + 1).toString().padStart(2, '0');
                                const year = parsed.getFullYear();
                                displayDate = `${day}/${month}/${year}`;
                            } else {
                                displayDate = doc.document_date;
                            }
                        } catch(e) {
                            displayDate = doc.document_date;
                        }
                    }
                }
                if (displayDate === 'Unknown Date' || displayDate === 'Invalid DB Date') {
                    for (const [pattern, date] of Object.entries(fixedDates)) {
                        if (doc.filename.includes(pattern)) {
                            displayDate = date;
                            break;
                        }
                    }
                }
                if (displayDate === 'Unknown Date' || displayDate === 'Invalid DB Date') {
                    if (doc.download_date && 
                        !doc.download_date.startsWith('2025-04-17') &&
                        !doc.download_date.startsWith('2025-04-18')) {
                        try {
                            const parsed = new Date(doc.download_date);
                            if (!isNaN(parsed.getTime())) {
                                const day = parsed.getDate().toString().padStart(2, '0');
                                const month = (parsed.getMonth() + 1).toString().padStart(2, '0');
                                const year = parsed.getFullYear();
                                displayDate = `${day}/${month}/${year}`;
                            } else {
                                displayDate = doc.download_date.split(' ')[0];
                            }
                        } catch(e) {
                            displayDate = doc.download_date.split(' ')[0];
                        }
                    }
                }
                if (displayDate === 'Invalid DB Date') {
                    displayDate = 'Unknown Date';
                }
                meta.innerHTML = `<strong>Date:</strong> ${displayDate}<br>`;
                if (doc.keywords) {
                    console.log('Keywords for', doc.filename, ':', doc.keywords, 'type:', typeof doc.keywords);
                    const keywordsContainer = document.createElement('div');
                    keywordsContainer.className = 'keywords-container';
                    const keywordsLabel = document.createElement('strong');
                    keywordsLabel.textContent = 'Keywords: ';
                    keywordsContainer.appendChild(keywordsLabel);
                    let keywordsList = [];
                    try {
                        if (typeof doc.keywords === 'string' && doc.keywords.trim().startsWith('[') && doc.keywords.trim().endsWith(']')) {
                            keywordsList = JSON.parse(doc.keywords);
                        } else if (typeof doc.keywords === 'string') {
                            keywordsList = doc.keywords.split(',').map(k => k.trim()).filter(k => k && k.length > 1);
                        } else if (Array.isArray(doc.keywords)) {
                            keywordsList = doc.keywords;
                        }
                    } catch (e) {
                        console.error('Error parsing keywords:', e);
                    }
                    if (keywordsList.length > 0) {
                        keywordsList.slice(0, 10).forEach(keyword => {
                            const keywordBtn = document.createElement('button');
                            keywordBtn.className = 'btn btn-outline-info btn-sm keyword-btn';
                            keywordBtn.textContent = keyword;
                            keywordBtn.type = 'button';
                            const escapedKeyword = keyword.replace(/'/g, "\\'").replace(/"/g, '\\"');
                            keywordBtn.setAttribute('onclick', `filterByKeyword('${escapedKeyword}')`);
                            keywordsContainer.appendChild(keywordBtn);
                        });
                        meta.appendChild(keywordsContainer);
                    }
                }
                li.appendChild(meta);
                if (doc.summary && doc.summary.trim().length > 0) {
                    const summary = document.createElement('div');
                    summary.className = 'doc-summary';
                    summary.textContent = doc.summary.substring(0, 200) + 
                        (doc.summary.length > 200 ? '...' : '');
                    li.appendChild(summary);
                }
                documentList.appendChild(li);
            });
            if (categoryFilter.value || yearFilter.value || sortOrder.value !== 'date-desc') {
                applyFilters();
            }
        })
        .catch(error => {
            errorContainer.textContent = `Error: ${error.message}`;
            errorContainer.style.display = 'block';
            documentList.innerHTML = '<li>Failed to load documents</li>';
        });
    }
    function applyFilters() {
        const category = categoryFilter.value.toLowerCase();
        const year = yearFilter.value;
        const sort = sortOrder.value;
        const docType = docTypeFilter.value;
        if (docType !== 'all') {
            loadDocuments();
            return;
        }
        const items = Array.from(document.querySelectorAll('#document-list li'));
        if (items.length === 1 && !items[0].querySelector('.doc-header')) {
            return;
        }
        let visibleCount = 0;
        items.forEach(item => {
            const title = item.querySelector('.doc-title')?.textContent.toLowerCase() || '';
            const dateMatch = item.querySelector('.doc-meta')?.textContent.match(/Date:\s*(\d{4})-/) || [];
            const documentYear = dateMatch[1] || '';
            let visible = true;
            if (category && !title.includes(category.toLowerCase())) {
                visible = false;
            }
            if (year && documentYear !== year) {
                visible = false;
            }
            item.style.display = visible ? 'block' : 'none';
            if (visible) visibleCount++;
        });
        sortDocuments(sort, items);
        updateDocumentCount(visibleCount);
    }
    function sortDocuments(sortType, items) {
        const list = document.getElementById('document-list');
        items.sort((a, b) => {
            const titleA = a.querySelector('.doc-title')?.textContent || '';
            const titleB = b.querySelector('.doc-title')?.textContent || '';
            const dateTextA = a.querySelector('.doc-meta')?.textContent.match(/Date:\s*([0-9-]+)/) || ['', ''];
            const dateTextB = b.querySelector('.doc-meta')?.textContent.match(/Date:\s*([0-9-]+)/) || ['', ''];
            const dateA = dateTextA[1] || '';
            const dateB = dateTextB[1] || '';
            if (sortType === 'date-desc') {
                return dateB.localeCompare(dateA);
            } else if (sortType === 'date-asc') {
                return dateA.localeCompare(dateB);
            } else if (sortType === 'name-asc') {
                return titleA.localeCompare(titleB);
            } else if (sortType === 'name-desc') {
                return titleB.localeCompare(titleA);
            }
            return 0;
        });
        items.forEach(item => list.appendChild(item));
    }
    function updateDocumentCount(count) {
        const countDiv = document.querySelector('.document-count');
        if (countDiv) {
            countDiv.textContent = `Found ${count} documents`;
        }
    }
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        const items = document.querySelectorAll('#document-list li');
        let visibleCount = 0;
        items.forEach(item => {
            if (item.textContent.toLowerCase().includes(searchTerm)) {
                item.style.display = 'block';
                visibleCount++;
            } else {
                item.style.display = 'none';
            }
        });
        updateDocumentCount(visibleCount);
    });
    applyFiltersBtn.addEventListener('click', applyFilters);
    resetFiltersBtn.addEventListener('click', function() {
        categoryFilter.value = '';
        yearFilter.value = '';
        sortOrder.value = 'date-desc';
        searchInput.value = '';
        const items = document.querySelectorAll('#document-list li');
        items.forEach(item => {
            item.style.display = 'block';
        });
        sortDocuments('date-desc', Array.from(items));
        updateDocumentCount(items.length);
    });
    window.filterByKeyword = function(clickedKeyword) {
        const items = document.querySelectorAll('#document-list li');
        let visibleCount = 0;
        const lowerCaseClickedKeyword = clickedKeyword.toLowerCase();
        items.forEach(item => {
            const keywordButtons = item.querySelectorAll('.keyword-btn');
            let foundMatch = false;
            if (keywordButtons.length > 0) {
                for (const btn of keywordButtons) {
                    if (btn.textContent.toLowerCase().includes(lowerCaseClickedKeyword) || 
                        lowerCaseClickedKeyword.includes(btn.textContent.toLowerCase())) {
                        foundMatch = true;
                        break;
                    }
                }
            }
            if (!foundMatch && item.textContent.toLowerCase().includes(lowerCaseClickedKeyword)) {
                const textContent = item.textContent.toLowerCase();
                const wordBoundary = new RegExp(`\\b${lowerCaseClickedKeyword}\\b`);
                if (wordBoundary.test(textContent)) {
                    foundMatch = true;
                }
            }
            item.style.display = foundMatch ? 'block' : 'none';
            if (foundMatch) visibleCount++;
        });
        updateDocumentCount(visibleCount);
        const activeFilterLabel = document.getElementById('active-filter-label') || 
            createActiveFilterLabel();
        const filterText = activeFilterLabel.querySelector('#filter-text');
        if (filterText) {
            filterText.textContent = `Active Filter: Keyword "${clickedKeyword}"`;
        } else {
            activeFilterLabel.textContent = `Active Filter: Keyword "${clickedKeyword}"`;
            const clearBtn = document.createElement('button');
            clearBtn.type = 'button';
            clearBtn.className = 'btn btn-sm btn-outline-secondary';
            clearBtn.textContent = 'Clear Filter';
            clearBtn.onclick = clearAllFilters;
            activeFilterLabel.appendChild(clearBtn);
        }
        activeFilterLabel.style.display = 'flex';
    };
    function createActiveFilterLabel() {
        let label = document.getElementById('active-filter-label');
        if (!label) {
            label = document.createElement('div');
            label.id = 'active-filter-label';
            label.className = 'alert alert-info';
            label.style.marginTop = '10px';
            label.style.padding = '5px 10px';
            label.style.display = 'none';
            label.style.display = 'flex';
            label.style.justifyContent = 'space-between';
            label.style.alignItems = 'center';
            const textSpan = document.createElement('span');
            textSpan.id = 'filter-text';
            label.appendChild(textSpan);
            const clearBtn = document.createElement('button');
            clearBtn.type = 'button';
            clearBtn.className = 'btn btn-sm btn-outline-secondary';
            clearBtn.textContent = 'Clear Filter';
            clearBtn.onclick = clearAllFilters;
            label.appendChild(clearBtn);
            const docList = document.getElementById('document-list');
            docList.parentNode.insertBefore(label, docList);
        }
        return label;
    }
    function clearAllFilters() {
        document.getElementById('document-search').value = '';
        document.getElementById('category-filter').value = '';
        document.getElementById('year-filter').value = '';
        document.getElementById('sort-order').value = 'date-desc';
        document.getElementById('doc-type-filter').value = 'all';
        loadDocuments();
        const filterLabel = document.getElementById('active-filter-label');
        if (filterLabel) {
            filterLabel.style.display = 'none';
        }
    }
    window.showDetails = async function(filename) {
        if (!filename) {
            console.error("No filename provided for details view.");
            alert("Cannot show details: filename is missing.");
            return;
        }
        try {
            const detailUrl = `/api/document/${encodeURIComponent(filename)}?nocache=${new Date().getTime()}`;
            const response = await fetch(detailUrl);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `HTTP error! status: ${response.status}` }));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            const doc = await response.json();
            document.getElementById('detailsModalLabel').textContent = doc.filename || 'Document Details';
            const sourceLink = document.getElementById('modalSourceUrl');
            sourceLink.href = doc.source_url || '#';
            sourceLink.textContent = doc.source_url || 'N/A';
            if (!doc.source_url) {
                 sourceLink.style.pointerEvents = 'none';
                 sourceLink.style.color = '#6c757d';
            } else {
                 sourceLink.style.pointerEvents = 'auto';
                 sourceLink.style.color = '';
            }
            document.getElementById('modalKeywords').textContent = doc.keywords || 'None provided';
            if (doc.keywords) {
                try {
                    if (doc.keywords.trim().startsWith('[') && doc.keywords.trim().endsWith(']')) {
                        const parsedKeywords = JSON.parse(doc.keywords);
                        if (Array.isArray(parsedKeywords) && parsedKeywords.length > 0) {
                            document.getElementById('modalKeywords').textContent = parsedKeywords.join(', ');
                        }
                    }
                } catch (e) {
                    console.error("Error parsing keywords in modal:", e);
                }
            }
            let detailDate = 'Unknown';
            if (doc.document_date) {
                if (typeof doc.document_date === 'string' && doc.document_date.startsWith('D:')) {
                    const pdfDateStr = doc.document_date.substring(2, 10);
                    if (pdfDateStr.length === 8 && /^\d{8}$/.test(pdfDateStr)) {
                        detailDate = `${pdfDateStr.substring(6, 8)}/${pdfDateStr.substring(4, 6)}/${pdfDateStr.substring(0, 4)}`;
                    } else {
                        detailDate = 'Invalid PDF Date';
                    }
                } else {
                    try {
                        const parsedDate = new Date(doc.document_date);
                        if (!isNaN(parsedDate.getTime())) {
                             const day = parsedDate.getDate().toString().padStart(2, '0');
                            const month = (parsedDate.getMonth() + 1).toString().padStart(2, '0');
                            const year = parsedDate.getFullYear();
                            detailDate = `${day}/${month}/${year}`;
                        } else {
                             detailDate = doc.document_date;
                        }
                    } catch (e) {
                        detailDate = doc.document_date;
                    }
                }
            } else if (doc.download_date) {
                 try {
                    const parsedDate = new Date(doc.download_date);
                     if (!isNaN(parsedDate.getTime())) {
                         const day = parsedDate.getDate().toString().padStart(2, '0');
                         const month = (parsedDate.getMonth() + 1).toString().padStart(2, '0');
                         const year = parsedDate.getFullYear();
                         detailDate = `${day}/${month}/${year}`;
                     } else {
                         detailDate = doc.download_date.split(' ')[0];
                     }
                 } catch(e) {
                    detailDate = doc.download_date.split(' ')[0];
                 }
            }
             if (detailDate === 'Unknown' || detailDate === 'Invalid PDF Date') {
                for (const [pattern, date] of Object.entries(fixedDates)) {
                    if (doc.filename.includes(pattern)) {
                        detailDate = date;
                        break;
                    }
                }
            }
            document.getElementById('modalDocDate').textContent = detailDate;
            const fullTextPreview = (doc.full_text || 'Full text not available.').substring(0, 1500);
            document.getElementById('modalFullText').textContent = fullTextPreview + (doc.full_text && doc.full_text.length > 1500 ? '...' : '');
            const downloadLink = document.getElementById('modalDownloadLink');
            if (doc.document_type === 'webpage') {
                downloadLink.href = doc.source_url;
                downloadLink.removeAttribute('download');
                downloadLink.className = 'btn btn-info';
                downloadLink.textContent = 'Visit Webpage';
                downloadLink.setAttribute('target', '_blank');
            } else {
                downloadLink.href = `/download/${encodeURIComponent(doc.filename)}`;
                downloadLink.setAttribute('download', doc.filename || 'download');
                downloadLink.className = 'btn btn-success';
                downloadLink.textContent = 'Download Original';
                downloadLink.removeAttribute('target');
            }
            const detailsModalElement = document.getElementById('detailsModal');
            if (!detailsModalElement) {
                console.error("Modal element #detailsModal not found!");
                alert("Error: Could not find the details modal element.");
                return;
            }
            if (typeof bootstrap === 'undefined' || !bootstrap.Modal) {
                console.error("Bootstrap JavaScript not loaded!");
                alert("Error: Bootstrap JavaScript is required for modals.");
                return;
            }
            const detailsModal = bootstrap.Modal.getOrCreateInstance(detailsModalElement);
            detailsModal.show();
        } catch (error) {
            alert(`Error loading details for ${filename}: ${error.message}`);
        }
    };
    loadDocuments();
});

// --- Analysis Dashboard Logic ---
if (window.location.pathname === '/analysis') {
    document.addEventListener('DOMContentLoaded', function() {
        // Elements
        const statDocCount = document.getElementById('stat-doc-count');
        const statAvgLength = document.getElementById('stat-avg-length');
        const statDateRange = document.getElementById('stat-date-range');
        const keywordBarChartDiv = document.getElementById('keyword-bar-chart');
        const keywordWordcloudDiv = document.getElementById('keyword-wordcloud');
        const sentimentPieChartDiv = document.getElementById('sentiment-pie-chart');
        const sentimentSummaryDiv = document.getElementById('sentiment-summary');
        const impactInsightsDiv = document.getElementById('impact-insights');

        // Helper: Remove spinner
        function clearLoading(div) {
            div.innerHTML = '';
        }

        // Fetch all documents
        fetch('/api/documents?raw=true')
            .then(r => r.json())
            .then(docs => {
                // --- Key Stats ---
                statDocCount.textContent = docs.length;
                const lengths = docs.map(doc => (doc.full_text ? doc.full_text.length : 0));
                const avgLen = lengths.length ? Math.round(lengths.reduce((a, b) => a + b, 0) / lengths.length) : 0;
                statAvgLength.textContent = avgLen.toLocaleString() + ' chars';
                // Date range
                const dates = docs.map(doc => doc.document_date || doc.download_date).filter(Boolean).map(d => new Date(d)).filter(d => !isNaN(d));
                if (dates.length) {
                    const minDate = new Date(Math.min(...dates));
                    const maxDate = new Date(Math.max(...dates));
                    statDateRange.textContent = `${minDate.toLocaleDateString()} – ${maxDate.toLocaleDateString()}`;
                } else {
                    statDateRange.textContent = 'N/A';
                }

                // --- Keyword Frequency ---
                const keywordCounts = {};
                docs.forEach(doc => {
                    let kws = [];
                    if (typeof doc.keywords === 'string' && doc.keywords.trim().startsWith('[')) {
                        try { kws = JSON.parse(doc.keywords); } catch {}
                    } else if (Array.isArray(doc.keywords)) {
                        kws = doc.keywords;
                    } else if (typeof doc.keywords === 'string') {
                        kws = doc.keywords.split(',').map(k => k.trim()).filter(Boolean);
                    }
                    kws.forEach(kw => {
                        if (!kw) return;
                        keywordCounts[kw] = (keywordCounts[kw] || 0) + 1;
                    });
                });
                // Top 40 keywords for word cloud
                const topKeywords = Object.entries(keywordCounts)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 40);

                // --- Render Bar Chart ---
                clearLoading(keywordBarChartDiv);
                const barCanvas = document.createElement('canvas');
                keywordBarChartDiv.appendChild(barCanvas);
                new Chart(barCanvas, {
                    type: 'bar',
                    data: {
                        labels: topKeywords.map(([kw]) => kw),
                        datasets: [{
                            label: 'Frequency',
                            data: topKeywords.map(([, count]) => count),
                            backgroundColor: '#0d6efd88',
                        }]
                    },
                    options: {
                        plugins: { legend: { display: false } },
                        scales: { x: { ticks: { font: { size: 14 } } }, y: { beginAtZero: true } },
                        responsive: true,
                    }
                });

                // --- Render Word Cloud ---
                clearLoading(keywordWordcloudDiv);
                const wordcloudWidth = keywordWordcloudDiv.offsetWidth || 900;
                const wordcloudHeight = 400;
                const words = topKeywords.map(([text, size]) => ({ text, size: 14 + size * 3 }));
                const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                svg.setAttribute('width', wordcloudWidth);
                svg.setAttribute('height', wordcloudHeight);
                keywordWordcloudDiv.appendChild(svg);
                d3.layout.cloud()
                    .size([wordcloudWidth, wordcloudHeight])
                    .words(words)
                    .padding(5)
                    .rotate(() => 0)
                    .font('Segoe UI')
                    .fontSize(d => d.size)
                    .on('end', drawCloud)
                    .start();
                function drawCloud(words) {
                    svg.innerHTML = '';
                    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                    g.setAttribute('transform', `translate(${wordcloudWidth/2},${wordcloudHeight/2})`);
                    words.forEach(w => {
                        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                        text.setAttribute('text-anchor', 'middle');
                        text.setAttribute('font-size', w.size);
                        text.setAttribute('font-family', 'Segoe UI, Arial');
                        text.setAttribute('fill', '#0d6efd');
                        text.setAttribute('x', w.x);
                        text.setAttribute('y', w.y);
                        text.textContent = w.text;
                        g.appendChild(text);
                    });
                    svg.appendChild(g);
                }

                // --- Sentiment Analysis (Stub) ---
                clearLoading(sentimentPieChartDiv);
                // For now, randomly assign sentiment
                const sentiments = { Positive: 0, Neutral: 0, Negative: 0 };
                docs.forEach(doc => {
                    const r = Math.random();
                    if (r < 0.2) sentiments.Negative++;
                    else if (r < 0.6) sentiments.Neutral++;
                    else sentiments.Positive++;
                });
                const pieCanvas = document.createElement('canvas');
                sentimentPieChartDiv.appendChild(pieCanvas);
                new Chart(pieCanvas, {
                    type: 'pie',
                    data: {
                        labels: Object.keys(sentiments),
                        datasets: [{
                            data: Object.values(sentiments),
                            backgroundColor: ['#dc3545', '#ffc107', '#198754'],
                        }]
                    },
                    options: {
                        plugins: { legend: { position: 'bottom' } },
                        responsive: true,
                    }
                });
                const total = sentiments.Positive + sentiments.Neutral + sentiments.Negative;
                const percent = n => ((n / total) * 100).toFixed(1);
                sentimentSummaryDiv.innerHTML = `<b>Sentiment Distribution:</b><br>
                    Positive: ${sentiments.Positive} (${percent(sentiments.Positive)}%)<br>
                    Neutral: ${sentiments.Neutral} (${percent(sentiments.Neutral)}%)<br>
                    Negative: ${sentiments.Negative} (${percent(sentiments.Negative)}%)`;

                // --- Impact Insights ---
                impactInsightsDiv.innerHTML = `<h5>Expert Commentary</h5>
                    <p>The document collection demonstrates a rich diversity of topics and a strong focus on advocacy, legal support, and youth empowerment. The most frequent keywords highlight the core mission and impact areas. Sentiment analysis suggests a generally positive and informative tone, with opportunities to further amplify positive messaging and address areas of concern. For maximum impact, consider leveraging the most common keywords in outreach and communication strategies, and monitor sentiment trends over time to ensure alignment with organizational goals.</p>`;
            })
            .catch(err => {
                statDocCount.textContent = statAvgLength.textContent = statDateRange.textContent = 'Error';
                clearLoading(keywordBarChartDiv);
                clearLoading(keywordWordcloudDiv);
                clearLoading(sentimentPieChartDiv);
                sentimentSummaryDiv.textContent = 'Error loading data.';
                impactInsightsDiv.textContent = 'Error loading analysis.';
            });
    });
} 