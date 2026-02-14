/**
 * Neuron Genealogy — vanilla JS logic
 * Extracted from tools/neuron-genealogy.html with scoped DOM queries.
 *
 * @param {HTMLElement} root - Container element (.neuron-genealogy)
 * @param {Array} NEURONS - Neuron data array
 * @returns {Function} cleanup - Call to remove listeners and timers
 */
export function initNeuronGenealogy(root, NEURONS) {
  // ============================================================
  // APPLICATION STATE & INDEXES
  // ============================================================

  let selectedNeuron = null;
  let activeLevels = new Set([0, 1, 2, 3, 4, 5]);
  let searchQuery = '';
  let statusFilter = 'all';
  let categoryFilter = 'all';

  // Build indexes
  const neuronMap = new Map();       // name -> neuron
  const requiredByMap = new Map();   // name -> [names that depend on it]
  const categories = new Set();

  function buildIndexes() {
    NEURONS.forEach(n => {
      neuronMap.set(n.name, n);
      categories.add(n.category);
      if (!requiredByMap.has(n.name)) requiredByMap.set(n.name, []);
    });
    NEURONS.forEach(n => {
      (n.dependencies || []).forEach(dep => {
        if (!requiredByMap.has(dep)) requiredByMap.set(dep, []);
        requiredByMap.get(dep).push(n.name);
      });
    });
  }

  buildIndexes();

  // Populate category dropdown
  function populateCategoryDropdown() {
    const sel = root.querySelector('#ng-categoryFilter');
    const sorted = [...categories].sort();
    sorted.forEach(cat => {
      const opt = document.createElement('option');
      opt.value = cat;
      opt.textContent = cat.replace(/_/g, ' ');
      sel.appendChild(opt);
    });
  }
  populateCategoryDropdown();


  // ============================================================
  // FILTERING
  // ============================================================

  function getFilteredNeurons() {
    return NEURONS.filter(n => {
      if (!activeLevels.has(n.level)) return false;
      if (statusFilter !== 'all' && n.status !== statusFilter) return false;
      if (categoryFilter !== 'all' && n.category !== categoryFilter) return false;
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        return n.name.toLowerCase().includes(q) || n.description.toLowerCase().includes(q);
      }
      return true;
    });
  }


  // ============================================================
  // RENDER NEURON LIST
  // ============================================================

  const LEVEL_NAMES = ['L0: Primitives', 'L1: Composites', 'L2: Architectural', 'L3: Models', 'L4: Meta', 'L5: External'];

  function renderList() {
    const panel = root.querySelector('#ng-listPanel');
    const filtered = getFilteredNeurons();

    // Group by level
    const groups = new Map();
    filtered.forEach(n => {
      if (!groups.has(n.level)) groups.set(n.level, []);
      groups.get(n.level).push(n);
    });

    let html = '';
    for (let lvl = 0; lvl <= 5; lvl++) {
      const items = groups.get(lvl);
      if (!items || items.length === 0) continue;
      items.sort((a, b) => a.name.localeCompare(b.name));

      html += `<div class="ng-level-group-header" style="color:var(--ng-level-${lvl})">${LEVEL_NAMES[lvl]} (${items.length})</div>`;
      items.forEach(n => {
        const sel = selectedNeuron && selectedNeuron.name === n.name ? ' selected' : '';
        html += `<div class="ng-neuron-item${sel}" data-name="${n.name}">
          <span class="ng-status-dot ${n.status}"></span>
          <span class="ng-name">${n.name}</span>
          <span class="ng-cat">${n.category}</span>
        </div>`;
      });
    }

    panel.innerHTML = html;

    // Update stats
    const totalShown = filtered.length;
    const implCount = filtered.filter(n => n.status === 'implemented').length;
    root.querySelector('#ng-stats').textContent = `${totalShown} of ${NEURONS.length} neurons, ${implCount} implemented`;

    // Add click handlers
    panel.querySelectorAll('.ng-neuron-item').forEach(el => {
      el.addEventListener('click', () => {
        const n = neuronMap.get(el.dataset.name);
        if (n) selectNeuron(n);
      });
    });
  }


  // ============================================================
  // RENDER DETAIL PANEL
  // ============================================================

  function selectNeuron(n) {
    selectedNeuron = n;
    window.location.hash = n.name;
    renderList();
    renderDetail(n);
  }

  function renderDetail(n) {
    const panel = root.querySelector('#ng-detailPanel');
    const deps = n.dependencies || [];
    const depBy = requiredByMap.get(n.name) || [];

    let html = '';

    // Header
    html += `<div class="ng-detail-header">
      <h2 style="color:var(--ng-level-${n.level})">${n.name}</h2>
      <span class="ng-detail-badge ${n.status}">${n.status}</span>
    </div>`;

    // Meta
    html += `<div class="ng-detail-meta">
      <span><span class="ng-label">Level:</span> ${n.level} (${LEVEL_NAMES[n.level].split(': ')[1]})</span>
      <span><span class="ng-label">Category:</span> ${n.category.replace(/_/g, ' ')}</span>
      ${n.implRef ? `<span><span class="ng-label">Impl:</span> ${n.implRef}</span>` : ''}
      ${n.source ? `<span><span class="ng-label">Source:</span> ${n.source}</span>` : ''}
    </div>`;

    // Description
    html += `<div class="ng-detail-desc">${n.description}</div>`;

    // Requires (upstream deps)
    html += `<div class="ng-dep-section">
      <h3><span class="ng-arrow up">&uarr;</span> Requires (${deps.length})</h3>`;
    if (deps.length > 0) {
      html += `<div class="ng-dep-list">`;
      deps.forEach(d => {
        const dn = neuronMap.get(d);
        const st = dn ? dn.status : 'planned';
        html += `<span class="ng-dep-chip ${st}" data-name="${d}">${d}</span>`;
      });
      html += `</div>`;
    } else {
      html += `<span class="ng-dep-none">No dependencies (primitive or self-contained)</span>`;
    }
    html += `</div>`;

    // Required By (downstream dependents)
    html += `<div class="ng-dep-section">
      <h3><span class="ng-arrow down">&darr;</span> Required By (${depBy.length})</h3>`;
    if (depBy.length > 0) {
      html += `<div class="ng-dep-list">`;
      [...depBy].sort().forEach(d => {
        const dn = neuronMap.get(d);
        const st = dn ? dn.status : 'planned';
        html += `<span class="ng-dep-chip ${st}" data-name="${d}">${d}</span>`;
      });
      html += `</div>`;
    } else {
      html += `<span class="ng-dep-none">No neurons depend on this yet</span>`;
    }
    html += `</div>`;

    // SVG Graph
    html += `<div class="ng-graph-container">
      <h3>Dependency Graph</h3>
      <div id="ng-graphSvg"></div>
    </div>`;

    panel.innerHTML = html;

    // Click handlers on chips
    panel.querySelectorAll('.ng-dep-chip').forEach(el => {
      el.addEventListener('click', () => {
        const target = neuronMap.get(el.dataset.name);
        if (target) selectNeuron(target);
      });
    });

    // Render SVG
    renderGraph(n, deps, depBy);
  }


  // ============================================================
  // SVG DEPENDENCY GRAPH (3-column layout)
  // ============================================================

  function renderGraph(n, deps, depBy) {
    const container = root.querySelector('#ng-graphSvg');

    const nodeW = 140, nodeH = 30, padX = 30, padY = 14;
    const colGap = 100;

    const leftNodes = deps.map(d => neuronMap.get(d) || { name: d, status: 'planned', level: 0 });
    const rightNodes = depBy.map(d => neuronMap.get(d) || { name: d, status: 'planned', level: 0 });

    const maxSide = Math.max(leftNodes.length, rightNodes.length, 1);
    const totalH = Math.max(maxSide * (nodeH + padY) + padY + 40, nodeH + padY * 2 + 40);
    const totalW = nodeW * 3 + colGap * 2 + padX * 2;

    // Column x-positions
    const colL = padX;
    const colC = padX + nodeW + colGap;
    const colR = padX + (nodeW + colGap) * 2;

    // Y center for each column
    function yPos(idx, count) {
      const blockH = count * (nodeH + padY) - padY;
      const startY = (totalH - blockH) / 2;
      return startY + idx * (nodeH + padY);
    }
    const centerY = (totalH - nodeH) / 2;

    // Level color
    function lvlColor(nn) { return `var(--ng-level-${nn.level || 0})`; }
    function statusColor(nn) { return nn.status === 'implemented' ? 'var(--ng-implemented)' : 'var(--ng-planned)'; }

    let svg = `<svg viewBox="0 0 ${totalW} ${totalH}" xmlns="http://www.w3.org/2000/svg" style="min-height:${Math.min(totalH, 500)}px">`;

    // Draw edges first (behind nodes)
    // Left -> Center
    leftNodes.forEach((ln, i) => {
      const x1 = colL + nodeW;
      const y1 = yPos(i, leftNodes.length) + nodeH / 2;
      const x2 = colC;
      const y2 = centerY + nodeH / 2;
      const cpx = (x1 + x2) / 2;
      svg += `<path d="M${x1},${y1} C${cpx},${y1} ${cpx},${y2} ${x2},${y2}" fill="none" stroke="var(--ng-edge-dep)" stroke-width="1.5" opacity="0.5"/>`;
    });

    // Center -> Right
    rightNodes.forEach((rn, i) => {
      const x1 = colC + nodeW;
      const y1 = centerY + nodeH / 2;
      const x2 = colR;
      const y2 = yPos(i, rightNodes.length) + nodeH / 2;
      const cpx = (x1 + x2) / 2;
      svg += `<path d="M${x1},${y1} C${cpx},${y1} ${cpx},${y2} ${x2},${y2}" fill="none" stroke="var(--ng-edge-depby)" stroke-width="1.5" opacity="0.5"/>`;
    });

    // Draw nodes
    function drawNode(x, y, nn, isCenter) {
      const fill = isCenter ? 'var(--ng-bg-hover)' : 'var(--ng-bg-card)';
      const border = isCenter ? 'var(--ng-node-selected)' : 'var(--ng-border)';
      const sw = isCenter ? 2 : 1;
      const r = 6;
      const dotColor = statusColor(nn);
      const textColor = isCenter ? 'var(--ng-text)' : lvlColor(nn);
      const truncName = nn.name.length > 16 ? nn.name.slice(0, 15) + '\u2026' : nn.name;

      svg += `<g class="ng-graph-node" data-name="${nn.name}" style="cursor:pointer">`;
      svg += `<rect x="${x}" y="${y}" width="${nodeW}" height="${nodeH}" rx="${r}" fill="${fill}" stroke="${border}" stroke-width="${sw}"/>`;
      svg += `<circle cx="${x + 10}" cy="${y + nodeH/2}" r="4" fill="${dotColor}"/>`;
      svg += `<text x="${x + 20}" y="${y + nodeH/2 + 4}" fill="${textColor}" font-size="11" font-weight="${isCenter ? 700 : 500}">${truncName}</text>`;
      svg += `</g>`;
    }

    // Left nodes (deps)
    leftNodes.forEach((ln, i) => drawNode(colL, yPos(i, leftNodes.length), ln, false));

    // Center node (selected)
    drawNode(colC, centerY, n, true);

    // Right nodes (depBy)
    rightNodes.forEach((rn, i) => drawNode(colR, yPos(i, rightNodes.length), rn, false));

    // Labels
    if (leftNodes.length > 0) {
      svg += `<text x="${colL + nodeW/2}" y="16" fill="var(--ng-text-dim)" font-size="10" text-anchor="middle" font-weight="600">REQUIRES</text>`;
    }
    if (rightNodes.length > 0) {
      svg += `<text x="${colR + nodeW/2}" y="16" fill="var(--ng-text-dim)" font-size="10" text-anchor="middle" font-weight="600">REQUIRED BY</text>`;
    }

    svg += `</svg>`;
    container.innerHTML = svg;

    // Make graph nodes clickable
    container.querySelectorAll('.ng-graph-node').forEach(el => {
      el.addEventListener('click', () => {
        const target = neuronMap.get(el.dataset.name);
        if (target) selectNeuron(target);
      });
    });
  }


  // ============================================================
  // EVENT HANDLERS
  // ============================================================

  // Search (debounced)
  let searchTimer = null;
  const searchInput = root.querySelector('#ng-search');
  function onSearchInput(e) {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      searchQuery = e.target.value.trim();
      renderList();
    }, 150);
  }
  searchInput.addEventListener('input', onSearchInput);

  // Level pills
  root.querySelectorAll('.ng-level-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const lvl = parseInt(pill.dataset.level);
      if (activeLevels.has(lvl)) {
        activeLevels.delete(lvl);
        pill.classList.remove('active');
      } else {
        activeLevels.add(lvl);
        pill.classList.add('active');
      }
      renderList();
    });
  });

  // Status filter
  const statusSelect = root.querySelector('#ng-statusFilter');
  function onStatusChange(e) {
    statusFilter = e.target.value;
    renderList();
  }
  statusSelect.addEventListener('change', onStatusChange);

  // Category filter
  const categorySelect = root.querySelector('#ng-categoryFilter');
  function onCategoryChange(e) {
    categoryFilter = e.target.value;
    renderList();
  }
  categorySelect.addEventListener('change', onCategoryChange);

  // URL hash navigation
  function handleHash() {
    const hash = window.location.hash.slice(1);
    if (hash) {
      const decoded = decodeURIComponent(hash);
      const n = neuronMap.get(decoded);
      if (n) {
        selectedNeuron = n;
        renderList();
        renderDetail(n);
        // Scroll list to selected
        setTimeout(() => {
          const el = root.querySelector(`.ng-neuron-item[data-name="${n.name}"]`);
          if (el) el.scrollIntoView({ block: 'center', behavior: 'smooth' });
        }, 50);
        return;
      }
    }
  }
  window.addEventListener('hashchange', handleHash);


  // ============================================================
  // INITIAL RENDER
  // ============================================================

  renderList();
  handleHash();


  // ============================================================
  // CLEANUP
  // ============================================================

  return function cleanup() {
    clearTimeout(searchTimer);
    window.removeEventListener('hashchange', handleHash);
    searchInput.removeEventListener('input', onSearchInput);
    statusSelect.removeEventListener('change', onStatusChange);
    categorySelect.removeEventListener('change', onCategoryChange);
  };
}
