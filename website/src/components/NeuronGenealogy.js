import React, { useRef, useEffect } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import { NEURONS } from './neuron-genealogy-data';
import { initNeuronGenealogy } from './neuron-genealogy-logic';
import '../css/neuron-genealogy.css';

/**
 * HTML structure for the neuron genealogy tool.
 * Injected into the React-managed container via innerHTML.
 */
const GENEALOGY_HTML = `
<!-- Controls Bar -->
<div class="ng-controls">
  <div class="ng-logo">NeuroScript <span>Neuron Genealogy</span></div>
  <input type="text" class="ng-search-box" id="ng-search" placeholder="Search neurons..." autocomplete="off">
  <div class="ng-level-pills" id="ng-levelPills">
    <div class="ng-level-pill active" data-level="0">L0</div>
    <div class="ng-level-pill active" data-level="1">L1</div>
    <div class="ng-level-pill active" data-level="2">L2</div>
    <div class="ng-level-pill active" data-level="3">L3</div>
    <div class="ng-level-pill active" data-level="4">L4</div>
    <div class="ng-level-pill active" data-level="5">L5</div>
  </div>
  <select class="ng-ctrl-select" id="ng-statusFilter">
    <option value="all">All status</option>
    <option value="implemented">Implemented</option>
    <option value="planned">Planned</option>
  </select>
  <select class="ng-ctrl-select" id="ng-categoryFilter">
    <option value="all">All categories</option>
  </select>
  <div class="ng-stats" id="ng-stats"></div>
</div>

<!-- Main two-panel layout -->
<div class="ng-main">
  <div class="ng-list-panel" id="ng-listPanel"></div>
  <div class="ng-detail-panel" id="ng-detailPanel">
    <div class="ng-detail-empty">Select a neuron from the list<br>to view its dependencies and details</div>
  </div>
</div>
`;

function NeuronGenealogyInner() {
  const containerRef = useRef(null);

  useEffect(() => {
    const root = containerRef.current;
    if (!root) return;

    root.innerHTML = GENEALOGY_HTML;
    const cleanup = initNeuronGenealogy(root, NEURONS);

    return () => {
      if (cleanup) cleanup();
      root.innerHTML = '';
    };
  }, []);

  return <div className="neuron-genealogy" ref={containerRef} />;
}

export default function NeuronGenealogy() {
  return (
    <BrowserOnly fallback={<div style={{ height: 'calc(100vh - 60px)', background: '#0d1117' }} />}>
      {() => <NeuronGenealogyInner />}
    </BrowserOnly>
  );
}
