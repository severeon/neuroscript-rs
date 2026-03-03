import React, { useState, useMemo, useEffect, useCallback } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import { NEURONS } from '../neuron-genealogy-data';
import '../../css/neuron-genealogy.css';

// Try to import build-time docs data (may not exist during dev)
let neuronDocsData = {};
try {
  neuronDocsData = require('@site/.docusaurus/neuron-docs-plugin/default/neuron-docs.json');
} catch (e) {
  // Plugin data not available yet
}

const LEVEL_NAMES = [
  'L0: Primitives',
  'L1: Composites',
  'L2: Architectural',
  'L3: Models',
  'L4: Meta',
  'L5: External',
];

function NeuronGenealogyInner() {
  // ---- State ----
  const [selectedNeuron, setSelectedNeuron] = useState(null);
  const [activeLevels, setActiveLevels] = useState(() => new Set([0, 1, 2, 3, 4, 5]));
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [mobileShowDetail, setMobileShowDetail] = useState(false);

  // ---- Computed indexes (stable across renders) ----
  const neuronMap = useMemo(() => {
    const map = new Map();
    NEURONS.forEach(n => map.set(n.name, n));
    return map;
  }, []);

  const requiredByMap = useMemo(() => {
    const map = new Map();
    NEURONS.forEach(n => {
      if (!map.has(n.name)) map.set(n.name, []);
    });
    NEURONS.forEach(n => {
      (n.dependencies || []).forEach(dep => {
        if (!map.has(dep)) map.set(dep, []);
        map.get(dep).push(n.name);
      });
    });
    return map;
  }, []);

  const categories = useMemo(() => {
    const cats = new Set();
    NEURONS.forEach(n => cats.add(n.category));
    return [...cats].sort();
  }, []);

  // ---- Filtered neurons ----
  const filteredNeurons = useMemo(() => {
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
  }, [activeLevels, searchQuery, statusFilter, categoryFilter]);

  // ---- Stats ----
  const stats = useMemo(() => ({
    shown: filteredNeurons.length,
    total: NEURONS.length,
    implemented: filteredNeurons.filter(n => n.status === 'implemented').length,
  }), [filteredNeurons]);

  // ---- Callbacks ----
  const handleSelectNeuron = useCallback((name) => {
    setSelectedNeuron(name);
    setMobileShowDetail(true);
    window.location.hash = name;
  }, []);

  const handleToggleLevel = useCallback((level) => {
    setActiveLevels(prev => {
      const next = new Set(prev);
      if (next.has(level)) {
        next.delete(level);
      } else {
        next.add(level);
      }
      return next;
    });
  }, []);

  const handleBack = useCallback(() => {
    setMobileShowDetail(false);
  }, []);

  // ---- URL hash sync ----
  useEffect(() => {
    const hash = window.location.hash.slice(1);
    if (hash) {
      const decoded = decodeURIComponent(hash);
      if (neuronMap.has(decoded)) {
        setSelectedNeuron(decoded);
      }
    }

    const onHashChange = () => {
      const h = window.location.hash.slice(1);
      if (h) {
        const d = decodeURIComponent(h);
        if (neuronMap.has(d)) {
          setSelectedNeuron(d);
        }
      }
    };

    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, [neuronMap]);

  // ---- Get selected neuron data + docs ----
  const selectedNeuronData = selectedNeuron ? neuronMap.get(selectedNeuron) : null;
  const selectedNeuronDocs = selectedNeuron ? neuronDocsData[selectedNeuron] || null : null;
  const selectedRequiredBy = selectedNeuron ? (requiredByMap.get(selectedNeuron) || []) : [];

  // ---- Render ----
  return (
    <div className="neuron-genealogy">
      {/* NeuronControls will go here */}
      <div className="ng-controls">
        <div className="ng-logo">NeuroScript <span>Neuron Genealogy</span></div>
        <div className="ng-stats">{stats.shown} of {stats.total} neurons, {stats.implemented} implemented</div>
      </div>

      <div className={`ng-main${mobileShowDetail ? ' ng-mobile-showing-detail' : ''}`}>
        {/* NeuronList will go here */}
        <div className="ng-list-panel" id="ng-listPanel">
          <div style={{ padding: '20px', color: 'var(--ng-text-muted)' }}>
            List placeholder — {filteredNeurons.length} neurons
          </div>
        </div>

        {/* NeuronDetail will go here */}
        <div className="ng-detail-panel" id="ng-detailPanel">
          {selectedNeuronData ? (
            <div style={{ padding: '20px' }}>
              <h2 style={{ color: `var(--ng-level-${selectedNeuronData.level})` }}>{selectedNeuronData.name}</h2>
              <p>{selectedNeuronData.description}</p>
              {selectedNeuronDocs && selectedNeuronDocs.docComment && (
                <div className="ng-doc-section">
                  <h3>Documentation</h3>
                  <p>{selectedNeuronDocs.docComment}</p>
                </div>
              )}
            </div>
          ) : (
            <div className="ng-detail-empty">
              Select a neuron from the list<br />to view its dependencies and details
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function NeuronGenealogy() {
  return (
    <BrowserOnly fallback={<div style={{ height: 'calc(100vh - 60px)' }} />}>
      {() => <NeuronGenealogyInner />}
    </BrowserOnly>
  );
}
