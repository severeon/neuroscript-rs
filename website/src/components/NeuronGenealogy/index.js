import React, { useState, useMemo, useEffect, useCallback } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import { usePluginData } from '@docusaurus/useGlobalData';
import { NEURONS } from '../neuron-genealogy-data';
import NeuronControls from './NeuronControls';
import NeuronList from './NeuronList';
import NeuronDetail from './NeuronDetail';
import { LEVEL_NAMES } from './constants';
import '../../css/neuron-genealogy.css';

function NeuronGenealogyInner() {
  const neuronDocsData = usePluginData('neuron-docs-plugin') ?? { neurons: {}, sources: {} };

  // ---- State ----
  const [selectedNeuron, setSelectedNeuron] = useState(null);
  const [activeLevels, setActiveLevels] = useState(() => new Set(LEVEL_NAMES.map((_, i) => i)));
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
    history.replaceState(null, '', '#' + encodeURIComponent(name));
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
  const selectedNeuronDocs = useMemo(() => {
    if (!selectedNeuron) return null;
    const entry = neuronDocsData.neurons?.[selectedNeuron];
    if (!entry) return null;
    return {
      ...entry,
      source: neuronDocsData.sources?.[entry.sourceFile] || null,
    };
  }, [selectedNeuron, neuronDocsData]);
  const selectedRequiredBy = selectedNeuron ? (requiredByMap.get(selectedNeuron) || []) : [];

  // ---- Render ----
  return (
    <div className="neuron-genealogy">
      <NeuronControls
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        activeLevels={activeLevels}
        onToggleLevel={handleToggleLevel}
        statusFilter={statusFilter}
        onStatusChange={setStatusFilter}
        categoryFilter={categoryFilter}
        onCategoryChange={setCategoryFilter}
        categories={categories}
        stats={stats}
      />

      <div className={`ng-main${mobileShowDetail ? ' ng-mobile-showing-detail' : ''}`}>
        <NeuronList
          neurons={filteredNeurons}
          selectedNeuron={selectedNeuron}
          onSelectNeuron={handleSelectNeuron}
        />

        <NeuronDetail
          neuron={selectedNeuronData}
          neuronDocs={selectedNeuronDocs}
          neuronMap={neuronMap}
          requiredBy={selectedRequiredBy}
          onSelectNeuron={handleSelectNeuron}
          onBack={mobileShowDetail ? handleBack : null}
        />
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
