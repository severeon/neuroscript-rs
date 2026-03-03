import React, { useState, useEffect } from 'react';

const LEVEL_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5'];

export default function NeuronControls({
  searchQuery,
  onSearchChange,
  activeLevels,
  onToggleLevel,
  statusFilter,
  onStatusChange,
  categoryFilter,
  onCategoryChange,
  categories,
  stats,
}) {
  // Internal state for debounced search
  const [localSearch, setLocalSearch] = useState(searchQuery);

  useEffect(() => {
    const timer = setTimeout(() => {
      onSearchChange(localSearch);
    }, 150);
    return () => clearTimeout(timer);
  }, [localSearch, onSearchChange]);

  // Sync external changes back to local
  useEffect(() => {
    setLocalSearch(searchQuery);
  }, [searchQuery]);

  return (
    <div className="ng-controls">
      <div className="ng-logo">
        NeuroScript <span>Neuron Genealogy</span>
      </div>

      <input
        type="text"
        className="ng-search-box"
        placeholder="Search neurons..."
        autoComplete="off"
        value={localSearch}
        onChange={(e) => setLocalSearch(e.target.value)}
      />

      <div className="ng-level-pills">
        {LEVEL_NAMES.map((name, level) => (
          <div
            key={level}
            className={`ng-level-pill${activeLevels.has(level) ? ' active' : ''}`}
            data-level={level}
            onClick={() => onToggleLevel(level)}
          >
            {name}
          </div>
        ))}
      </div>

      <select
        className="ng-ctrl-select"
        value={statusFilter}
        onChange={(e) => onStatusChange(e.target.value)}
      >
        <option value="all">All status</option>
        <option value="implemented">Implemented</option>
        <option value="planned">Planned</option>
      </select>

      <select
        className="ng-ctrl-select"
        value={categoryFilter}
        onChange={(e) => onCategoryChange(e.target.value)}
      >
        <option value="all">All categories</option>
        {categories.map((cat) => (
          <option key={cat} value={cat}>
            {cat.replace(/_/g, ' ')}
          </option>
        ))}
      </select>

      <div className="ng-stats">
        {stats.shown} of {stats.total} neurons, {stats.implemented} implemented
      </div>
    </div>
  );
}
