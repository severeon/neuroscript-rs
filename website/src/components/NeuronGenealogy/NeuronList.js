import React, { useMemo, useEffect, useRef } from 'react';
import { LEVEL_NAMES } from './constants';

export default function NeuronList({ neurons, selectedNeuron, onSelectNeuron }) {
  const listRef = useRef(null);
  const selectedRef = useRef(null);

  // Group neurons by level
  const groups = useMemo(() => {
    const map = new Map();
    neurons.forEach((n) => {
      if (!map.has(n.level)) map.set(n.level, []);
      map.get(n.level).push(n);
    });
    // Sort within each group
    for (const [, items] of map) {
      items.sort((a, b) => a.name.localeCompare(b.name));
    }
    return map;
  }, [neurons]);

  // Auto-scroll selected into view
  useEffect(() => {
    if (selectedRef.current) {
      selectedRef.current.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  }, [selectedNeuron]);

  return (
    <div className="ng-list-panel" ref={listRef}>
      {Array.from({ length: LEVEL_NAMES.length }, (_, i) => i).map((level) => {
        const items = groups.get(level);
        if (!items || items.length === 0) return null;
        return (
          <React.Fragment key={level}>
            <div
              className="ng-level-group-header"
              style={{ color: `var(--ng-level-${level})` }}
            >
              {LEVEL_NAMES[level]} ({items.length})
            </div>
            {items.map((n) => {
              const isSelected = selectedNeuron === n.name;
              return (
                <div
                  key={n.name}
                  ref={isSelected ? selectedRef : null}
                  className={`ng-neuron-item${isSelected ? ' selected' : ''}`}
                  data-name={n.name}
                  role="button"
                  tabIndex={0}
                  onClick={() => onSelectNeuron(n.name)}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelectNeuron(n.name); } }}
                >
                  <span className={`ng-status-dot ${n.status}`} />
                  <span className="ng-name">{n.name}</span>
                  <span className="ng-cat">{n.category}</span>
                </div>
              );
            })}
          </React.Fragment>
        );
      })}
    </div>
  );
}
