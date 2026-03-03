import React from 'react';
import DependencyGraph from './DependencyGraph';
import NeuronSourceCode from './NeuronSourceCode';
import { LEVEL_NAMES } from './constants';

export default function NeuronDetail({
  neuron,
  neuronDocs,
  neuronMap,
  requiredBy,
  onSelectNeuron,
  onBack,
}) {
  if (!neuron) {
    return (
      <div className="ng-detail-panel">
        <div className="ng-detail-empty">
          Select a neuron from the list<br />to view its dependencies and details
        </div>
      </div>
    );
  }

  const deps = neuron.dependencies || [];
  const depNeurons = deps.map(d => neuronMap.get(d) || { name: d, level: 0, status: 'planned' });
  const reqByNeurons = requiredBy.map(d => neuronMap.get(d) || { name: d, level: 0, status: 'planned' });

  return (
    <div className="ng-detail-panel">
      {/* Mobile back button */}
      {onBack && (
        <button className="ng-back-button" onClick={onBack}>
          &larr; Back to list
        </button>
      )}

      {/* Header */}
      <div className="ng-detail-header">
        <h2 style={{ color: `var(--ng-level-${neuron.level})` }}>{neuron.name}</h2>
        <span className={`ng-detail-badge ${neuron.status}`}>{neuron.status}</span>
      </div>

      {/* Meta */}
      <div className="ng-detail-meta">
        <span><span className="ng-label">Level:</span> {neuron.level} ({LEVEL_NAMES[neuron.level]?.split(': ')[1] ?? 'Unknown'})</span>
        <span><span className="ng-label">Category:</span> {neuron.category.replace(/_/g, ' ')}</span>
        {neuron.implRef && <span><span className="ng-label">Impl:</span> {neuron.implRef}</span>}
        {neuron.source && <span><span className="ng-label">Source:</span> {neuron.source}</span>}
      </div>

      {/* Description */}
      <div className="ng-detail-desc">{neuron.description}</div>

      {/* Documentation from /// doc comments */}
      {neuronDocs && neuronDocs.docComment && (
        <div className="ng-doc-section">
          <h3>Documentation</h3>
          <div className="ng-detail-desc">
            {neuronDocs.docComment.split('\n').map((line, i) => (
              <React.Fragment key={i}>
                {line}
                {i < neuronDocs.docComment.split('\n').length - 1 && <br />}
              </React.Fragment>
            ))}
          </div>
        </div>
      )}

      {/* Source Code (collapsible) */}
      <NeuronSourceCode
        source={neuronDocs?.source}
        sourceFile={neuronDocs?.sourceFile}
      />

      {/* Requires (upstream deps) */}
      <div className="ng-dep-section">
        <h3><span className="ng-arrow up">&uarr;</span> Requires ({deps.length})</h3>
        {deps.length > 0 ? (
          <div className="ng-dep-list">
            {deps.map(d => {
              const dn = neuronMap.get(d);
              const st = dn ? dn.status : 'planned';
              return (
                <span
                  key={d}
                  className={`ng-dep-chip ${st}`}
                  role="button"
                  tabIndex={0}
                  onClick={() => onSelectNeuron(d)}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelectNeuron(d); } }}
                >
                  {d}
                </span>
              );
            })}
          </div>
        ) : (
          <span className="ng-dep-none">No dependencies (primitive or self-contained)</span>
        )}
      </div>

      {/* Required By (downstream dependents) */}
      <div className="ng-dep-section">
        <h3><span className="ng-arrow down">&darr;</span> Required By ({requiredBy.length})</h3>
        {requiredBy.length > 0 ? (
          <div className="ng-dep-list">
            {[...requiredBy].sort().map(d => {
              const dn = neuronMap.get(d);
              const st = dn ? dn.status : 'planned';
              return (
                <span
                  key={d}
                  className={`ng-dep-chip ${st}`}
                  role="button"
                  tabIndex={0}
                  onClick={() => onSelectNeuron(d)}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelectNeuron(d); } }}
                >
                  {d}
                </span>
              );
            })}
          </div>
        ) : (
          <span className="ng-dep-none">No neurons depend on this yet</span>
        )}
      </div>

      {/* Dependency Graph */}
      <DependencyGraph
        neuron={neuron}
        dependencies={depNeurons}
        requiredBy={reqByNeurons}
        onSelectNeuron={onSelectNeuron}
      />
    </div>
  );
}
