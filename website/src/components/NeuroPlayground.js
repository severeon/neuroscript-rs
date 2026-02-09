import React, { useState, useEffect, useCallback, useRef } from 'react';
import init, { compile, list_neurons } from '@site/static/wasm/neuroscript.js';
import { EXAMPLES, getExamplesByCategory } from './PlaygroundExamples';
import { STDLIB_BUNDLE } from './StdlibBundle';

const DEFAULT_EXAMPLE_ID = 'mlp';

export default function NeuroPlayground() {
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');
  const [ready, setReady] = useState(false);
  const [selectedExample, setSelectedExample] = useState(DEFAULT_EXAMPLE_ID);
  const [availableNeurons, setAvailableNeurons] = useState([]);
  const [selectedNeuron, setSelectedNeuron] = useState(null);
  const [copied, setCopied] = useState(false);
  const [compiling, setCompiling] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  const compileTimerRef = useRef(null);
  const examplesByCategory = getExamplesByCategory();

  // Initialize WASM
  useEffect(() => {
    init().then(() => {
      setReady(true);
      const defaultExample = EXAMPLES.find(ex => ex.id === DEFAULT_EXAMPLE_ID);
      if (defaultExample) {
        loadExample(defaultExample);
      }
    }).catch(e => {
      console.error("WASM init failed:", e);
      setError("Failed to initialize compiler: " + e);
    });
  }, []);

  // Detect mobile layout
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };

    // Initial check
    checkMobile();

    // Listen for resize
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Debounced compile
  const debouncedCompile = useCallback((source, neuronName) => {
    if (compileTimerRef.current) {
      clearTimeout(compileTimerRef.current);
    }
    compileTimerRef.current = setTimeout(() => {
      performCompile(source, neuronName);
    }, 500);
  }, []);

  // Perform compilation
  const performCompile = useCallback((source, neuronName = null) => {
    if (!ready) return;

    setCompiling(true);
    try {
      // Remove 'use' statements (examples don't need them since we bundle stdlib)
      const cleanSource = source.replace(/^use .*$/gm, '# $&');

      // Prepend stdlib bundle
      const fullSource = STDLIB_BUNDLE + '\n' + cleanSource;

      // Get list of available neurons
      const neuronsJson = list_neurons(fullSource);
      const neurons = JSON.parse(neuronsJson);
      setAvailableNeurons(neurons);

      // Determine which neuron to compile
      const targetNeuron = neuronName || (neurons.length === 1 ? neurons[0] : null);

      // Compile
      const result = compile(fullSource, targetNeuron);
      setOutput(result);
      setError('');

      // Update selected neuron if not manually set
      if (!neuronName && targetNeuron) {
        setSelectedNeuron(targetNeuron);
      }
    } catch (e) {
      setError(e.toString());
      setOutput('');
      setAvailableNeurons([]);
    } finally {
      setCompiling(false);
    }
  }, [ready]);

  // Load an example
  const loadExample = useCallback((example) => {
    setInput(example.code);
    setSelectedExample(example.id);
    setSelectedNeuron(example.targetNeuron);
    performCompile(example.code, example.targetNeuron);
  }, [performCompile]);

  // Handle example selection
  const handleExampleChange = (e) => {
    const exampleId = e.target.value;
    const example = EXAMPLES.find(ex => ex.id === exampleId);
    if (example) {
      loadExample(example);
    }
  };

  // Handle code input change
  const handleInputChange = (e) => {
    const newCode = e.target.value;
    setInput(newCode);
    // Clear any pending compilation
    if (compileTimerRef.current) {
      clearTimeout(compileTimerRef.current);
      compileTimerRef.current = null;
    }
  };

  // Handle manual compile button
  const handleManualCompile = () => {
    performCompile(input, selectedNeuron);
  };

  // Handle neuron selection
  const handleNeuronChange = (e) => {
    const neuron = e.target.value;
    setSelectedNeuron(neuron);
  };

  // Handle copy to clipboard
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(output);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error('Failed to copy:', e);
    }
  };

  // Get current example for feature display
  const currentExample = EXAMPLES.find(ex => ex.id === selectedExample);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '1rem',
      maxWidth: '100%'
    }}>
      {/* Top controls */}
      <div style={{
        display: 'flex',
        gap: '1rem',
        alignItems: 'center',
        flexWrap: 'wrap'
      }}>
        <div style={{ flex: '1', minWidth: '250px' }}>
          <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
            Example:
          </label>
          <select
            value={selectedExample}
            onChange={handleExampleChange}
            disabled={!ready}
            style={{
              width: '100%',
              padding: '0.5rem',
              borderRadius: '4px',
              border: '1px solid var(--ifm-color-emphasis-300)',
              backgroundColor: 'var(--ifm-background-color)',
              color: 'var(--ifm-font-color-base)',
              fontSize: '14px'
            }}
          >
            {Object.entries(examplesByCategory).map(([category, examples]) => (
              <optgroup key={category} label={category}>
                {examples.map(ex => (
                  <option key={ex.id} value={ex.id}>
                    {ex.title}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>

        {availableNeurons.length > 1 && (
          <div style={{ minWidth: '200px' }}>
            <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
              Compile:
            </label>
            <select
              value={selectedNeuron || ''}
              onChange={handleNeuronChange}
              disabled={!ready}
              style={{
                width: '100%',
                padding: '0.5rem',
                borderRadius: '4px',
                border: '1px solid var(--ifm-color-emphasis-300)',
                backgroundColor: 'var(--ifm-background-color)',
                color: 'var(--ifm-font-color-base)',
                fontSize: '14px'
              }}
            >
              <option value="">Auto-detect</option>
              {availableNeurons.map(neuron => (
                <option key={neuron} value={neuron}>
                  {neuron}
                </option>
              ))}
            </select>
          </div>
        )}

        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <button
            onClick={handleManualCompile}
            disabled={!ready || compiling}
            className="button button--primary"
            style={{ fontSize: '14px' }}
          >
            {compiling ? '⏳ Compiling...' : '▶️ Compile'}
          </button>
        </div>
      </div>

      {/* Example description and features */}
      {currentExample && (
        <div style={{
          padding: '0.75rem',
          backgroundColor: 'var(--ifm-color-emphasis-100)',
          borderRadius: '8px',
          borderLeft: '4px solid var(--ifm-color-primary)'
        }}>
          <p style={{ margin: '0 0 0.5rem 0', fontWeight: 'bold' }}>
            {currentExample.description}
          </p>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {currentExample.features.map(feature => (
              <span
                key={feature}
                style={{
                  padding: '0.25rem 0.5rem',
                  backgroundColor: 'var(--ifm-color-primary)',
                  color: 'white',
                  borderRadius: '12px',
                  fontSize: '12px',
                  fontWeight: '500'
                }}
              >
                {feature}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Main editor area */}
      <div style={{
        display: 'flex',
        gap: '1rem',
        minHeight: '500px',
        flexDirection: isMobile ? 'column' : 'row'
      }}>
        {/* Source panel */}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '0.5rem'
          }}>
            <h3 style={{ margin: 0 }}>Source (NeuroScript)</h3>
            {compiling && (
              <span style={{ fontSize: '12px', color: 'var(--ifm-color-primary)' }}>
                Compiling...
              </span>
            )}
          </div>
          <textarea
            value={input}
            onChange={handleInputChange}
            disabled={!ready}
            style={{
              flex: 1,
              fontFamily: '"Fira Code", "Cascadia Code", "SF Mono", Monaco, "Inconsolata", "Roboto Mono", "Source Code Pro", Menlo, Consolas, "DejaVu Sans Mono", monospace',
              padding: '1rem',
              resize: 'none',
              borderRadius: '8px',
              border: '1px solid var(--ifm-color-emphasis-300)',
              backgroundColor: 'var(--ifm-background-color)',
              color: 'var(--ifm-font-color-base)',
              fontSize: '13px',
              lineHeight: '1.6',
              minHeight: '400px'
            }}
            spellCheck="false"
          />
        </div>

        {/* Output panel */}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '0.5rem'
          }}>
            <h3 style={{ margin: 0 }}>Output (PyTorch)</h3>
            {output && (
              <button
                onClick={handleCopy}
                className="button button--sm button--secondary"
                style={{ fontSize: '12px' }}
              >
                {copied ? '✓ Copied!' : '📋 Copy'}
              </button>
            )}
          </div>

          {error ? (
            <div style={{
              flex: 1,
              backgroundColor: 'var(--ifm-color-danger-contrast-background)',
              color: 'var(--ifm-color-danger-contrast-foreground)',
              padding: '1rem',
              fontFamily: 'monospace',
              whiteSpace: 'pre-wrap',
              overflow: 'auto',
              borderRadius: '8px',
              border: '1px solid var(--ifm-color-danger)',
              fontSize: '13px',
              lineHeight: '1.6',
              minHeight: '400px'
            }}>
              {error}
            </div>
          ) : (
            <div style={{
              flex: 1,
              fontFamily: '"Fira Code", "Cascadia Code", "SF Mono", Monaco, "Inconsolata", "Roboto Mono", "Source Code Pro", Menlo, Consolas, "DejaVu Sans Mono", monospace',
              padding: '1rem',
              backgroundColor: '#1e1e1e',
              color: '#d4d4d4',
              overflow: 'auto',
              borderRadius: '8px',
              border: '1px solid #333',
              whiteSpace: 'pre',
              fontSize: '13px',
              lineHeight: '1.6',
              minHeight: '400px'
            }}>
              {output || 'Compiled PyTorch code will appear here...'}
            </div>
          )}
        </div>
      </div>

      {/* Stats footer */}
      {output && (
        <div style={{
          padding: '0.5rem',
          backgroundColor: 'var(--ifm-color-emphasis-100)',
          borderRadius: '4px',
          fontSize: '12px',
          color: 'var(--ifm-color-emphasis-700)',
          textAlign: 'right'
        }}>
          {availableNeurons.length} neuron{availableNeurons.length !== 1 ? 's' : ''} • {output.split('\n').length} lines generated
        </div>
      )}
    </div>
  );
}
