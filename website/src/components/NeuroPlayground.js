import React, { useState, useEffect } from 'react';
import init, { compile } from '@site/static/wasm/neuroscript.js';
import { STDLIB_BUNDLE } from './StdlibBundle';

const SAMPLE_CODE = `neuron MLP(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in ->
      Linear(dim, dim * 4)
      GELU()
      Linear(dim * 4, dim)
      out`;

export default function NeuroPlayground() {
  const [input, setInput] = useState(SAMPLE_CODE);
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');
  const [ready, setReady] = useState(false);

  useEffect(() => {
    init().then(() => {
      setReady(true);
      handleCompile(SAMPLE_CODE);
    }).catch(e => {
      console.error("WASM init failed:", e);
      setError("Failed to initialize compiler: " + e);
    });
  }, []);

  const handleCompile = (source) => {
    if (!ready) return;
    try {
      // Prepend stdlib bundle and remove 'use' statements to avoid resolution errors
      const cleanSource = source.replace(/^use .*$/gm, '# $&');
      const fullSource = STDLIB_BUNDLE + '\n' + cleanSource;
      const result = compile(fullSource);
      setOutput(result);
      setError('');
    } catch (e) {
      setError(e.toString());
      setOutput('');
    }
  };

  const handleChange = (e) => {
    const newVal = e.target.value;
    setInput(newVal);
    handleCompile(newVal);
  };

  const generateRandom = () => {
    const templates = [
      // Simple MLP
      () => {
        const d = [64, 128, 256, 512][Math.floor(Math.random() * 4)];
        const expansion = d * [2, 4][Math.floor(Math.random() * 2)];
        return `neuron RandomMLP(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in ->
      Linear(dim, ${expansion})
      GELU()
      Linear(${expansion}, dim)
      out`;
      },
      // Residual Block
      () => {
        const d = [128, 256, 512][Math.floor(Math.random() * 3)];
        return `neuron RandomResidual(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Fork() -> (main, skip)
    main -> 
      LayerNorm(dim)
      Linear(dim, dim)
      GELU()
      Linear(dim, dim)
      processed
    (processed, skip) -> Add() -> out`;
      },
      // Attention Head
      () => {
        const d = [256, 512, 768][Math.floor(Math.random() * 3)];
        const heads = [4, 8, 12][Math.floor(Math.random() * 3)];
        return `neuron RandomAttention(dim, heads):
  in: [*, seq, dim]
  out: [*, seq, dim]
  graph:
    in ->
      LayerNorm(dim)
      MultiHeadSelfAttention(dim, heads)
      Dropout(0.1)
      out`;
      },
      // ConvNet Block
      () => {
        const c = [32, 64, 128][Math.floor(Math.random() * 3)];
        return `neuron ConvBlock(channels):
  in: [*, channels, h, w]
  out: [*, channels, h, w]
  graph:
    in ->
      Conv2d(channels, channels, kernel_size=3, padding=1)
      BatchNorm2d(channels)
      ReLU()
      out`;
      }
    ];

    const template = templates[Math.floor(Math.random() * templates.length)];
    const code = template();
    setInput(code);
    handleCompile(code);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ display: 'flex', gap: '1rem' }}>
        <button 
          onClick={() => handleCompile(input)} 
          disabled={!ready}
          className="button button--primary"
        >
          Compile
        </button>
        <button 
          onClick={generateRandom}
          disabled={!ready}
          className="button button--secondary"
        >
          Random Model
        </button>
      </div>

      <div style={{ display: 'flex', gap: '1rem', height: '500px' }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <h3>Source (NeuroScript)</h3>
          <textarea
            value={input}
            onChange={handleChange}
            style={{ 
              flex: 1, 
              fontFamily: 'monospace', 
              padding: '1rem',
              resize: 'none',
              borderRadius: '8px',
              border: '1px solid #ccc',
              fontSize: '14px',
              lineHeight: '1.5'
            }}
            spellCheck="false"
          />
        </div>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <h3>Output (PyTorch)</h3>
          {error ? (
            <div style={{ 
              flex: 1, 
              backgroundColor: '#fff0f0', 
              color: '#d00', 
              padding: '1rem', 
              fontFamily: 'monospace',
              whiteSpace: 'pre-wrap',
              overflow: 'auto',
              borderRadius: '8px',
              border: '1px solid #ffcccc'
            }}>
              {error}
            </div>
          ) : (
            <textarea
              readOnly
              value={output}
              style={{ 
                flex: 1, 
                fontFamily: 'monospace', 
                padding: '1rem',
                backgroundColor: '#1e1e1e',
                color: '#d4d4d4',
                resize: 'none',
                borderRadius: '8px',
                border: '1px solid #333'
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
