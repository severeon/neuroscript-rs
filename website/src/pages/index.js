import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';

const GPT2_CODE = `# GPT-2 Small: 12 transformer blocks
neuron GPT2Small(vocab=50257, d_model=768, heads=12, d_ff=3072, layers=12):
    in: [*, seq]
    out: [*, seq, vocab]
    context:
        embed = Embedding(vocab, d_model)
        blocks = unroll(layers):
            block = TransformerBlock(d_model, heads, d_ff)
        ln_f = LayerNorm(d_model)
        head = Linear(d_model, vocab)
    graph:
        in ->
            embed
            blocks
            ln_f
            head
            out`;

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary')} style={{textAlign: 'center', padding: '3rem 0 2rem'}}>
      <div className="container">
        <h1 className="hero__title" style={{fontSize: '3.5rem', fontWeight: 'bold'}}>{siteConfig.title}</h1>
        <p className="hero__subtitle" style={{fontSize: '1.25rem', marginBottom: '1.5rem'}}>
          {siteConfig.tagline}
        </p>
        <div style={{display: 'flex', gap: '0.75rem', justifyContent: 'center', flexWrap: 'wrap'}}>
          <Link className="button button--secondary button--lg" to="/docs/intro">
            Get Started
          </Link>
          <Link className="button button--secondary button--lg" to="/playground">
            Playground
          </Link>
        </div>
      </div>
    </header>
  );
}

function GPT2Demo() {
  return (
    <section style={{padding: '2.5rem 0 3rem'}} aria-label="Interactive GPT-2 demo">
      <div className="container">
        <div style={{textAlign: 'center', marginBottom: '1.5rem'}}>
          <h2 style={{fontSize: '1.75rem', marginBottom: '0.5rem'}}>
            Define GPT-2 in 17 lines. Compile to PyTorch.
          </h2>
          <p style={{
            fontSize: '1rem',
            color: 'var(--ifm-color-emphasis-700)',
            maxWidth: '640px',
            margin: '0 auto',
          }}>
            NeuroScript compiles neural architecture definitions into production-ready
            PyTorch modules. Backend: Rust + Python 3.13.
          </p>
        </div>

        <BrowserOnly fallback={
          <div style={{minHeight: '500px', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
            <p style={{color: 'var(--ifm-color-emphasis-600)'}}>Loading interactive demo...</p>
          </div>
        }>
          {() => {
            const NeuroEditor = require('@site/src/components/NeuroEditor').default;
            return (
              <NeuroEditor
                mode="tutorial"
                layout="horizontal"
                initialCode={GPT2_CODE}
                showAnalysis={false}
                showCompileButton={false}
                height="500px"
                responsive
              />
            );
          }}
        </BrowserOnly>

        <div style={{textAlign: 'center', marginTop: '1.5rem'}}>
          <Link
            className="button button--outline button--primary"
            to="/playground"
            style={{fontSize: '0.95rem'}}
          >
            Try more examples in the Playground
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout
      title="Home"
      description="Neural architecture composition language">
      <HomepageHeader />
      <main>
        <GPT2Demo />
      </main>
    </Layout>
  );
}
