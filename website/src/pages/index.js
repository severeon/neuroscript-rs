import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary')} style={{textAlign: 'center', padding: '4rem 0'}}>
      <div className="container">
        <h1 className="hero__title" style={{fontSize: '4rem', fontWeight: 'bold'}}>{siteConfig.title}</h1>
        <p className="hero__subtitle" style={{fontSize: '1.5rem'}}>{siteConfig.tagline}</p>
        <div style={{marginTop: '2rem'}}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Get Started
          </Link>
          <span style={{margin: '0 10px'}}></span>
          <Link
            className="button button--secondary button--lg"
            to="/playground">
            Playground
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Home`}
      description="Neural architecture composition language">
      <HomepageHeader />
      <main>
        <div className="container">
          <section style={{padding: '4rem 0', textAlign: 'center'}}>
            <h2>Neural architecture composition language.</h2>
            <p style={{fontSize: '1.2rem'}}>Neurons all the way down.</p>
          </section>
        </div>
      </main>
    </Layout>
  );
}
