import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import NeuroPlayground from '../components/NeuroPlayground';

export default function Playground() {
  return (
    <Layout
      title="Playground"
      description="Compile NeuroScript to PyTorch in your browser">
      <main>
        <div className="container margin-vert--lg">
          <h1>NeuroScript Playground</h1>
          <p>Write NeuroScript code on the left, see compiled PyTorch code on the right.</p>
          <NeuroPlayground />
        </div>
      </main>
    </Layout>
  );
}
