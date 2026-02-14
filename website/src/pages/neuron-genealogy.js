import React from 'react';
import Layout from '@theme/Layout';
import NeuronGenealogy from '../components/NeuronGenealogy';

export default function NeuronGenealogyPage() {
  return (
    <Layout
      title="Neuron Genealogy"
      description="Interactive explorer for the NeuroScript neuron hierarchy — 300+ neurons with dependency graphs">
      <NeuronGenealogy />
    </Layout>
  );
}
