import React, { useMemo } from 'react';

const NODE_W = 140;
const NODE_H = 30;
const PAD_X = 30;
const PAD_Y = 14;
const COL_GAP = 100;

const COL_L = PAD_X;
const COL_C = PAD_X + NODE_W + COL_GAP;
const COL_R = PAD_X + (NODE_W + COL_GAP) * 2;
const TOTAL_W = NODE_W * 3 + COL_GAP * 2 + PAD_X * 2;

function truncName(name) {
  return name.length > 16 ? name.slice(0, 15) + '\u2026' : name;
}

function GraphNode({ x, y, neuron, isCenter, onClick }) {
  const fill = isCenter ? 'var(--ng-bg-hover)' : 'var(--ng-bg-card)';
  const stroke = isCenter ? 'var(--ng-node-selected)' : 'var(--ng-border)';
  const strokeWidth = isCenter ? 2 : 1;
  const dotColor = neuron.status === 'implemented'
    ? 'var(--ng-implemented)'
    : 'var(--ng-planned)';
  const textColor = isCenter
    ? 'var(--ng-text)'
    : `var(--ng-level-${neuron.level || 0})`;
  const fontWeight = isCenter ? 700 : 500;

  return (
    <g style={{ cursor: 'pointer' }} onClick={() => onClick(neuron.name)}>
      <rect
        x={x}
        y={y}
        width={NODE_W}
        height={NODE_H}
        rx={6}
        fill={fill}
        stroke={stroke}
        strokeWidth={strokeWidth}
      />
      <circle cx={x + 10} cy={y + NODE_H / 2} r={4} fill={dotColor} />
      <text
        x={x + 20}
        y={y + NODE_H / 2 + 4}
        fill={textColor}
        fontSize={11}
        fontWeight={fontWeight}
      >
        {truncName(neuron.name)}
      </text>
    </g>
  );
}

export default function DependencyGraph({ neuron, dependencies, requiredBy, onSelectNeuron }) {
  const layout = useMemo(() => {
    if (!neuron) return null;

    const leftNodes = dependencies || [];
    const rightNodes = requiredBy || [];

    if (leftNodes.length === 0 && rightNodes.length === 0) return null;

    const maxSide = Math.max(leftNodes.length, rightNodes.length, 1);
    const totalH = Math.max(
      maxSide * (NODE_H + PAD_Y) + PAD_Y + 40,
      NODE_H + PAD_Y * 2 + 40
    );

    const centerY = (totalH - NODE_H) / 2;

    function yPos(idx, count) {
      const blockH = count * (NODE_H + PAD_Y) - PAD_Y;
      const startY = (totalH - blockH) / 2;
      return startY + idx * (NODE_H + PAD_Y);
    }

    // Compute left node positions
    const leftPositions = leftNodes.map((n, i) => ({
      neuron: n,
      x: COL_L,
      y: yPos(i, leftNodes.length),
    }));

    // Compute right node positions
    const rightPositions = rightNodes.map((n, i) => ({
      neuron: n,
      x: COL_R,
      y: yPos(i, rightNodes.length),
    }));

    // Compute edges: left -> center
    const leftEdges = leftPositions.map((pos) => {
      const x1 = COL_L + NODE_W;
      const y1 = pos.y + NODE_H / 2;
      const x2 = COL_C;
      const y2 = centerY + NODE_H / 2;
      const cpx = (x1 + x2) / 2;
      return `M${x1},${y1} C${cpx},${y1} ${cpx},${y2} ${x2},${y2}`;
    });

    // Compute edges: center -> right
    const rightEdges = rightPositions.map((pos) => {
      const x1 = COL_C + NODE_W;
      const y1 = centerY + NODE_H / 2;
      const x2 = COL_R;
      const y2 = pos.y + NODE_H / 2;
      const cpx = (x1 + x2) / 2;
      return `M${x1},${y1} C${cpx},${y1} ${cpx},${y2} ${x2},${y2}`;
    });

    return {
      totalH,
      centerY,
      leftPositions,
      rightPositions,
      leftEdges,
      rightEdges,
      hasLeft: leftNodes.length > 0,
      hasRight: rightNodes.length > 0,
    };
  }, [neuron, dependencies, requiredBy]);

  if (!layout) return null;

  const {
    totalH,
    centerY,
    leftPositions,
    rightPositions,
    leftEdges,
    rightEdges,
    hasLeft,
    hasRight,
  } = layout;

  return (
    <div className="ng-graph-container">
      <h3>Dependency Graph</h3>
      <svg
        viewBox={`0 0 ${TOTAL_W} ${totalH}`}
        xmlns="http://www.w3.org/2000/svg"
        style={{ minHeight: Math.min(totalH, 500), width: '100%' }}
      >
        {/* Edges: left -> center */}
        {leftEdges.map((d, i) => (
          <path
            key={`le-${i}`}
            d={d}
            fill="none"
            stroke="var(--ng-edge-dep)"
            strokeWidth={1.5}
            opacity={0.5}
          />
        ))}

        {/* Edges: center -> right */}
        {rightEdges.map((d, i) => (
          <path
            key={`re-${i}`}
            d={d}
            fill="none"
            stroke="var(--ng-edge-depby)"
            strokeWidth={1.5}
            opacity={0.5}
          />
        ))}

        {/* Left nodes (dependencies) */}
        {leftPositions.map((pos) => (
          <GraphNode
            key={`l-${pos.neuron.name}`}
            x={pos.x}
            y={pos.y}
            neuron={pos.neuron}
            isCenter={false}
            onClick={onSelectNeuron}
          />
        ))}

        {/* Center node (selected) */}
        <GraphNode
          x={COL_C}
          y={centerY}
          neuron={neuron}
          isCenter={true}
          onClick={onSelectNeuron}
        />

        {/* Right nodes (required by) */}
        {rightPositions.map((pos) => (
          <GraphNode
            key={`r-${pos.neuron.name}`}
            x={pos.x}
            y={pos.y}
            neuron={pos.neuron}
            isCenter={false}
            onClick={onSelectNeuron}
          />
        ))}

        {/* Column labels */}
        {hasLeft && (
          <text
            x={COL_L + NODE_W / 2}
            y={16}
            fill="var(--ng-text-dim)"
            fontSize={10}
            textAnchor="middle"
            fontWeight={600}
          >
            REQUIRES
          </text>
        )}
        {hasRight && (
          <text
            x={COL_R + NODE_W / 2}
            y={16}
            fill="var(--ng-text-dim)"
            fontSize={10}
            textAnchor="middle"
            fontWeight={600}
          >
            REQUIRED BY
          </text>
        )}
      </svg>
    </div>
  );
}
