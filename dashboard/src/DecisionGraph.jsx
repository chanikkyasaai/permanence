import React from 'react';

const NODE_LAYOUT = [
  { id: 'draft_internal_memo', label: 'Draft Internal Memo', x: 80, y: 70, tier: 1 },
  { id: 'schedule_conversation', label: 'Schedule Conversation', x: 80, y: 190, tier: 1 },
  { id: 'review_contract_internally', label: 'Review Contract Internally', x: 80, y: 310, tier: 1 },
  { id: 'begin_internal_investigation', label: 'Begin Internal Investigation', x: 80, y: 430, tier: 1 },
  { id: 'send_internal_communication', label: 'Send Internal Communication', x: 350, y: 110, tier: 2 },
  { id: 'reassign_project_lead', label: 'Reassign Project Lead', x: 350, y: 230, tier: 2 },
  { id: 'prepare_response_draft', label: 'Prepare Response Draft', x: 350, y: 350, tier: 2 },
  { id: 'align_with_legal', label: 'Align With Legal', x: 350, y: 470, tier: 2 },
  { id: 'send_external_communication', label: 'Send External Communication', x: 620, y: 140, tier: 3 },
  { id: 'approve_staged_rollout', label: 'Approve Staged Rollout', x: 620, y: 260, tier: 3 },
  { id: 'delay_release', label: 'Delay Release', x: 620, y: 380, tier: 3 },
  { id: 'issue_public_statement', label: 'Issue Public Statement', x: 620, y: 500, tier: 4 },
  { id: 'communicate_resolution_externally', label: 'Communicate Resolution Externally', x: 900, y: 220, tier: 4 },
  { id: 'approve_full_launch', label: 'Approve Full Launch', x: 900, y: 340, tier: 4 },
  { id: 'initiate_hr_formal_process', label: 'Initiate HR Process', x: 900, y: 460, tier: 5 },
  { id: 'update_contract_system', label: 'Update Contract System', x: 1180, y: 210, tier: 5 },
  { id: 'update_internal_records', label: 'Update Internal Records', x: 1180, y: 330, tier: 5 },
  { id: 'schedule_client_follow_up', label: 'Schedule Client Follow-Up', x: 1180, y: 450, tier: 5 },
];

const EDGES = [
  ['draft_internal_memo', 'send_internal_communication'],
  ['schedule_conversation', 'reassign_project_lead'],
  ['review_contract_internally', 'align_with_legal'],
  ['begin_internal_investigation', 'prepare_response_draft'],
  ['send_internal_communication', 'send_external_communication'],
  ['reassign_project_lead', 'approve_staged_rollout'],
  ['prepare_response_draft', 'issue_public_statement'],
  ['align_with_legal', 'communicate_resolution_externally'],
  ['send_external_communication', 'issue_public_statement'],
  ['approve_staged_rollout', 'approve_full_launch'],
  ['issue_public_statement', 'communicate_resolution_externally'],
  ['communicate_resolution_externally', 'update_contract_system'],
  ['communicate_resolution_externally', 'update_internal_records'],
  ['communicate_resolution_externally', 'schedule_client_follow_up'],
];

function buildNodeMap(lockedActions = {}) {
  const lockedKeys = Array.isArray(lockedActions)
    ? Object.fromEntries(lockedActions.map((actionId) => [actionId, 'Locked by prior irreversible action']))
    : lockedActions && typeof lockedActions === 'object'
      ? lockedActions
      : {};
  const lockLookup = new Set(Object.keys(lockedKeys));
  return NODE_LAYOUT.map((node) => {
    const locked = lockLookup.has(node.id);
    return {
      ...node,
      locked,
      reason: locked ? lockedKeys[node.id] : '',
    };
  });
}

function edgePath(source, target) {
  const startX = source.x + 190;
  const startY = source.y + 28;
  const endX = target.x;
  const endY = target.y + 28;
  const c1X = startX + 90;
  const c1Y = startY;
  const c2X = endX - 90;
  const c2Y = endY;
  return `M ${startX} ${startY} C ${c1X} ${c1Y}, ${c2X} ${c2Y}, ${endX} ${endY}`;
}

export default function DecisionGraph({ lockedActions = {}, recentActions = [] }) {
  const nodes = buildNodeMap(lockedActions);
  const byId = new Map(nodes.map((node) => [node.id, node]));

  return (
    <div className="decision-graph-card">
      <div className="card-header">
        <div>
          <h2>Decision Tree</h2>
          <p>Locked actions turn dark red with causal provenance.</p>
        </div>
      </div>

      <svg className="decision-graph-svg" viewBox="0 0 1450 620" role="img" aria-label="Decision tree of the action space">
        <defs>
          <linearGradient id="nodeGlow" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#2a3145" />
            <stop offset="100%" stopColor="#111827" />
          </linearGradient>
          <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="10" stdDeviation="18" floodColor="#000" floodOpacity="0.45" />
          </filter>
        </defs>

        {EDGES.map(([sourceId, targetId]) => {
          const source = byId.get(sourceId);
          const target = byId.get(targetId);
          if (!source || !target) {
            return null;
          }
          return (
            <path
              key={`${sourceId}-${targetId}`}
              d={edgePath(source, target)}
              stroke="rgba(110, 118, 140, 0.35)"
              strokeWidth="2"
              fill="none"
              strokeDasharray="8 8"
            />
          );
        })}

        {nodes.map((node) => {
          const color = node.locked ? '#4a0f16' : node.tier === 1 ? '#1b2336' : node.tier === 2 ? '#172033' : node.tier === 3 ? '#1d2c44' : node.tier === 4 ? '#27324c' : '#31415c';
          const stroke = node.locked ? '#8b1d2d' : 'rgba(128, 146, 184, 0.36)';
          const textDecoration = node.locked ? 'line-through' : 'none';
          const labelColor = node.locked ? '#ffd4db' : '#ecf2ff';

          return (
            <g key={node.id} transform={`translate(${node.x}, ${node.y})`} filter="url(#shadow)">
              <rect
                width="190"
                height="56"
                rx="16"
                fill={color}
                stroke={stroke}
                strokeWidth="1.5"
              />
              <rect
                x="0"
                y="0"
                width="190"
                height="56"
                rx="16"
                fill="url(#nodeGlow)"
                opacity="0.3"
              />
              <text
                x="95"
                y="27"
                fill={labelColor}
                textAnchor="middle"
                fontSize="13"
                fontWeight="700"
                style={{ textDecoration, letterSpacing: '0.02em' }}
              >
                {node.label}
              </text>
              {node.locked ? (
                <text x="95" y="43" fill="#ff8fa0" textAnchor="middle" fontSize="9">
                  {node.reason}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>

      <div className="tree-footer">
        <div><span className="legend-dot unlocked" /> Available</div>
        <div><span className="legend-dot locked" /> Locked</div>
        <div>{recentActions.length} recent action events loaded</div>
      </div>
    </div>
  );
}
