'use client'

import type { ServingInfo } from '@/lib/hooks'

function pathsDiffer(a?: string, b?: string): boolean {
  if (!a || !b) return false
  const norm = (p: string) => p.replace(/\/+$/, '')
  return norm(a) !== norm(b)
}

export default function EmptyRunsState({ serving }: { serving: ServingInfo | null }) {
  const wrongDir = pathsDiffer(serving?.cwd, serving?.project_root)

  return (
    <div className="px-6 py-16 text-center" style={{ color: 'var(--text-muted)' }}>
      <div
        className="mx-auto mb-5 w-12 h-12 rounded-xl flex items-center justify-center"
        style={{ background: 'var(--bg-elevated, var(--muted))' }}
      >
        <svg width="20" height="20" viewBox="0 0 18 18" fill="none" aria-hidden="true">
          <path
            d="M9 1.5L16.5 5.5V12.5L9 16.5L1.5 12.5V5.5L9 1.5Z"
            stroke="currentColor"
            strokeWidth="1.2"
            fill="none"
          />
          <circle cx="9" cy="9" r="2" stroke="currentColor" strokeWidth="1" />
        </svg>
      </div>
      <div className="text-sm font-medium" style={{ color: 'var(--text-secondary, var(--foreground))' }}>
        {wrongDir ? 'No runs in the directory this UI is serving' : 'No runs yet'}
      </div>
      <p className="mx-auto mt-2 max-w-md text-[13px] leading-relaxed" style={{ color: 'var(--text-muted)' }}>
        ARGUS writes runs under the project root (git / pyproject.toml / $ARGUS_DIR),
        not necessarily the folder you launched <code className="font-mono">argus ui</code> from.
      </p>
      {serving?.runs_dir && (
        <p className="mt-3 font-mono text-[12px] break-all" style={{ color: 'var(--text-secondary, var(--foreground))' }}>
          looking in {serving.runs_dir}
        </p>
      )}
      {wrongDir && serving?.cwd && serving?.project_root && (
        <p className="mt-1 font-mono text-[11px] break-all" style={{ color: 'var(--text-muted)' }}>
          started from {serving.cwd} · project root {serving.project_root}
        </p>
      )}
      <ul className="mx-auto mt-6 max-w-sm space-y-2 text-left text-[13px] leading-relaxed" style={{ color: 'var(--text-secondary, var(--foreground))' }}>
        <li>1. Run the graph with ARGUS attached (same project).</li>
        <li>
          2. Confirm the run exists:{' '}
          <code className="font-mono">argus show last</code>
        </li>
        <li>
          3. If that works but this table is empty, start{' '}
          <code className="font-mono">argus ui</code> from the project root — not a nested cwd.
        </li>
      </ul>
    </div>
  )
}
