import { openSessionTile, patchSessionTile } from '@/store/session-states'

function normalizeVisibleSessionLabel(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined
  }

  const singleLine = value
    .replace(/\s+/g, ' ')
    .trim()

  return singleLine ? singleLine.slice(0, 96) : undefined
}

export function ensureVisibleSessionTile(storedSessionId: string, titleOverride?: string): void {
  openSessionTile(storedSessionId, 'center')

  if (titleOverride) {
    patchSessionTile(storedSessionId, { titleOverride })
  }
}

export function subagentVisibleLabel(payload: Record<string, unknown>): string | undefined {
  return normalizeVisibleSessionLabel(payload.goal ?? payload.preview ?? payload.text)
}

export function applyVisibleSubagentEvent(eventType: string, payload: Record<string, unknown>): void {
  const childSessionId = typeof payload.child_session_id === 'string' ? payload.child_session_id.trim() : ''

  if (!childSessionId || (eventType !== 'subagent.spawn_requested' && eventType !== 'subagent.start')) {
    return
  }

  ensureVisibleSessionTile(childSessionId, subagentVisibleLabel(payload))
}
