import { beforeEach, describe, expect, it, vi } from 'vitest'

const openSessionTile = vi.fn()
const patchSessionTile = vi.fn()

vi.mock('@/store/session-states', () => ({
  openSessionTile: (...args: unknown[]) => openSessionTile(...args),
  patchSessionTile: (...args: unknown[]) => patchSessionTile(...args)
}))

import { applyVisibleSubagentEvent, subagentVisibleLabel } from './visible-subagent-sessions'

describe('visible subagent sessions', () => {
  beforeEach(() => {
    openSessionTile.mockReset()
    patchSessionTile.mockReset()
  })

  it('opens the child session as a centered tile on subagent.start', () => {
    applyVisibleSubagentEvent('subagent.start', {
      child_session_id: 'child-123',
      goal: 'scan the repo and summarize results'
    })

    expect(openSessionTile).toHaveBeenCalledWith('child-123', 'center')
    expect(patchSessionTile).toHaveBeenCalledWith('child-123', {
      titleOverride: 'scan the repo and summarize results'
    })
  })

  it('uses the same path for spawn_requested so queued workers are visible early', () => {
    applyVisibleSubagentEvent('subagent.spawn_requested', {
      child_session_id: 'child-queued',
      preview: 'queued worker goal'
    })

    expect(openSessionTile).toHaveBeenCalledWith('child-queued', 'center')
    expect(patchSessionTile).toHaveBeenCalledWith('child-queued', {
      titleOverride: 'queued worker goal'
    })
  })

  it('ignores subagent events that have no real child session id', () => {
    applyVisibleSubagentEvent('subagent.start', { goal: 'missing id' })
    applyVisibleSubagentEvent('subagent.progress', { child_session_id: 'child-123', goal: 'late progress' })

    expect(openSessionTile).not.toHaveBeenCalled()
    expect(patchSessionTile).not.toHaveBeenCalled()
  })

  it('normalizes noisy labels into one short single-line title', () => {
    expect(
      subagentVisibleLabel({ goal: '  investigate\n\n  the    flaky    queue   consumer  ' })
    ).toBe('investigate the flaky queue consumer')
  })
})
