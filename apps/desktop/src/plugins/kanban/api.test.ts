import { beforeEach, describe, expect, it, vi } from 'vitest'

const closeSessionTile = vi.fn()
const openSessionTile = vi.fn()
const patchSessionTile = vi.fn()

vi.mock('@/store/session-states', () => ({
  closeSessionTile: (...args: unknown[]) => closeSessionTile(...args),
  openSessionTile: (...args: unknown[]) => openSessionTile(...args),
  patchSessionTile: (...args: unknown[]) => patchSessionTile(...args)
}))

import { applyVisibleSessionEventEffects } from './api'

describe('kanban visible worker session events', () => {
  beforeEach(() => {
    closeSessionTile.mockReset()
    openSessionTile.mockReset()
    patchSessionTile.mockReset()
  })

  it('opens a centered visible worker tile from the existing spawned event payload', () => {
    applyVisibleSessionEventEffects([
      {
        kind: 'spawned',
        payload: {
          worker_launch: {
            display_label: 'ERP-287 · material reallocation',
            pid: 4321,
            session_id: 'kanban_t_abc_run_7'
          }
        },
        task_id: 't_abc'
      }
    ])

    expect(openSessionTile).toHaveBeenCalledWith('kanban_t_abc_run_7', 'center')
    expect(patchSessionTile).toHaveBeenCalledWith('kanban_t_abc_run_7', {
      titleOverride: 'ERP-287 · material reallocation'
    })
    expect(closeSessionTile).not.toHaveBeenCalled()
  })

  it('closes only the exact session id carried by cleanup-ready completed events', () => {
    applyVisibleSessionEventEffects([
      {
        kind: 'completed',
        payload: {
          visible_session_cleanup: {
            endpoint_id: 'not-used-for-ui-close',
            session_id: 'kanban_t_abc_run_7'
          }
        },
        task_id: 't_abc'
      }
    ])

    expect(closeSessionTile).toHaveBeenCalledWith('kanban_t_abc_run_7')
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  it('refuses to open or close when the payload lacks exact session identity', () => {
    applyVisibleSessionEventEffects([
      { kind: 'spawned', payload: { worker_launch: { pid: 1 } }, task_id: 't_missing' },
      { kind: 'completed', payload: { visible_session_cleanup: { endpoint_id: 'only-endpoint' } }, task_id: 't_missing' }
    ])

    expect(openSessionTile).not.toHaveBeenCalled()
    expect(patchSessionTile).not.toHaveBeenCalled()
    expect(closeSessionTile).not.toHaveBeenCalled()
  })
})
