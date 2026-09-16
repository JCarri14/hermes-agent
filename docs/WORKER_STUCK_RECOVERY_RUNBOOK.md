# WORKER_STUCK_RECOVERY_RUNBOOK

Workstream: WORKER_LONG_RUNNING_LIFECYCLE_RELIABILITY_V1 (2026-09-16).
Evidencia R0: worker crashes long-running (~30-40 min, npm/build workloads)
y dispatcher "stuck" persistente; ROOT_CAUSE del stuck = cards ready
classificadas DESTRUCTIVE_LIVE por el destructive gate son saltadas por el
dispatcher SILENCIOSAMENTE (sin evento), lo que produce el warning genérico
"ready queue non-empty but 0 workers spawned" que no se cura con restarts.

## 1. Cómo detectar el patrón

Síntoma de cachete:
- journal gateway: `kanban dispatcher stuck: ready queue non-empty for N
  consecutive ticks but 0 workers spawned` durante decenas/hundreds de ticks.
- `hermes kanban list --status ready` muestra cards assignee real (perfil
  saludable) que NO se despechan pese a capacidad libre.
- Workers de una categoría (p.ej. developer) fluyen mientras otras quedan
  paradas.

Con la instrumentación v1.4 el propio warning imprime:
`gate-held ready cards: [t_xxx[assignee]: reason]` → ya no es ciego.

## 2. Qué comprobar antes de actuar (orden)

1. `journalctl --user -u hermes-gateway-admin.service --since "2 h ago"`
   → ¿stuck warning? ¿con gate-held list?
2. `sqlite3 ~/.hermes/kanban/boards/control-plane/kanban.db
   "SELECT id,title,status,assignee FROM tasks WHERE status='ready'"` → cards en cola.
3. Evaluar el gate de cada ready card (runtime):
   `cd /tmp && <venv>/python -c "import sys; sys.path.insert(0,'<runtime>');
   import hermes_cli.destructive_gate as g; import sqlite3; ..."` → cls por card.
4. Perfil del assignee: `hermes --profile X auth status openai-codex` /
   token relogin_required (lección 09-14: firmas muertas matan spawn).
5. `ps aux | grep "work kanban"` → workers vivos reales.
6. Runs zombi: `SELECT task_id,status,datetime(started_at,'unixepoch','localtime')
   FROM task_runs WHERE status='running' AND ended_at IS NULL` → normalmente
   irrelevante (el cap cuenta tasks, no runs), pero documenta leche.

## 3. Distinguir

| Caso | Señal | Acción |
|---|---|---|
| Worker legítimamente largo | heartbeats recientes, proceso vivo en ps, sin reaper event | NO tocar. Liveness real, no elapsed. |
| Worker muerto | "pid not alive" / "exited cleanly rc=0" (protocol_violation) en task_runs.error | Reclaim vía retry con checkpoints (abajo). |
| Stale claim | claim_expires pasado + sin pid | `kanban reclaim <id>` si la card sigue running; retry. |
| Stuck dispatcher (gate) | warning con `gate-held ready cards: [...]` | Si el gate está DOBLADO (research/design card research-only DESCRIPTIVA → DESTRUCTIVE_LIVE): es un false positive del classifier → extender intent-context (v1.4+), NO firmar GO. |
| Stuck dispatcher (env) | warning SIN gate-held + perfiles muertos (auth) | re-auth / restart. |

## 4. Reclaim seguro

- `hermes kanban reclaim <task_id>` solo para claims vivos/stale.
- NUNCA matar un worker por duración: liveness = proceso + heartbeat + lease.
- Tras reclaim de una card crashed: retry con instrucción "valida el trabajo
  parcial existente en el worktree y completa" (los workers crasheados suelen
  dejar work durable; audit/IMPL/PA-FIX recuperados así).

## 5. Retry desde checkpoint (patrón probado 09-14→16)

1. kanban_comment con instrucciones: "PRIMERO git status/diff del worktree,
   validar lo existente, completar lo que falte, ejecutar gates, kanban_complete
   con evidencia; si el worker no termina, checkpoint comment + kanban_block
   review-required".
2. kanban_unblock → respawn.
3. Si vuelve a morir >25 min: git status local del worktree para recuperar
   artifacts antes de siguientes reintentos.

## 6. Cuándo restart del gateway es apropiado

ÚLTIMO RECURSO, con evidencia y cooldown:
- Apropiado: stuck env persistente tras diagnóstico (auth/perfil/spawn_fn) o
  tras bulk-crash sin auto-recovery; 0 workers running (no matar trabajo vivo;
  si hay workers, esperar o aceptar checkpoint-retry).
- NO como primera reacción: primero R0 (gate-held / auth / locks).
- Tras restart: verificar `is-active`, nuevo MainPID, y que el primer tick
  claim cards ready normales (sanidad de spawn).

## 7. Verificar recuperación

- `journalctl` sin "stuck" creciente; cards ready claimadas (running) sin GO.
- workers: `ps` con heartbeats; card completa con summary.
- En el caso gate (v1.4): DCA-R1/PB-MODEL-class payloads → SAFE → el dispatcher
  las claima SOLO (la corrección del clasificador ES el self-recovery; sin
  restart tras cutover).

## 8. Evidencia a conservar para postmortem

- task_runs (outcome, error, timestamps), task_events (spawned/heartbeat/
  crashed/protocol_violation), journal gateway, worktree git status/diff,
  md5 del módulo cargado, pid gateway, memoria/free en el momento.