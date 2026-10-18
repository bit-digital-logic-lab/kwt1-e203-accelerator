import { randomUUID } from "node:crypto"
import { mkdir, rename, writeFile } from "node:fs/promises"
import { join } from "node:path"

// OpenCode 2 loads this project plugin for sessions in the worktree.
export default {
  id: "quarter-hour-check",
  async setup(ctx) {
    const root = ctx.location.project?.canonical || ctx.location.project?.directory || ctx.location.directory
    const checkins = join(root, ".ai-checks", "checkins")
    const controller = new AbortController()
    const observedSessions = new Set()
    const record = async (id, state, note) => {
      if (!id) return
      const safeId = id.replace(/[^A-Za-z0-9_.-]/g, "_")
      const path = join(checkins, `opencode-${safeId}.json`)
      const temporary = `${path}.${randomUUID()}.tmp`
      await mkdir(checkins, { recursive: true })
      await writeFile(temporary, `${JSON.stringify({
        id: `opencode-${safeId}`,
        state,
        updated_at: new Date().toISOString(),
        source: "opencode-hook",
        note,
        result: "",
        service: "",
        pid: null,
      }, null, 2)}\n`)
      await rename(temporary, path)
    }
    const contextHook = await ctx.session.hook("context", async (event) => {
      observedSessions.add(event.sessionID)
      await record(event.sessionID, "busy", `agent=${event.agent || "unknown"}`)
    })
    void (async () => {
      for await (const event of ctx.event.subscribe({ signal: controller.signal })) {
        if (!["session.created", "session.status", "session.idle", "session.error"].includes(event.type)) continue
        const state = event.type === "session.error" ? "failed"
          : event.type === "session.idle" ? "idle"
            : event.type === "session.status" ? event.properties.status?.type || "running"
              : "running"
        await record(event.properties?.sessionID, state, event.type)
      }
    })().catch((error) => {
      if (!controller.signal.aborted) console.error("quarter-hour-check event hook failed", error)
    })
    return async () => {
      controller.abort()
      await contextHook.dispose()
      for (const id of observedSessions) await record(id, "idle", "plugin stopped")
    }
  },
}
