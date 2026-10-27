# Agent panel (Claude or Ollama)

The **Agent** panel is a chat with a model that works the lobby through the
same functions as the panels: you write what you want ("spawn two
microducks a metre apart and have them swap places", "kick the ball with
r1"), and it spawns, drives, waits, looks and answers, each step shown as it
goes (a tool call with its input and result; a thinking summary).

Everything runs in the page (`public/js/agent.js`): the conversation, the
tools, and the requests to the model, so it works wherever the page does
(local mode, a shared world, GitHub Pages) with no server part. The first
menu picks who answers:

- **Claude**: the Claude API, called from the page with the Anthropic
  TypeScript SDK (`public/vendor/anthropic`, bundled by `scripts/vendor.mjs`
  from `scripts/anthropic.mjs`, loaded on the first message). Paste an API
  key from [platform.claude.com](https://platform.claude.com/settings/keys)
  (billed to that account; a claude.ai Pro/Max subscription is not an API
  account); it is kept in this browser's localStorage and sent only to
  api.anthropic.com, **Forget key** removes it. There is no sign-in in the
  page: the Claude Platform's OAuth login is for Anthropic's own tools
  (`ant auth login`, Claude Code), not for a web page. Model: Claude Opus
  5.5 (default), Sonnet 5.5 or Haiku 5.5; effort low, medium (default) or
  high. Requests ask for adaptive thinking with summaries, prompt caching,
  and the API's default fallback model when a request is declined (not for
  Haiku).
- **Ollama**: a model of an [Ollama](https://ollama.com) server, by default
  `http://localhost:11434` (the machine the browser runs on), through its
  `/api/chat` with tools. The panel lists the server's models and, from
  `/api/show`, what the picked one can do: it needs **tools** (e.g. qwen3,
  llama3.1, mistral-small); with **vision** it sees the screenshots (else
  it is told there is a picture it cannot see); with **thinking** it is
  asked to think. The context is 32k tokens (`num_ctx`; the tools'
  descriptions alone are some 5k). The server must let the page's origin
  call it:

  ```sh
  OLLAMA_ORIGINS=http://localhost:4007 ollama serve   # the page's origin, or * for any
  ollama pull qwen3
  ```

  From an https page (GitHub Pages, a Jupyter proxy) to an Ollama on
  localhost, the browser may ask for access to the local network.

Switching between them starts a new chat (their histories do not mix). The
choices (not the key) are kept in the browser too.

The tools (`tools()` in `agent.js`):

| Tool | Does |
|---|---|
| `get_world_state` | every spawned thing (name, label, pose, drives, goal, controllers, policy, skills), sim time, physics, the behaviour tree, the view |
| `search_assets` | the model library by name words (and category) |
| `spawn`, `remove`, `clear_scene` | as the asset panel and Clear / ✕ (`window.lobby` in `main.js`) |
| `reset_scene`, `set_paused` | Reset, Pause |
| `set_controller` | the Teleop panel's Policy / Base / IK chips (a policy by id) |
| `set_goal`, `move`, `stop` | goal markers (`window.goals`) |
| `wait` | simulated seconds, or until robots have arrived |
| `use_skill` | a policy skill (the Microduck's kick) |
| `run_behavior_tree`, `behavior_tree_status`, `stop_behavior_tree` | a py_trees script in the Behavior panel (`window.bt`); its description carries the `lobby` module's docstring |
| `set_physics` | the Physics panel's options |
| `set_view`, `screenshot` | the main camera's orbit; a 640 px picture of the view, sent to the model as an image |

`remove`, `clear_scene` and `reset_scene` ask in a dialog first while
**Ask before removing or resetting** is on. **Stop** ends a turn (the tool
running finishes or gives up; a `wait` stops at once); **New chat** forgets
the conversation. A turn makes at most 40 requests.

The model is not a controller: each request takes seconds, so it does not
steer joints or speeds; it sets goals and skills, and for anything
coordinated or reactive it writes a behaviour tree that the page then ticks
10 times a second.
