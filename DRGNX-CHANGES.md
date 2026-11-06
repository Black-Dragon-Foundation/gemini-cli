# drgnx changes

What the drgnx fork adds on top of the upstream release it is based on. Upstream's own files stay as upstream
released them.

## Features

- `_drgnx/panel` extension (version 1) for drgnx's editor panel, advertised in
  `agentCapabilities._meta["_drgnx/panel"]` only to a client that sent
  `clientCapabilities._meta["_drgnx/panel"] = {version: 1}`. Features:
  - `before-tool`: a `_drgnx/before_tool` request after the tool is built and
    before its confirmation; a deny, a failed request or an unreadable answer
    ends the tool with an error, an allow only continues and grants nothing. A
    subagent's tool calls are asked too, where Gemini checks them (the
    policy-check request), with the same rules.
  - `context`: a `_drgnx/context` notification after each turn; `used` is the
    model's own prompt count, the parts are character-based estimates, and the
    conversation part lists the exchanges as droppable items.
  - `drop`: `_drgnx/context/drop` removes one whole exchange (a prompt, its
    replies and tool rounds) from the history; refused while a turn is running.
  - `compact-preview`: `_drgnx/compact/preview` runs a forced compression and
    reports what it would free without applying it. It costs model calls (a
    summary and a check). Adds `GeminiClient.previewCompression`.
  - The advertised `_meta["_drgnx/panel"]` also carries
    `adminPolicy: {loaded, rules}`: whether the `--admin-policy` file gave rules
    to Gemini's policy engine, and how many (Gemini has no other signal).
  - `pin` is not advertised: a compression has no keep-list.
