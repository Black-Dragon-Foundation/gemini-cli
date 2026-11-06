/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * The `_drgnx/panel` extension (drgnx's editor panel), version 1.
 *
 * The agent advertises `agentCapabilities._meta["_drgnx/panel"] = {version, features}`
 * to a client that sent `clientCapabilities._meta["_drgnx/panel"] = {version: 1}`,
 * and only with what Gemini CLI's internals can serve:
 * - `before-tool`: a `_drgnx/before_tool` request before each tool runs; a deny,
 *   a failed request or an unreadable answer denies the tool, an allow only continues.
 * - `context`: a `_drgnx/context` notification after each turn. `used` is the model's
 *   own prompt count; the parts are estimates from the history (characters / 4).
 * - `drop`: `_drgnx/context/drop` removes one whole exchange (a prompt, its replies and
 *   tool rounds) from the history.
 * - `compact-preview`: `_drgnx/compact/preview` runs the compaction's summary without
 *   applying it. It costs model calls.
 *
 * Not advertised: `pin` (a compaction has no keep-list).
 */

import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import type { Content, Part } from '@google/genai';
import type { HistoryTurn } from '@google/gemini-cli-core';

export const DRGNX_PANEL_KEY = '_drgnx/panel';
export const DRGNX_CONTEXT_METHOD = '_drgnx/context';
export const DRGNX_BEFORE_TOOL_METHOD = '_drgnx/before_tool';
export const DRGNX_DROP_METHOD = '_drgnx/context/drop';
export const DRGNX_COMPACT_PREVIEW_METHOD = '_drgnx/compact/preview';

const VERSION = 1;
const FEATURES = ['before-tool', 'context', 'drop', 'compact-preview'];
const MAX_ITEMS = 200;
const MAX_LABEL = 256;
const ITEM_PREFIX = 'turn:';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

/** Whether the `--admin-policy` files loaded, and how many rules they hold. */
export interface AdminPolicyStatus {
  loaded: boolean;
  rules: number;
}

/**
 * The names of the policy files that `--admin-policy` PATHS point at: a file is itself, a
 * directory is its `.toml` files. A path that cannot be read names nothing.
 */
export async function adminPolicyFileNames(
  paths: readonly string[] | undefined,
): Promise<string[]> {
  const names: string[] = [];
  for (const p of paths ?? []) {
    try {
      if ((await fs.stat(p)).isDirectory()) {
        names.push(...(await fs.readdir(p)).filter((f) => f.endsWith('.toml')));
      } else {
        names.push(path.basename(p));
      }
    } catch {
      // unreadable: no rules will come from it
    }
  }
  return names;
}

/**
 * Gemini gives no signal that an admin policy loaded, so it is read from the engine: the
 * rules whose source is `Admin: <file>` for one of NAMES. A file that failed to parse, or
 * that Gemini ignored (a system policy exists), contributes no rule, so `loaded` is false.
 */
export function adminPolicyStatus(
  names: readonly string[],
  rules: ReadonlyArray<{ source?: string }>,
): AdminPolicyStatus {
  const wanted = new Set(names.map((n) => `Admin: ${n}`));
  const count = rules.filter((r) => r.source && wanted.has(r.source)).length;
  return { loaded: count > 0, rules: count };
}

/**
 * What the agent advertises, or nothing when the client did not send `_drgnx/panel` version 1.
 * ADMINPOLICY, when given, is reported beside the features.
 */
export function panelCapability(
  clientCapabilities: { _meta?: unknown } | null | undefined,
  adminPolicy?: AdminPolicyStatus,
):
  | { version: number; features: string[]; adminPolicy?: AdminPolicyStatus }
  | undefined {
  const meta = clientCapabilities?._meta;
  const offered = isRecord(meta) ? meta[DRGNX_PANEL_KEY] : undefined;
  if (!isRecord(offered) || offered['version'] !== VERSION) return undefined;
  return {
    version: VERSION,
    features: [...FEATURES],
    ...(adminPolicy ? { adminPolicy } : {}),
  };
}

/** A token estimate for text (characters / 4); the breakdown labels it an estimate. */
export function estimateTokens(value: unknown): number {
  const text = typeof value === 'string' ? value : JSON.stringify(value);
  return Math.ceil((text?.length ?? 0) / 4);
}

type Kind = 'system' | 'tools' | 'files' | 'tool-output' | 'conversation';

export interface ContextItem {
  id: string;
  label: string;
  tokens: number;
}

export interface DrgnxContext {
  used: number;
  size: number;
  parts: Array<{ kind: Kind; tokens: number; items?: ContextItem[] }>;
}

function isToolPart(part: Part): boolean {
  return !!part.functionResponse;
}

function startsExchange(content: Content): boolean {
  return (
    content.role === 'user' &&
    !!content.parts?.some((p) => typeof p.text === 'string' && p.text !== '') &&
    !content.parts.some(isToolPart)
  );
}

function labelOf(content: Content): string {
  const text = (content.parts ?? []).map((p) => p.text ?? '').join(' ');
  const line = text.replace(/\s+/g, ' ').trim();
  return (line || 'prompt').slice(0, MAX_LABEL);
}

export interface Exchange {
  id: string;
  label: string;
  turns: HistoryTurn[];
}

/**
 * The history cut into exchanges: each begins at a user turn with text and holds everything up to
 * the next one, so a tool call is never separated from its response. The id is the first
 * turn's durable id, which stays the same when other exchanges are dropped.
 */
export function exchanges(turns: readonly HistoryTurn[]): Exchange[] {
  const out: Exchange[] = [];
  for (const turn of turns) {
    const last = out[out.length - 1];
    if (!last || startsExchange(turn.content)) {
      out.push({
        id: `${ITEM_PREFIX}${turn.id}`,
        label: labelOf(turn.content),
        turns: [turn],
      });
    } else {
      last.turns.push(turn);
    }
  }
  return out;
}

function toolOutputTokens(turn: HistoryTurn): number {
  const responses = (turn.content.parts ?? [])
    .filter(isToolPart)
    .map((p) => p.functionResponse);
  return responses.length === 0 ? 0 : estimateTokens(responses);
}

/**
 * The breakdown for `_drgnx/context`. `used` is the model's own count of the last prompt (the
 * estimate of everything when the model has not answered yet); the parts are estimates. The
 * conversation part lists the exchanges (the newest 200) as droppable items.
 */
export function buildContext(args: {
  turns: readonly HistoryTurn[];
  systemInstruction: string;
  tools: unknown;
  lastPromptTokens: number;
  size: number;
}): DrgnxContext {
  const system = estimateTokens(args.systemInstruction);
  const tools =
    Array.isArray(args.tools) && args.tools.length === 0
      ? 0
      : estimateTokens(args.tools ?? []);
  let toolOutput = 0;
  const all = exchanges(args.turns).map((ex) => {
    const total = estimateTokens(
      ex.turns.flatMap((t) => t.content.parts ?? []),
    );
    const tool = ex.turns.reduce((n, t) => n + toolOutputTokens(t), 0);
    toolOutput += tool;
    return { ex, tokens: Math.max(total - tool, 0) };
  });
  const conversation = all.reduce((n, e) => n + e.tokens, 0);
  const estimated = system + tools + toolOutput + conversation;
  const used = args.lastPromptTokens > 0 ? args.lastPromptTokens : estimated;
  const items = all.slice(-MAX_ITEMS).map(({ ex, tokens }) => ({
    id: ex.id,
    label: ex.label,
    tokens,
  }));
  const parts: DrgnxContext['parts'] = [];
  const add = (kind: Kind, tokens: number, list?: ContextItem[]) => {
    if (tokens > 0)
      parts.push({ kind, tokens, ...(list ? { items: list } : {}) });
  };
  add('system', system);
  add('tools', tools);
  add('tool-output', toolOutput);
  add('conversation', conversation, items.length > 0 ? items : undefined);
  return { used, size: args.size, parts };
}

/** The history without the exchange ITEM, or undefined when no exchange has that id. */
export function dropExchange(
  turns: readonly HistoryTurn[],
  item: unknown,
): HistoryTurn[] | undefined {
  if (typeof item !== 'string') return undefined;
  const all = exchanges(turns);
  const at = all.findIndex((e) => e.id === item);
  if (at < 0) return undefined;
  return all.flatMap((e, i) => (i === at ? [] : e.turns));
}

export type BeforeToolAnswer =
  | { allow: true }
  | { allow: false; reason: string };

/** The `_drgnx/before_tool` answer, failing closed: only an exact allow allows. */
export function readBeforeToolAnswer(answer: unknown): BeforeToolAnswer {
  if (isRecord(answer) && answer['outcome'] === 'allow') return { allow: true };
  if (isRecord(answer) && answer['outcome'] === 'deny') {
    const reason = answer['reason'];
    return {
      allow: false,
      reason:
        typeof reason === 'string' && reason !== ''
          ? reason
          : 'drgnx refused this tool call',
    };
  }
  return {
    allow: false,
    reason: 'drgnx gave no valid answer to _drgnx/before_tool',
  };
}
