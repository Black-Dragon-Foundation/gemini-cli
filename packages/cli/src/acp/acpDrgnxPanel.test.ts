/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  describe,
  it,
  expect,
  vi,
  beforeEach,
  afterEach,
  type Mock,
} from 'vitest';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import type * as acp from '@agentclientprotocol/sdk';
import {
  GeminiEventType,
  CompressionStatus,
  MessageBusType,
  PolicyDecision,
  type Config,
  type GeminiChat,
  type HistoryTurn,
  type ServerGeminiStreamEvent,
} from '@google/gemini-cli-core';
import { FinishReason } from '@google/genai';
import type { LoadedSettings } from '../config/settings.js';
import type { CliArgs } from '../config/config.js';
import { GeminiAgent } from './acpRpcDispatcher.js';
import { AcpSessionManager } from './acpSessionManager.js';
import { Session } from './acpSession.js';
import {
  buildContext,
  dropExchange,
  exchanges,
  panelCapability,
  readBeforeToolAnswer,
} from './acpDrgnxPanel.js';

vi.mock(
  '@google/gemini-cli-core',
  async (
    importOriginal: () => Promise<typeof import('@google/gemini-cli-core')>,
  ) => ({
    ...(await importOriginal()),
    logToolCall: vi.fn(),
  }),
);

const user = (id: string, text: string): HistoryTurn => ({
  id,
  content: { role: 'user', parts: [{ text }] },
});
const model = (id: string, text: string): HistoryTurn => ({
  id,
  content: { role: 'model', parts: [{ text }] },
});
const call = (id: string): HistoryTurn => ({
  id,
  content: {
    role: 'model',
    parts: [{ functionCall: { id: `c-${id}`, name: 'read_file', args: {} } }],
  },
});
const result = (id: string, body: string): HistoryTurn => ({
  id,
  content: {
    role: 'user',
    parts: [
      {
        functionResponse: {
          id: `c-${id}`,
          name: 'read_file',
          response: { output: body },
        },
      },
    ],
  },
});

const HISTORY: HistoryTurn[] = [
  user('a', 'first question'),
  model('b', 'first answer'),
  user('c', 'second question'),
  call('d'),
  result('e', 'x'.repeat(400)),
  model('f', 'second answer'),
];

describe('_drgnx/panel: capability', () => {
  it('is offered only to a client that sent version 1', () => {
    expect(panelCapability(undefined)).toBeUndefined();
    expect(panelCapability({})).toBeUndefined();
    expect(
      panelCapability({ _meta: { '_drgnx/panel': { version: 2 } } }),
    ).toBeUndefined();
    expect(
      panelCapability({ _meta: { '_drgnx/panel': { version: '1' } } }),
    ).toBeUndefined();
    expect(
      panelCapability({ _meta: { '_drgnx/panel': { version: 1 } } }),
    ).toEqual({
      version: 1,
      features: ['before-tool', 'context', 'drop', 'compact-preview'],
    });
  });
});

describe('_drgnx/panel: breakdown', () => {
  it('cuts the history into exchanges that keep a tool call with its response', () => {
    const list = exchanges(HISTORY);
    expect(list.map((e) => e.id)).toEqual(['turn:a', 'turn:c']);
    expect(list[1].turns.map((t) => t.id)).toEqual(['c', 'd', 'e', 'f']);
    expect(list[1].label).toBe('second question');
  });

  it('reports the model count as used and estimates the parts', () => {
    const ctx = buildContext({
      turns: HISTORY,
      systemInstruction: 's'.repeat(80),
      tools: [],
      lastPromptTokens: 777,
      size: 1000,
    });
    expect(ctx.used).toBe(777);
    expect(ctx.size).toBe(1000);
    const byKind = Object.fromEntries(ctx.parts.map((p) => [p.kind, p]));
    expect(byKind['system'].tokens).toBe(20);
    expect(byKind['tool-output'].tokens).toBeGreaterThan(100);
    expect(byKind['conversation'].items?.map((i) => i.id)).toEqual([
      'turn:a',
      'turn:c',
    ]);
    expect(byKind['tools']).toBeUndefined();
  });

  it('falls back to the estimate before the model has answered', () => {
    const ctx = buildContext({
      turns: [user('a', 'hello world!')],
      systemInstruction: '',
      tools: [],
      lastPromptTokens: 0,
      size: 100,
    });
    expect(ctx.used).toBeGreaterThan(0);
    expect(ctx.used).toBe(ctx.parts.reduce((n, p) => n + p.tokens, 0));
  });

  it('lists at most the newest 200 exchanges', () => {
    const turns = Array.from({ length: 250 }, (_, i) => user(`u${i}`, `q${i}`));
    const ctx = buildContext({
      turns,
      systemInstruction: '',
      tools: [],
      lastPromptTokens: 1,
      size: 1,
    });
    const items = ctx.parts[0].items ?? [];
    expect(items).toHaveLength(200);
    expect(items[0].id).toBe('turn:u50');
  });
});

describe('_drgnx/panel: drop', () => {
  it('removes the whole exchange and keeps the ids of the rest', () => {
    const kept = dropExchange(HISTORY, 'turn:c');
    expect(kept?.map((t) => t.id)).toEqual(['a', 'b']);
    expect(dropExchange(HISTORY, 'turn:a')?.map((t) => t.id)).toEqual([
      'c',
      'd',
      'e',
      'f',
    ]);
  });

  it('refuses an item that is not in the history', () => {
    expect(dropExchange(HISTORY, 'turn:zzz')).toBeUndefined();
    expect(dropExchange(HISTORY, 7)).toBeUndefined();
  });
});

describe('_drgnx/panel: before_tool answer', () => {
  it('allows only an exact allow', () => {
    expect(readBeforeToolAnswer({ outcome: 'allow' })).toEqual({ allow: true });
    expect(readBeforeToolAnswer({ outcome: 'deny', reason: 'no' })).toEqual({
      allow: false,
      reason: 'no',
    });
    for (const bad of [
      undefined,
      null,
      'allow',
      {},
      { outcome: 'ALLOW' },
      { outcome: 'selected' },
      { allow: true },
    ]) {
      expect(readBeforeToolAnswer(bad).allow).toBe(false);
    }
  });
});

async function* stream(
  items: readonly ServerGeminiStreamEvent[],
): AsyncGenerator<ServerGeminiStreamEvent> {
  yield* items;
  yield {
    type: GeminiEventType.Finished,
    value: {
      reason: FinishReason.STOP,
      usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 10 },
    },
  };
}

const toolRequest: ServerGeminiStreamEvent = {
  type: GeminiEventType.ToolCallRequest,
  value: {
    callId: 'call-1',
    name: 'test_tool',
    args: { path: 'a.txt' },
    isClientInitiated: false,
    prompt_id: 'p',
  },
};

describe('_drgnx/panel: session', () => {
  let chat: Record<string, Mock>;
  let ext: { extMethod: Mock; extNotification: Mock };
  let connection: Record<string, Mock>;
  let execute: Mock;
  let sendMessageStream: Mock;
  let previewCompression: Mock;
  let bus: { publish: Mock; subscribe: Mock; unsubscribe: Mock };
  let policyEngine: { check: Mock };
  let config: Config;

  const makeSession = (features: string[]) =>
    new Session(
      'session-1',
      chat as unknown as GeminiChat,
      config,
      connection as unknown as acp.AgentSideConnection,
      {
        merged: { security: { enablePermanentToolApproval: true } },
      } as unknown as LoadedSettings,
      features,
    );

  const runToolTurn = async (session: Session) => {
    sendMessageStream
      .mockReturnValueOnce(stream([toolRequest]))
      .mockReturnValueOnce(stream([]));
    await session.prompt({
      sessionId: 'session-1',
      prompt: [{ type: 'text', text: 'go' }],
    });
  };

  beforeEach(() => {
    chat = {
      addHistory: vi.fn(),
      recordCompletedToolCalls: vi.fn(),
      getHistory: vi.fn().mockReturnValue([]),
      getHistoryTurns: vi.fn().mockReturnValue(HISTORY),
      getSystemInstruction: vi.fn().mockReturnValue('be brief'),
      getTools: vi.fn().mockReturnValue([]),
      getLastPromptTokenCount: vi.fn().mockReturnValue(321),
      setHistory: vi.fn(),
    };
    ext = { extMethod: vi.fn(), extNotification: vi.fn() };
    connection = {
      sessionUpdate: vi.fn(),
      requestPermission: vi.fn(),
      ...ext,
    };
    execute = vi.fn().mockResolvedValue({ llmContent: 'ok' });
    const tool = {
      kind: 'read',
      build: vi.fn().mockReturnValue({
        getDescription: () => 'Test Tool',
        toolLocations: () => [],
        shouldConfirmExecute: vi.fn().mockResolvedValue(null),
        execute,
      }),
    };
    const registry = { getTool: vi.fn().mockReturnValue(tool) };
    bus = { publish: vi.fn(), subscribe: vi.fn(), unsubscribe: vi.fn() };
    policyEngine = {
      check: vi.fn().mockResolvedValue({ decision: PolicyDecision.ALLOW }),
    };
    sendMessageStream = vi.fn();
    previewCompression = vi.fn();
    config = {
      getModel: vi.fn().mockReturnValue('gemini-pro'),
      getActiveModel: vi.fn().mockReturnValue('gemini-pro'),
      getToolRegistry: vi.fn().mockReturnValue(registry),
      getMessageBus: vi.fn().mockReturnValue(bus),
      getPolicyEngine: vi.fn().mockReturnValue(policyEngine),
      getMaxSessionTurns: vi.fn().mockReturnValue(-1),
      waitForMcpInit: vi.fn(),
      geminiClient: { sendMessageStream, previewCompression },
      get config() {
        return this;
      },
      get toolRegistry() {
        return registry;
      },
    } as unknown as Config;
  });

  it('asks before the tool runs and lets an allow continue', async () => {
    ext.extMethod.mockResolvedValue({ outcome: 'allow' });
    await runToolTurn(makeSession(['before-tool']));
    expect(ext.extMethod).toHaveBeenCalledWith(
      '_drgnx/before_tool',
      expect.objectContaining({
        sessionId: 'session-1',
        toolCall: expect.objectContaining({
          toolCallId: 'call-1',
          title: 'test_tool',
          rawInput: { path: 'a.txt' },
        }),
      }),
    );
    expect(execute).toHaveBeenCalledOnce();
  });

  it.each([
    ['a deny', () => Promise.resolve({ outcome: 'deny', reason: 'policy' })],
    ['a failed request', () => Promise.reject(new Error('link lost'))],
    ['an unreadable answer', () => Promise.resolve({ outcome: 'maybe' })],
    ['no answer', () => Promise.resolve(undefined)],
  ])('denies the tool on %s', async (_name, answer) => {
    ext.extMethod.mockImplementation(answer);
    await runToolTurn(makeSession(['before-tool']));
    expect(execute).not.toHaveBeenCalled();
    expect(connection['sessionUpdate']).toHaveBeenCalledWith(
      expect.objectContaining({
        update: expect.objectContaining({
          sessionUpdate: 'tool_call_update',
          toolCallId: 'call-1',
          status: 'failed',
        }),
      }),
    );
  });

  it('quotes the hub words of a deny in the tool error', async () => {
    ext.extMethod.mockResolvedValue({ outcome: 'deny', reason: 'policy X' });
    await runToolTurn(makeSession(['before-tool']));
    expect(connection['sessionUpdate']).toHaveBeenCalledWith(
      expect.objectContaining({
        update: expect.objectContaining({
          status: 'failed',
          content: [
            { type: 'content', content: { type: 'text', text: 'policy X' } },
          ],
        }),
      }),
    );
  });

  describe("a subagent's tool call", () => {
    // The bus handler a session subscribes for tool confirmation requests.
    const confirm = async (features: string[], subagent?: string) => {
      makeSession(features);
      const handler = bus.subscribe.mock.calls.at(-1)?.[1] as (
        request: unknown,
      ) => Promise<void>;
      await handler({
        type: MessageBusType.TOOL_CONFIRMATION_REQUEST,
        toolCall: { id: 'sub-1', name: 'test_tool', args: { path: 'b.txt' } },
        correlationId: 'corr-1',
        subagent,
      });
    };
    const published = () => bus.publish.mock.calls.map((c) => c[0]);

    it('is asked, and an allow goes on to the policy check', async () => {
      ext.extMethod.mockResolvedValue({ outcome: 'allow' });
      await confirm(['before-tool'], 'helper');
      expect(ext.extMethod).toHaveBeenCalledWith(
        '_drgnx/before_tool',
        expect.objectContaining({
          sessionId: 'session-1',
          toolCall: expect.objectContaining({
            toolCallId: 'sub-1',
            title: 'test_tool',
            rawInput: { path: 'b.txt' },
          }),
        }),
      );
      expect(policyEngine.check).toHaveBeenCalledOnce();
      expect(published()).toEqual([
        expect.objectContaining({ correlationId: 'corr-1', confirmed: true }),
      ]);
    });

    it.each([
      ['a deny', () => Promise.resolve({ outcome: 'deny', reason: 'no' })],
      ['a failed request', () => Promise.reject(new Error('link lost'))],
      ['an unreadable answer', () => Promise.resolve({ outcome: 'maybe' })],
    ])('is denied on %s, before the policy check', async (_name, answer) => {
      ext.extMethod.mockImplementation(answer);
      await confirm(['before-tool'], 'helper');
      expect(policyEngine.check).not.toHaveBeenCalled();
      expect(published()).toEqual([
        {
          type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
          correlationId: 'corr-1',
          confirmed: false,
          requiresUserConfirmation: false,
        },
      ]);
    });

    it('is not asked without the feature, and the main agent is not asked here', async () => {
      await confirm([], 'helper');
      await confirm(['before-tool'], undefined);
      expect(ext.extMethod).not.toHaveBeenCalled();
      expect(policyEngine.check).toHaveBeenCalledTimes(2);
    });
  });

  it('does not ask when before-tool was not negotiated', async () => {
    await runToolTurn(makeSession([]));
    expect(ext.extMethod).not.toHaveBeenCalled();
    expect(execute).toHaveBeenCalledOnce();
  });

  it('sends _drgnx/context after a turn', async () => {
    sendMessageStream.mockReturnValueOnce(stream([]));
    await makeSession(['context']).prompt({
      sessionId: 'session-1',
      prompt: [{ type: 'text', text: 'hi' }],
    });
    expect(ext.extNotification).toHaveBeenCalledWith(
      '_drgnx/context',
      expect.objectContaining({ sessionId: 'session-1', used: 321 }),
    );
  });

  it('does not send _drgnx/context when it was not negotiated, and survives a failure', async () => {
    sendMessageStream.mockReturnValueOnce(stream([]));
    await makeSession([]).prompt({
      sessionId: 'session-1',
      prompt: [{ type: 'text', text: 'hi' }],
    });
    expect(ext.extNotification).not.toHaveBeenCalled();

    ext.extNotification.mockRejectedValue(new Error('gone'));
    sendMessageStream.mockReturnValueOnce(stream([]));
    await expect(
      makeSession(['context']).prompt({
        sessionId: 'session-1',
        prompt: [{ type: 'text', text: 'hi' }],
      }),
    ).resolves.toMatchObject({ stopReason: 'end_turn' });
  });

  it('drops an exchange through setHistory and sends the new breakdown', async () => {
    const session = makeSession(['drop', 'context']);
    await expect(session.dropContextItem('turn:c')).resolves.toEqual({});
    expect(chat['setHistory']).toHaveBeenCalledWith([HISTORY[0], HISTORY[1]]);
    expect(ext.extNotification).toHaveBeenCalledOnce();
  });

  it('refuses to drop an unknown item', async () => {
    const session = makeSession(['drop']);
    await expect(session.dropContextItem('turn:nope')).rejects.toThrow(
      'No such item',
    );
    expect(chat['setHistory']).not.toHaveBeenCalled();
  });

  it('refuses to drop while a turn is running', async () => {
    const session = makeSession(['drop']);
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => (release = r));
    sendMessageStream.mockImplementationOnce(async function* () {
      await gate;
      yield* stream([]);
    });
    const running = session.prompt({
      sessionId: 'session-1',
      prompt: [{ type: 'text', text: 'hi' }],
    });
    await vi.waitFor(() => expect(sendMessageStream).toHaveBeenCalled());
    await expect(session.dropContextItem('turn:a')).rejects.toThrow(
      'turn is running',
    );
    release();
    await running;
    await expect(session.dropContextItem('turn:a')).resolves.toEqual({});
  });

  it('previews a compaction without applying it, and says it costs model calls', async () => {
    previewCompression.mockResolvedValue({
      newHistory: [{ role: 'user', parts: [{ text: 'summary' }] }],
      info: {
        originalTokenCount: 900,
        newTokenCount: 300,
        compressionStatus: CompressionStatus.COMPRESSED,
      },
    });
    const preview = await makeSession(['compact-preview']).previewCompaction();
    expect(preview.frees).toBe(600);
    expect(preview.note).toMatch(/model calls/);
    expect(preview.note).toMatch(/nothing is compacted/);
    expect(chat['setHistory']).not.toHaveBeenCalled();
  });

  it('previews a no-op compaction as freeing nothing', async () => {
    previewCompression.mockResolvedValue({
      newHistory: null,
      info: {
        originalTokenCount: 10,
        newTokenCount: 10,
        compressionStatus: CompressionStatus.NOOP,
      },
    });
    const preview = await makeSession(['compact-preview']).previewCompaction();
    expect(preview.frees).toBe(0);
  });
});

describe('_drgnx/panel: dispatcher', () => {
  let agent: GeminiAgent;
  const initialize = (meta?: Record<string, unknown>) =>
    agent.initialize({
      protocolVersion: 1,
      clientCapabilities: meta ? { _meta: meta } : {},
    });

  let rules: Array<{ source?: string }>;
  let config: Config;
  const makeAgent = (adminPolicy?: string[]) =>
    new GeminiAgent(
      config,
      {} as unknown as LoadedSettings,
      { adminPolicy } as unknown as CliArgs,
      {} as unknown as acp.AgentSideConnection,
    );

  beforeEach(() => {
    rules = [];
    config = {
      initialize: vi.fn(),
      getPolicyEngine: () => ({ getRules: () => rules }),
      get config() {
        return this;
      },
    } as unknown as Config;
    agent = makeAgent();
  });

  describe('the admin policy report', () => {
    let dir: string;
    beforeEach(async () => {
      dir = await fs.mkdtemp(path.join(os.tmpdir(), 'drgnx-panel-'));
      await fs.writeFile(path.join(dir, 'gemini-policy.toml'), '# policy\n');
    });
    afterEach(async () => {
      await fs.rm(dir, { recursive: true, force: true });
    });
    const report = async () =>
      (await initialize({ '_drgnx/panel': { version: 1 } })).agentCapabilities
        ?._meta?.['_drgnx/panel'];

    it('says loaded, with the rule count, when the file gave rules', async () => {
      rules = [
        { source: 'Admin: gemini-policy.toml' },
        { source: 'Admin: gemini-policy.toml' },
        { source: 'Default: read-only.toml' },
      ];
      agent = makeAgent([path.join(dir, 'gemini-policy.toml')]);
      expect(await report()).toMatchObject({
        adminPolicy: { loaded: true, rules: 2 },
      });
    });

    it('reads a directory of policies the same way', async () => {
      rules = [{ source: 'Admin: gemini-policy.toml' }];
      agent = makeAgent([dir]);
      expect(await report()).toMatchObject({
        adminPolicy: { loaded: true, rules: 1 },
      });
    });

    it('says not loaded when the file gave no rules (it did not parse, or Gemini ignored it)', async () => {
      rules = [{ source: 'Default: read-only.toml' }];
      agent = makeAgent([path.join(dir, 'gemini-policy.toml')]);
      expect(await report()).toMatchObject({
        adminPolicy: { loaded: false, rules: 0 },
      });
    });

    it('says not loaded when the file is missing or no policy was asked for', async () => {
      rules = [{ source: 'Admin: gemini-policy.toml' }];
      agent = makeAgent([path.join(dir, 'gone.toml')]);
      expect(await report()).toMatchObject({
        adminPolicy: { loaded: false, rules: 0 },
      });
      agent = makeAgent(undefined);
      expect(await report()).toMatchObject({
        adminPolicy: { loaded: false, rules: 0 },
      });
    });

    it('reports nothing to a client that did not ask for the panel', async () => {
      rules = [{ source: 'Admin: gemini-policy.toml' }];
      agent = makeAgent([path.join(dir, 'gemini-policy.toml')]);
      expect((await initialize()).agentCapabilities?._meta).toBeUndefined();
    });
  });

  it('advertises the panel only to a client that asked for version 1', async () => {
    expect((await initialize()).agentCapabilities?._meta).toBeUndefined();
    expect(
      (await initialize({ '_drgnx/panel': { version: 2 } })).agentCapabilities
        ?._meta,
    ).toBeUndefined();
    expect(
      (await initialize({ '_drgnx/panel': { version: 1 } })).agentCapabilities
        ?._meta,
    ).toEqual({
      '_drgnx/panel': {
        version: 1,
        features: ['before-tool', 'context', 'drop', 'compact-preview'],
        adminPolicy: { loaded: false, rules: 0 },
      },
    });
  });

  it('refuses extension methods when the panel was not negotiated', async () => {
    await initialize();
    await expect(
      agent.extMethod('_drgnx/context/drop', {
        sessionId: 's',
        item: 'turn:a',
      }),
    ).rejects.toMatchObject({ code: -32601 });
  });

  it('refuses methods it does not serve, pin among them', async () => {
    await initialize({ '_drgnx/panel': { version: 1 } });
    for (const method of [
      '_drgnx/context/pin',
      '_drgnx/before_tool',
      '_drgnx/unknown',
      'toString',
    ]) {
      await expect(
        agent.extMethod(method, { sessionId: 's' }),
      ).rejects.toMatchObject({ code: -32601 });
    }
  });

  it('routes drop and preview to the session', async () => {
    await initialize({ '_drgnx/panel': { version: 1 } });
    const session = {
      dropContextItem: vi.fn().mockResolvedValue({}),
      previewCompaction: vi.fn().mockResolvedValue({ frees: 5, note: 'n' }),
    };
    vi.spyOn(AcpSessionManager.prototype, 'getSession').mockImplementation(
      (id) => (id === 's' ? (session as unknown as Session) : undefined),
    );
    await expect(
      agent.extMethod('_drgnx/context/drop', {
        sessionId: 's',
        item: 'turn:a',
      }),
    ).resolves.toEqual({});
    expect(session.dropContextItem).toHaveBeenCalledWith('turn:a');
    await expect(
      agent.extMethod('_drgnx/compact/preview', { sessionId: 's' }),
    ).resolves.toEqual({ frees: 5, note: 'n' });
    await expect(
      agent.extMethod('_drgnx/compact/preview', { sessionId: 'other' }),
    ).rejects.toMatchObject({ code: -32602 });
  });
});
