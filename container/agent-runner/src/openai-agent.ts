/**
 * OpenAI-compatible Agent Runner for NanoClaw
 * Alternative to the Claude Code SDK path. Uses the OpenAI /v1/chat/completions
 * API with tool calling, which both vLLM and Ollama serve.
 *
 * Maintains the same container I/O protocol (stdin JSON, stdout OUTPUT_MARKER pairs,
 * IPC file polling) so the host code requires no changes.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import {
  BUILTIN_TOOL_DEFINITIONS,
  ToolDefinition,
  executeBuiltinTool,
  isBuiltinTool,
} from './builtin-tools.js';
import { McpBridge } from './mcp-bridge.js';

// --- Types ---

interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  assistantName?: string;
}

interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
}

interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

interface ChatCompletionResponse {
  id: string;
  choices: Array<{
    index: number;
    message: {
      role: 'assistant';
      content: string | null;
      tool_calls?: ToolCall[];
    };
    finish_reason: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// --- Constants ---

const IPC_INPUT_DIR = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const IPC_POLL_MS = 500;
const MAX_TURNS = 50; // Safety limit on agentic turns per query
const MAX_HISTORY_MESSAGES = 200; // Truncate history when it gets too long
const SESSION_DIR = '/workspace/group/.sessions';

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

// --- Helpers ---

function log(message: string): void {
  console.error(`[openai-agent] ${message}`);
}

function writeOutput(output: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }
    return true;
  }
  return false;
}

function drainIpcInput(): string[] {
  try {
    fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
    const files = fs.readdirSync(IPC_INPUT_DIR)
      .filter(f => f.endsWith('.json'))
      .sort();

    const messages: string[] = [];
    for (const file of files) {
      const filePath = path.join(IPC_INPUT_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        fs.unlinkSync(filePath);
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
        }
      } catch (err) {
        log(`Failed to process input file ${file}: ${err instanceof Error ? err.message : String(err)}`);
        try { fs.unlinkSync(filePath); } catch { /* ignore */ }
      }
    }
    return messages;
  } catch (err) {
    log(`IPC drain error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

function waitForIpcMessage(): Promise<string | null> {
  return new Promise((resolve) => {
    const poll = () => {
      if (shouldClose()) { resolve(null); return; }
      const messages = drainIpcInput();
      if (messages.length > 0) { resolve(messages.join('\n')); return; }
      setTimeout(poll, IPC_POLL_MS);
    };
    poll();
  });
}

// --- Session management ---

function loadSession(sessionId: string): ChatMessage[] {
  const sessionFile = path.join(SESSION_DIR, `${sessionId}.json`);
  try {
    if (fs.existsSync(sessionFile)) {
      const data = JSON.parse(fs.readFileSync(sessionFile, 'utf-8'));
      return data.messages || [];
    }
  } catch (err) {
    log(`Failed to load session ${sessionId}: ${err instanceof Error ? err.message : String(err)}`);
  }
  return [];
}

function saveSession(sessionId: string, messages: ChatMessage[]): void {
  fs.mkdirSync(SESSION_DIR, { recursive: true });
  const sessionFile = path.join(SESSION_DIR, `${sessionId}.json`);
  fs.writeFileSync(sessionFile, JSON.stringify({
    sessionId,
    updatedAt: new Date().toISOString(),
    messages,
  }, null, 2));
}

function generateSessionId(): string {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/** Truncate history to stay within context limits */
function truncateHistory(messages: ChatMessage[]): ChatMessage[] {
  if (messages.length <= MAX_HISTORY_MESSAGES) return messages;
  // Keep system prompt + recent messages
  const systemMessages = messages.filter(m => m.role === 'system');
  const nonSystem = messages.filter(m => m.role !== 'system');
  const kept = nonSystem.slice(-MAX_HISTORY_MESSAGES);
  return [...systemMessages, ...kept];
}

// --- System prompt ---

function buildSystemPrompt(input: ContainerInput): string {
  const parts: string[] = [];

  const name = input.assistantName || 'Assistant';
  parts.push(`You are ${name}, a helpful AI assistant running inside a sandboxed container.`);
  parts.push(`Your working directory is /workspace/group. You have access to tools for running shell commands, reading/writing files, and interacting with the messaging system.`);
  parts.push(`When you want to send a message to the user, use the mcp__nanoclaw__send_message tool. Your final text response will also be sent to the user.`);

  if (input.isMain) {
    parts.push(`You are the MAIN group agent with elevated privileges. You can register new groups and manage all scheduled tasks.`);
  }

  // Load CLAUDE.md as instructions (works for any model)
  const claudeMdPaths = [
    '/workspace/group/CLAUDE.md',
    '/workspace/project/CLAUDE.md',
  ];
  for (const mdPath of claudeMdPaths) {
    if (fs.existsSync(mdPath)) {
      try {
        const content = fs.readFileSync(mdPath, 'utf-8');
        parts.push(`\n--- Instructions from ${path.basename(mdPath)} ---\n${content}`);
      } catch { /* ignore */ }
    }
  }

  // Load global context for non-main groups
  if (!input.isMain) {
    const globalMd = '/workspace/global/CLAUDE.md';
    if (fs.existsSync(globalMd)) {
      try {
        const content = fs.readFileSync(globalMd, 'utf-8');
        parts.push(`\n--- Global Instructions ---\n${content}`);
      } catch { /* ignore */ }
    }
  }

  return parts.join('\n\n');
}

// --- OpenAI API client ---

async function chatCompletion(
  messages: ChatMessage[],
  tools: ToolDefinition[],
): Promise<ChatCompletionResponse> {
  const baseUrl = process.env.LLM_BASE_URL || 'http://localhost:11434/v1';
  const model = process.env.LLM_MODEL || 'llama3.2';
  const apiKey = process.env.LLM_API_KEY || 'not-needed';

  const url = `${baseUrl.replace(/\/$/, '')}/chat/completions`;

  const body: Record<string, unknown> = {
    model,
    messages,
  };

  // Only include tools if we have them — some models don't support empty tools array
  if (tools.length > 0) {
    body.tools = tools;
    body.tool_choice = 'auto';
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`LLM API error ${response.status}: ${errorText.slice(0, 500)}`);
  }

  return await response.json() as ChatCompletionResponse;
}

// --- Agentic loop ---

async function runAgentLoop(
  prompt: string,
  sessionId: string,
  tools: ToolDefinition[],
  mcpBridge: McpBridge | null,
  containerInput: ContainerInput,
): Promise<{ closedDuringQuery: boolean }> {
  // Load or start history
  let messages: ChatMessage[] = loadSession(sessionId);

  // Ensure system prompt is first
  const systemPrompt = buildSystemPrompt(containerInput);
  if (messages.length === 0 || messages[0].role !== 'system') {
    messages = [{ role: 'system', content: systemPrompt }, ...messages.filter(m => m.role !== 'system')];
  } else {
    // Update system prompt in case it changed
    messages[0].content = systemPrompt;
  }

  // Add the new user message
  messages.push({ role: 'user', content: prompt });
  messages = truncateHistory(messages);

  let closedDuringQuery = false;

  for (let turn = 0; turn < MAX_TURNS; turn++) {
    // Check for close sentinel
    if (shouldClose()) {
      log('Close sentinel detected during agent loop');
      closedDuringQuery = true;
      break;
    }

    // Drain any IPC messages that arrived during tool execution
    const ipcMessages = drainIpcInput();
    for (const text of ipcMessages) {
      log(`IPC message during turn ${turn}: ${text.slice(0, 100)}`);
      messages.push({ role: 'user', content: text });
    }

    log(`Turn ${turn + 1}: sending ${messages.length} messages to LLM...`);

    let response: ChatCompletionResponse;
    try {
      response = await chatCompletion(messages, tools);
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      log(`LLM API error: ${errMsg}`);
      writeOutput({ status: 'error', result: null, newSessionId: sessionId, error: errMsg });
      saveSession(sessionId, messages);
      return { closedDuringQuery };
    }

    const choice = response.choices?.[0];
    if (!choice) {
      log('No choices in LLM response');
      writeOutput({ status: 'error', result: null, newSessionId: sessionId, error: 'No response from model' });
      saveSession(sessionId, messages);
      return { closedDuringQuery };
    }

    const assistantMessage = choice.message;
    log(`Turn ${turn + 1}: finish_reason=${choice.finish_reason}, tool_calls=${assistantMessage.tool_calls?.length || 0}, content=${assistantMessage.content ? assistantMessage.content.slice(0, 100) : '(none)'}`);

    // Add assistant message to history
    const historyEntry: ChatMessage = {
      role: 'assistant',
      content: assistantMessage.content,
    };
    if (assistantMessage.tool_calls && assistantMessage.tool_calls.length > 0) {
      historyEntry.tool_calls = assistantMessage.tool_calls;
    }
    messages.push(historyEntry);

    // If there are tool calls, execute them
    if (assistantMessage.tool_calls && assistantMessage.tool_calls.length > 0) {
      for (const toolCall of assistantMessage.tool_calls) {
        const toolName = toolCall.function.name;
        let args: Record<string, unknown>;
        try {
          args = JSON.parse(toolCall.function.arguments);
        } catch {
          args = {};
          log(`Failed to parse tool arguments for ${toolName}: ${toolCall.function.arguments}`);
        }

        log(`Executing tool: ${toolName}`);

        let result;
        if (isBuiltinTool(toolName)) {
          result = executeBuiltinTool(toolName, args);
        } else if (mcpBridge?.isMcpTool(toolName)) {
          result = await mcpBridge.executeTool(toolName, args);
        } else {
          result = { success: false, output: `Unknown tool: ${toolName}` };
        }

        if (result) {
          log(`Tool ${toolName}: success=${result.success}, output=${result.output.slice(0, 100)}`);
        }

        messages.push({
          role: 'tool',
          tool_call_id: toolCall.id,
          content: result?.output || '(no output)',
        });
      }

      // Continue the loop to send tool results back to the model
      continue;
    }

    // No tool calls — this is the final response
    const textResult = assistantMessage.content;
    if (textResult) {
      writeOutput({
        status: 'success',
        result: textResult,
        newSessionId: sessionId,
      });
    } else {
      // Model returned no content and no tool calls
      writeOutput({
        status: 'success',
        result: null,
        newSessionId: sessionId,
      });
    }

    break;
  }

  // Save session for potential resume
  saveSession(sessionId, messages);
  return { closedDuringQuery };
}

// --- Main entry point ---

export async function runOpenAIAgent(containerInput: ContainerInput): Promise<void> {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'ipc-mcp-stdio.js');

  // Initialize MCP bridge for nanoclaw tools
  let mcpBridge: McpBridge | null = null;
  try {
    mcpBridge = new McpBridge(mcpServerPath, {
      NANOCLAW_CHAT_JID: containerInput.chatJid,
      NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
      NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
    });
    await mcpBridge.start();
  } catch (err) {
    log(`Failed to start MCP bridge: ${err instanceof Error ? err.message : String(err)}`);
    log('Continuing without MCP tools (send_message, schedule_task, etc.)');
    mcpBridge = null;
  }

  // Build tool definitions
  const tools: ToolDefinition[] = [
    ...BUILTIN_TOOL_DEFINITIONS,
    ...(mcpBridge?.getToolDefinitions() || []),
  ];
  log(`Available tools: ${tools.map(t => t.function.name).join(', ')}`);

  // Session management
  let sessionId = containerInput.sessionId || generateSessionId();
  fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });

  // Clean up stale _close sentinel
  try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }

  // Build initial prompt
  let prompt = containerInput.prompt;
  if (containerInput.isScheduledTask) {
    prompt = `[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n${prompt}`;
  }
  const pending = drainIpcInput();
  if (pending.length > 0) {
    log(`Draining ${pending.length} pending IPC messages into initial prompt`);
    prompt += '\n' + pending.join('\n');
  }

  // Emit session init
  writeOutput({ status: 'success', result: null, newSessionId: sessionId });

  // Query loop: run agent → wait for IPC message → run again → repeat
  try {
    while (true) {
      log(`Starting agent loop (session: ${sessionId})...`);

      const result = await runAgentLoop(prompt, sessionId, tools, mcpBridge, containerInput);

      if (result.closedDuringQuery) {
        log('Close sentinel consumed during agent loop, exiting');
        break;
      }

      // Emit session update
      writeOutput({ status: 'success', result: null, newSessionId: sessionId });

      log('Agent loop ended, waiting for next IPC message...');
      const nextMessage = await waitForIpcMessage();
      if (nextMessage === null) {
        log('Close sentinel received, exiting');
        break;
      }

      log(`Got new message (${nextMessage.length} chars), starting new agent loop`);
      prompt = nextMessage;
    }
  } finally {
    await mcpBridge?.shutdown();
  }
}
