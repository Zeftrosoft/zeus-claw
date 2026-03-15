/**
 * MCP Bridge for the OpenAI-compatible agent runner.
 * Spawns ipc-mcp-stdio.ts as a child process, discovers its tools via JSON-RPC,
 * and bridges them into OpenAI function calling format.
 */

import { ChildProcess, spawn } from 'child_process';
import { ToolDefinition, ToolResult } from './builtin-tools.js';

interface JsonRpcRequest {
  jsonrpc: '2.0';
  id: number;
  method: string;
  params?: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: '2.0';
  id: number;
  result?: unknown;
  error?: { code: number; message: string };
}

interface McpToolSchema {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
}

interface McpToolsListResult {
  tools: McpToolSchema[];
}

interface McpCallToolResult {
  content: Array<{ type: string; text?: string }>;
  isError?: boolean;
}

function log(message: string): void {
  console.error(`[mcp-bridge] ${message}`);
}

export class McpBridge {
  private proc: ChildProcess | null = null;
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: JsonRpcResponse) => void; reject: (e: Error) => void }>();
  private buffer = '';
  private tools: McpToolSchema[] = [];
  private initialized = false;

  constructor(
    private mcpServerPath: string,
    private env: Record<string, string>,
  ) {}

  async start(): Promise<void> {
    this.proc = spawn('node', [this.mcpServerPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, ...this.env },
    });

    this.proc.stdout!.on('data', (data: Buffer) => {
      this.buffer += data.toString();
      this.processBuffer();
    });

    this.proc.stderr!.on('data', (data: Buffer) => {
      // MCP server logs
      const lines = data.toString().trim().split('\n');
      for (const line of lines) {
        if (line) log(`server: ${line}`);
      }
    });

    this.proc.on('error', (err) => {
      log(`MCP server process error: ${err.message}`);
    });

    this.proc.on('close', (code) => {
      log(`MCP server exited with code ${code}`);
      // Reject all pending requests
      for (const [, { reject }] of this.pending) {
        reject(new Error(`MCP server exited with code ${code}`));
      }
      this.pending.clear();
    });

    // Initialize the MCP protocol
    await this.initialize();
  }

  private processBuffer(): void {
    // MCP uses JSON-RPC over stdio with Content-Length headers
    while (this.buffer.length > 0) {
      const headerEnd = this.buffer.indexOf('\r\n\r\n');
      if (headerEnd === -1) break;

      const header = this.buffer.slice(0, headerEnd);
      const match = header.match(/Content-Length:\s*(\d+)/i);
      if (!match) {
        // Skip malformed header
        this.buffer = this.buffer.slice(headerEnd + 4);
        continue;
      }

      const contentLength = parseInt(match[1], 10);
      const contentStart = headerEnd + 4;
      if (this.buffer.length < contentStart + contentLength) break; // Wait for more data

      const content = this.buffer.slice(contentStart, contentStart + contentLength);
      this.buffer = this.buffer.slice(contentStart + contentLength);

      try {
        const message = JSON.parse(content) as JsonRpcResponse;
        if (message.id != null && this.pending.has(message.id)) {
          const { resolve } = this.pending.get(message.id)!;
          this.pending.delete(message.id);
          resolve(message);
        }
      } catch (err) {
        log(`Failed to parse MCP response: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }

  private sendRequest(method: string, params?: Record<string, unknown>): Promise<JsonRpcResponse> {
    return new Promise((resolve, reject) => {
      if (!this.proc?.stdin?.writable) {
        reject(new Error('MCP server not running'));
        return;
      }

      const id = this.nextId++;
      const request: JsonRpcRequest = {
        jsonrpc: '2.0',
        id,
        method,
        params,
      };

      this.pending.set(id, { resolve, reject });

      const body = JSON.stringify(request);
      const message = `Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`;
      this.proc.stdin!.write(message);

      // Timeout after 10s
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`MCP request timed out: ${method}`));
        }
      }, 10_000);
    });
  }

  private async initialize(): Promise<void> {
    // Send initialize request
    const initResponse = await this.sendRequest('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'nanoclaw-openai-agent', version: '1.0.0' },
    });

    if (initResponse.error) {
      throw new Error(`MCP initialize failed: ${initResponse.error.message}`);
    }

    // Send initialized notification (no response expected, but send as request for simplicity)
    const body = JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' });
    const message = `Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`;
    this.proc!.stdin!.write(message);

    // Discover tools
    const toolsResponse = await this.sendRequest('tools/list', {});
    if (toolsResponse.error) {
      throw new Error(`MCP tools/list failed: ${toolsResponse.error.message}`);
    }

    const result = toolsResponse.result as McpToolsListResult;
    this.tools = result.tools || [];
    this.initialized = true;
    log(`Discovered ${this.tools.length} MCP tools: ${this.tools.map(t => t.name).join(', ')}`);
  }

  /** Get OpenAI-format tool definitions for all MCP tools */
  getToolDefinitions(): ToolDefinition[] {
    return this.tools.map((tool) => ({
      type: 'function' as const,
      function: {
        name: `mcp__nanoclaw__${tool.name}`,
        description: tool.description || tool.name,
        parameters: tool.inputSchema || { type: 'object', properties: {} },
      },
    }));
  }

  /** Check if a tool name belongs to this MCP bridge */
  isMcpTool(name: string): boolean {
    return name.startsWith('mcp__nanoclaw__');
  }

  /** Execute an MCP tool call */
  async executeTool(name: string, args: Record<string, unknown>): Promise<ToolResult> {
    if (!this.initialized) {
      return { success: false, output: 'MCP bridge not initialized' };
    }

    // Strip the mcp__nanoclaw__ prefix
    const toolName = name.replace('mcp__nanoclaw__', '');

    try {
      const response = await this.sendRequest('tools/call', {
        name: toolName,
        arguments: args,
      });

      if (response.error) {
        return { success: false, output: `MCP error: ${response.error.message}` };
      }

      const result = response.result as McpCallToolResult;
      const text = result.content
        .filter(c => c.type === 'text' && c.text)
        .map(c => c.text!)
        .join('\n');

      return { success: !result.isError, output: text || '(no output)' };
    } catch (err) {
      return { success: false, output: `MCP call failed: ${err instanceof Error ? err.message : String(err)}` };
    }
  }

  /** Shut down the MCP server process */
  async shutdown(): Promise<void> {
    if (this.proc) {
      this.proc.stdin?.end();
      this.proc.kill('SIGTERM');
      this.proc = null;
    }
  }
}
