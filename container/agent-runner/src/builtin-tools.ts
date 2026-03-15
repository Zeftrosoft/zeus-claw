/**
 * Built-in tools for the OpenAI-compatible agent runner.
 * Replaces the Claude Code SDK's built-in tools (Bash, Read, Write, Edit, Glob, Grep).
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

// OpenAI function calling schema
export interface ToolDefinition {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export interface ToolResult {
  success: boolean;
  output: string;
}

const CWD = '/workspace/group';
const BASH_TIMEOUT_MS = 120_000;
const MAX_OUTPUT_SIZE = 100_000; // chars

function truncate(text: string, max = MAX_OUTPUT_SIZE): string {
  if (text.length <= max) return text;
  return text.slice(0, max) + `\n... (truncated, ${text.length - max} chars omitted)`;
}

function resolvePath(p: string): string {
  if (path.isAbsolute(p)) return p;
  return path.resolve(CWD, p);
}

// --- Tool implementations ---

function bashTool(args: { command: string }): ToolResult {
  try {
    const result = execSync(args.command, {
      cwd: CWD,
      timeout: BASH_TIMEOUT_MS,
      maxBuffer: 10 * 1024 * 1024,
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    return { success: true, output: truncate(result) };
  } catch (err: unknown) {
    const e = err as { stdout?: string; stderr?: string; status?: number; message?: string };
    const stdout = e.stdout || '';
    const stderr = e.stderr || '';
    const code = e.status ?? 'unknown';
    return {
      success: false,
      output: truncate(`Exit code: ${code}\n${stdout}\n${stderr}`.trim()),
    };
  }
}

function readFileTool(args: { path: string; start_line?: number; end_line?: number }): ToolResult {
  const filePath = resolvePath(args.path);
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    if (args.start_line || args.end_line) {
      const lines = content.split('\n');
      const start = Math.max(0, (args.start_line || 1) - 1);
      const end = args.end_line || lines.length;
      const sliced = lines.slice(start, end);
      const numbered = sliced.map((line, i) => `${start + i + 1}|${line}`).join('\n');
      return { success: true, output: truncate(numbered) };
    }
    const lines = content.split('\n');
    const numbered = lines.map((line, i) => `${i + 1}|${line}`).join('\n');
    return { success: true, output: truncate(numbered) };
  } catch (err) {
    return { success: false, output: `Error reading file: ${err instanceof Error ? err.message : String(err)}` };
  }
}

function writeFileTool(args: { path: string; content: string }): ToolResult {
  const filePath = resolvePath(args.path);
  try {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, args.content);
    return { success: true, output: `Wrote ${args.content.length} bytes to ${args.path}` };
  } catch (err) {
    return { success: false, output: `Error writing file: ${err instanceof Error ? err.message : String(err)}` };
  }
}

function editFileTool(args: { path: string; old_string: string; new_string: string }): ToolResult {
  const filePath = resolvePath(args.path);
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const idx = content.indexOf(args.old_string);
    if (idx === -1) {
      return { success: false, output: `old_string not found in ${args.path}. Make sure it matches exactly (including whitespace).` };
    }
    // Check for multiple occurrences
    const secondIdx = content.indexOf(args.old_string, idx + 1);
    if (secondIdx !== -1) {
      return { success: false, output: `old_string appears multiple times in ${args.path}. Include more context to make it unique.` };
    }
    const updated = content.slice(0, idx) + args.new_string + content.slice(idx + args.old_string.length);
    fs.writeFileSync(filePath, updated);
    return { success: true, output: `Edited ${args.path}` };
  } catch (err) {
    return { success: false, output: `Error editing file: ${err instanceof Error ? err.message : String(err)}` };
  }
}

function listFilesTool(args: { path?: string; pattern?: string }): ToolResult {
  const dir = resolvePath(args.path || '.');
  try {
    if (args.pattern) {
      // Use find with name pattern
      const result = execSync(
        `find ${JSON.stringify(dir)} -maxdepth 5 -type f -name ${JSON.stringify(args.pattern)} 2>/dev/null | head -200`,
        { encoding: 'utf-8', timeout: 10_000 },
      );
      return { success: true, output: result.trim() || 'No files found.' };
    }
    // List directory recursively (limited depth)
    const result = execSync(
      `find ${JSON.stringify(dir)} -maxdepth 3 -type f 2>/dev/null | head -200`,
      { encoding: 'utf-8', timeout: 10_000 },
    );
    return { success: true, output: result.trim() || 'No files found.' };
  } catch (err) {
    return { success: false, output: `Error listing files: ${err instanceof Error ? err.message : String(err)}` };
  }
}

function searchFilesTool(args: { pattern: string; path?: string; include?: string }): ToolResult {
  const dir = resolvePath(args.path || '.');
  try {
    const includeArg = args.include ? `--include=${JSON.stringify(args.include)}` : '';
    const result = execSync(
      `grep -rn ${includeArg} -- ${JSON.stringify(args.pattern)} ${JSON.stringify(dir)} 2>/dev/null | head -100`,
      { encoding: 'utf-8', timeout: 15_000 },
    );
    return { success: true, output: truncate(result.trim()) || 'No matches found.' };
  } catch (err: unknown) {
    const e = err as { status?: number };
    // grep returns exit 1 for no matches
    if (e.status === 1) return { success: true, output: 'No matches found.' };
    return { success: false, output: `Error searching: ${err instanceof Error ? err.message : String(err)}` };
  }
}

// --- Tool definitions (OpenAI function calling schema) ---

export const BUILTIN_TOOL_DEFINITIONS: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'bash',
      description: 'Execute a shell command. Working directory is /workspace/group. Use for running scripts, installing packages, git operations, or any system command.',
      parameters: {
        type: 'object',
        properties: {
          command: { type: 'string', description: 'The shell command to execute' },
        },
        required: ['command'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'read_file',
      description: 'Read the contents of a file. Returns line-numbered content.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path (relative to /workspace/group or absolute)' },
          start_line: { type: 'number', description: 'Optional start line (1-indexed)' },
          end_line: { type: 'number', description: 'Optional end line (inclusive)' },
        },
        required: ['path'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'write_file',
      description: 'Write content to a file. Creates parent directories if needed. Overwrites existing files.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path (relative to /workspace/group or absolute)' },
          content: { type: 'string', description: 'The complete file content to write' },
        },
        required: ['path', 'content'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'edit_file',
      description: 'Edit a file by replacing an exact string match. The old_string must appear exactly once in the file.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path' },
          old_string: { type: 'string', description: 'Exact string to find and replace (must be unique in file)' },
          new_string: { type: 'string', description: 'Replacement string' },
        },
        required: ['path', 'old_string', 'new_string'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'list_files',
      description: 'List files in a directory. Optionally filter by glob pattern.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Directory path (default: current directory)' },
          pattern: { type: 'string', description: 'Optional filename glob pattern (e.g., "*.ts", "*.json")' },
        },
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'search_files',
      description: 'Search for a text pattern in files (like grep). Returns matching lines with file paths and line numbers.',
      parameters: {
        type: 'object',
        properties: {
          pattern: { type: 'string', description: 'Text or regex pattern to search for' },
          path: { type: 'string', description: 'Directory to search in (default: current directory)' },
          include: { type: 'string', description: 'Optional file glob to filter (e.g., "*.ts")' },
        },
        required: ['pattern'],
      },
    },
  },
];

// --- Tool executor ---

const TOOL_HANDLERS: Record<string, (args: Record<string, unknown>) => ToolResult> = {
  bash: (args) => bashTool(args as { command: string }),
  read_file: (args) => readFileTool(args as { path: string; start_line?: number; end_line?: number }),
  write_file: (args) => writeFileTool(args as { path: string; content: string }),
  edit_file: (args) => editFileTool(args as { path: string; old_string: string; new_string: string }),
  list_files: (args) => listFilesTool(args as { path?: string; pattern?: string }),
  search_files: (args) => searchFilesTool(args as { pattern: string; path?: string; include?: string }),
};

export function executeBuiltinTool(name: string, args: Record<string, unknown>): ToolResult | null {
  const handler = TOOL_HANDLERS[name];
  if (!handler) return null;
  return handler(args);
}

export function isBuiltinTool(name: string): boolean {
  return name in TOOL_HANDLERS;
}
