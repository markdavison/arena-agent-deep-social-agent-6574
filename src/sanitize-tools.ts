import { jsonSchema, asSchema } from "ai";
import type { ToolSet } from "ai";
import type { JSONSchema7 } from "@ai-sdk/provider";

/**
 * JSON Schema properties that many non-OpenAI providers
 * (DeepSeek via Chutes, etc.) reject or cannot parse.
 */
const STRIP_KEYS = new Set([
  "minLength",
  "maxLength",
  "minItems",
  "maxItems",
  "exclusiveMinimum",
  "exclusiveMaximum",
  "pattern",
  "format",
  "default",
  "$schema",
]);

function sanitizeSchema(
  schema: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(schema)) {
    if (STRIP_KEYS.has(key)) continue;

    if (
      value !== null &&
      typeof value === "object" &&
      !Array.isArray(value)
    ) {
      out[key] = sanitizeSchema(
        value as Record<string, unknown>,
      );
    } else if (Array.isArray(value)) {
      out[key] = value.map((item) =>
        item !== null &&
        typeof item === "object" &&
        !Array.isArray(item)
          ? sanitizeSchema(item as Record<string, unknown>)
          : item,
      );
    } else {
      out[key] = value;
    }
  }

  return out;
}

/**
 * MCP tools return content as arrays of content blocks:
 *   [{ type: "text", text: "..." }]
 * The AI SDK serializes this as a JSON string in tool
 * results, which DeepSeek cannot parse. This function
 * extracts plain text from content-block arrays.
 */
function extractTexts(
  parts: unknown[],
): string[] {
  const texts: string[] = [];
  for (const part of parts) {
    if (
      part !== null &&
      typeof part === "object" &&
      "type" in part &&
      (part as Record<string, unknown>).type === "text" &&
      "text" in part &&
      typeof (part as Record<string, unknown>).text ===
        "string"
    ) {
      texts.push(
        (part as Record<string, unknown>).text as string,
      );
    }
  }
  return texts;
}

function flattenContent(result: unknown): string {
  if (typeof result === "string") return result;

  // MCP result: { content: [{ type: "text", text }] }
  if (
    result !== null &&
    typeof result === "object" &&
    "content" in result &&
    Array.isArray(
      (result as Record<string, unknown>).content,
    )
  ) {
    const parts = (result as Record<string, unknown>)
      .content as unknown[];
    const texts = extractTexts(parts);
    if (texts.length > 0) return texts.join("\n");
  }

  // Bare array of content blocks
  if (Array.isArray(result)) {
    const texts = extractTexts(result);
    if (texts.length > 0) return texts.join("\n");
  }

  return JSON.stringify(result);
}

/**
 * Wraps a tool's execute to flatten MCP content-block
 * arrays into plain text strings.
 */
function wrapExecute(
  tool: ToolSet[string],
): ToolSet[string] {
  const original = tool.execute;
  if (!original) return tool;

  return {
    ...tool,
    // Override toModelOutput so the AI SDK doesn't
    // serialize content arrays — we flatten them instead
    toModelOutput: undefined,
    execute: async (...executeArgs: Parameters<
      NonNullable<typeof original>
    >) => {
      const result = await original(...executeArgs);
      return flattenContent(result);
    },
  };
}

/**
 * Sanitize tool schemas and execution for non-OpenAI
 * providers:
 * 1. Strip unsupported JSON Schema properties
 * 2. Flatten MCP content-block arrays in tool results
 */
export function sanitizeTools(tools: ToolSet): ToolSet {
  const out: ToolSet = {};

  for (const [name, tool] of Object.entries(tools)) {
    const resolved = asSchema(tool.inputSchema);
    const raw = resolved.jsonSchema;
    if (!raw || typeof raw !== "object") {
      out[name] = wrapExecute(tool);
      continue;
    }

    const cleaned = sanitizeSchema(
      raw as Record<string, unknown>,
    );

    out[name] = wrapExecute({
      ...tool,
      inputSchema: jsonSchema(cleaned as JSONSchema7),
    });
  }

  return out;
}
