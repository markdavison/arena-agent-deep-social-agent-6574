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
 * Strips unsupported JSON Schema constraints from every
 * tool's parameter schema so non-OpenAI providers don't
 * reject the request.
 */
export function sanitizeTools(tools: ToolSet): ToolSet {
  const out: ToolSet = {};

  for (const [name, tool] of Object.entries(tools)) {
    const resolved = asSchema(tool.inputSchema);
    const raw = resolved.jsonSchema;
    if (!raw || typeof raw !== "object") {
      out[name] = tool;
      continue;
    }

    const cleaned = sanitizeSchema(
      raw as Record<string, unknown>,
    );

    out[name] = {
      ...tool,
      inputSchema: jsonSchema(cleaned as JSONSchema7),
    };
  }

  return out;
}
