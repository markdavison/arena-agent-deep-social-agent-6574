import { generateText, stepCountIs } from "ai";
import type { ToolSet } from "ai";
import { createXai } from "@ai-sdk/xai";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";

const MAX_ATTEMPTS = 3;
const SUBMIT_TOOL = "submit_decision";

interface AgentConfig {
  system_prompt: string;
  model: {
    provider: string;
    model_id: string;
    base_url?: string;
  };
}

function getModel(config: AgentConfig) {
  const apiKey = process.env["LLM_API_KEY"];
  if (!apiKey) throw new Error("Missing LLM_API_KEY env var");

  const { provider, model_id, base_url } = config.model;

  switch (provider) {
    case "xai":
      return createXai({ apiKey })(model_id);
    case "openai":
      return createOpenAI({ apiKey })(model_id);
    case "anthropic":
      return createAnthropic({ apiKey })(model_id);
    case "google":
      return createGoogleGenerativeAI({ apiKey })(model_id);
    default:
      if (!base_url) {
        throw new Error(
          `Unknown provider "${provider}" with no base_url`,
        );
      }
      return createOpenAI({
        baseURL: base_url,
        apiKey,
      }).chat(model_id);
  }
}

function hasSubmitDecision(
  steps: Awaited<ReturnType<typeof generateText>>["steps"],
): boolean {
  return steps.some((step) =>
    step.toolCalls.some((c) => c.toolName === SUBMIT_TOOL),
  );
}

function logSteps(
  steps: Awaited<ReturnType<typeof generateText>>["steps"],
): void {
  for (const step of steps) {
    for (const call of step.toolCalls) {
      if (call.toolName === SUBMIT_TOOL) {
        console.log(
          `[agent] Tool call: ${call.toolName}`,
          JSON.stringify(call.input),
        );
      } else {
        console.log(`[agent] Tool call: ${call.toolName}`);
      }
    }
  }
}

export async function runConfigStrategy(
  tools: ToolSet,
  config: AgentConfig,
): Promise<void> {
  const model = getModel(config);
  const messages: Array<{ role: "user" | "assistant"; content: string }> = [
    {
      role: "user",
      content:
        "Analyze the market and make your trading " +
        "decision for this interval.",
    },
  ];

  let totalSteps = 0;
  let totalToolCalls = 0;

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const result = await generateText({
      model,
      tools,
      stopWhen: stepCountIs(10),
      system: config.system_prompt,
      messages,
    });

    logSteps(result.steps);
    totalSteps += result.steps.length;
    totalToolCalls += result.steps.reduce(
      (n, s) => n + s.toolCalls.length,
      0,
    );

    if (hasSubmitDecision(result.steps)) break;

    if (attempt < MAX_ATTEMPTS) {
      console.log(
        `[agent] No ${SUBMIT_TOOL} in attempt ` +
          `${String(attempt)}, retrying...`,
      );
      messages.push({
        role: "assistant",
        content: result.text,
      });
      messages.push({
        role: "user",
        content:
          "You did not call submit_decision. You MUST call " +
          "submit_decision with your trades to complete " +
          "your turn. Analyze the market and submit now.",
      });
    } else {
      console.log(
        `[agent] Warning: no ${SUBMIT_TOOL} after ` +
          `${String(MAX_ATTEMPTS)} attempts`,
      );
    }
  }

  console.log(
    `[agent] Completed ${String(totalSteps)} step(s), ` +
      `${String(totalToolCalls)} tool call(s)`,
  );
}
