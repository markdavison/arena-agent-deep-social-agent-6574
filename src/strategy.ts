import { generateText, stepCountIs } from "ai";
import { xai } from "@ai-sdk/xai";
import type { ToolSet } from "ai";

const SYSTEM_PROMPT = [
  "You are a trading agent in Agent Arena, a paper-trading",
  "competition on Bittensor. Virtual money — no downside to trading.",
  "Holding idle USD means falling behind agents who deploy capital.",
  "",
  "You have Arena tools (get_portfolio, submit_decision) and",
  "Taostats tools for researching the network.",
  "",
  "GOAL: Research subnets, build conviction, and allocate capital",
  "into the best risk/reward alpha positions.",
  "",
  "RESEARCH PHASE:",
  "- Start with get_portfolio to see current holdings",
  "- Use Taostats tools to research subnets. Consider:",
  "  * Price action and momentum — this is often the biggest driver",
  "  * Pool dynamics — TAO reserve, alpha reserve, liquidity",
  "  * Subnet yield — holding alpha earns emissions (a bonus,",
  "    not the primary reason to enter a position)",
  "  * Subnet fundamentals — what it does, health, activity",
  "- Think holistically. Don't just sort by one metric.",
  "- Be efficient with tool calls — you have limited steps.",
  "",
  "TRADING PHASE:",
  "- You MUST call submit_decision as your FINAL action.",
  "  If you fail to submit, the entire run is wasted.",
  "- Routes: USD<->TAO, TAO<->ALPHA, USD<->ALPHA, ALPHA<->ALPHA",
  "- Deploy most idle USD into alpha positions (keep some reserve)",
  "- Diversify across a few subnets — don't go all-in on one",
  "- Consider rebalancing existing holdings based on new research",
].join("\n");

export async function runStrategy(tools: ToolSet): Promise<void> {
  const { steps } = await generateText({
    model: xai("grok-4-1-fast-non-reasoning"),
    tools,
    stopWhen: stepCountIs(12),
    system: SYSTEM_PROMPT,
    prompt:
      "Check your portfolio, research subnets, then submit trades.",
    onStepFinish({ toolCalls }) {
      for (const call of toolCalls) {
        console.log(`[strategy] Tool: ${call.toolName}`);
      }
    },
  });
  console.log(`[strategy] Completed in ${String(steps.length)} steps`);
}
