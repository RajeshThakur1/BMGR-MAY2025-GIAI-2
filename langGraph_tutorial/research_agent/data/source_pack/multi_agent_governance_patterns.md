# Multi-Agent Governance Patterns

Multi-agent systems fail less often because one agent is weak and more often because handoffs are ambiguous. A Researcher may produce a paragraph when the Writer needs evidence fields. A Writer may produce persuasive language that the Fact-Checker cannot map back to sources. An Editor may polish a claim until the uncertainty disappears.

Structured messages reduce this problem. Each handoff should include sender, receiver, task, payload, confidence, and trace ID. The payload should use a schema that makes required fields explicit. When a message fails validation, the system can stop, repair, or request a revised output before the next agent acts on bad data.

Team state is the shared memory of the workflow. It stores the user query, source IDs, messages, intermediate outputs, revision count, and final answer. Without shared state, debugging becomes a matter of reading a long transcript and guessing where the failure began.

Review loops need stop conditions. A critic agent should provide concrete revision instructions, but the orchestrator should limit retry count and define what level of reliability is enough. Otherwise, the system can enter an expensive loop where agents keep rewriting without improving the answer.
