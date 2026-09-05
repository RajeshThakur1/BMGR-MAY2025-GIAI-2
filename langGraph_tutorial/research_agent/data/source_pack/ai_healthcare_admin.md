# Healthcare Administrative Agents

A hospital operations team explored AI agents for appointment scheduling, referral summarization, and insurance pre-authorization support. Clinical diagnosis was deliberately out of scope. The highest-value use case was administrative summarization: turning long referral notes into structured intake summaries for staff review.

Privacy controls shaped the architecture. Patient identifiers were removed before the LLM step whenever possible, and the workflow logged which fields were sent to the model. Staff could regenerate a summary, but the system preserved the original version and the edited version for audit purposes.

A fact-checking step reduced risk. The reviewer compared the generated summary against the source note and flagged missing medications, dates, or constraints. The system was not allowed to invent clinical recommendations. It could summarize, structure, and highlight missing information, but final interpretation stayed with qualified staff.

The final design used clear boundaries: de-identify first, summarize second, verify third, and require human approval before the summary entered the operational record. Success metrics included time saved per referral, correction rate, missing-field rate, and staff satisfaction.
