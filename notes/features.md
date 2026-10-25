Yes—two user-facing features on top of the engine:

**1. Review findings**

Users inspect the quotes and sources, then **confirm or dismiss** each finding.

Add these fields to `findings`:

```json
{
  "review_status": "unreviewed",
  "review_note": null
}
```

`review_status` can be `unreviewed`, `confirmed`, or `dismissed`. This records the human decision separately from the LLM’s judgment.

**2. Chat with documents**

Users can ask:

- General questions about the documents.
- Questions about detected conflicts.
- Follow-ups such as “Show me the full clause.”

Chat retrieves information from your **chunks, facts, and findings**, then answers with source citations.

So your app has three main parts: **scan, review, and chat**.
