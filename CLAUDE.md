# Science Repo Guidelines

## Contribution Workflow (Open-Source Scientific Projects)

### Commit Messages
- **NEVER add AI/LLM attribution to commit messages.** No "Co-Authored-By: Claude", no "Generated with Claude Code", no AI mentions of any kind. The scientific community is hostile to LLM contributions and this will get PRs rejected.
- Write commit messages as if a human wrote them. Focus on what changed and why.

### Git Permissions
- The `gh` CLI token does not have repo write (push/fork) permissions.
- After committing locally, **always ask the user to push**. Do not attempt `git push` yourself.
- For forking, **ask the user to fork manually** and provide the fork URL. Then set up remotes accordingly.

### PR Workflow
1. Clone the upstream repo to `/tmp/`
2. Ask the user to fork on GitHub and provide the fork URL
3. Set up remotes: `origin` = fork, `upstream` = original
4. Create a feature branch, make changes, commit
5. Ask the user to push: `cd /tmp/<repo> && git push -u origin <branch>`
6. Once pushed, create the PR via `gh pr create` (read access works for this)

### Issue Selection
- **Only target active repos.** Must have commits or meaningful issue activity within the last 4 months. Dormant projects waste effort.
- **Use judgement on whether an issue is something we can/should go after.** Labels like `help-wanted` are a signal but not required.
- **Never insert yourself into conversations between maintainers and specific people/orgs.** If an issue is a direct offer from one org to another (e.g., a HuggingFace employee writing to a repo maintainer), that's their conversation, not an open call.
- **Read the social context.** Who opened it, who it's addressed to, whether outside contributions make sense.
- When in doubt, ask the user before commenting or starting work.

### Contribution Style
- Read issues carefully. Match the maintainer's intent, not your interpretation.
- Comment on the issue before starting work if there are open questions.
- Keep PRs focused. One issue per PR.
- Run the project's test/lint/build commands before considering the work done.
- Preserve author attributions when moving content between files.

## Writing Voice (for comments, PRs, issues, docs)

All written output for external-facing contributions must match this style:

### Tone
- Concise, direct, technically confident. No fluff.
- Conversational but professional. Not corporate.
- State things directly; don't hedge with "I think maybe" or "it seems like perhaps."
- When uncertain, ask a concrete question: "Could you confirm X?" not "I was wondering if..."

### Structure
- Short sentences. Fragments when context is clear: "awesome thanks", "updated"
- Lead with action/conclusion, not setup
- Use colons to introduce lists, not subordinate clauses
- Markdown headers (##) to organize longer comments
- Tables and code blocks over prose when structured data is clearer
- Numbered lists for proposals; bullets for features/details
- Close with clear next action or don't close at all

### Commit Messages
- Imperative mood: "Add", "Fix", "Update", "Implement"
- ~50 char subject, no period
- Body explains "what" concisely when needed
- Examples: "Add anomaly detection pipeline for Gaia CV Hunter", "Fix persistent SyntaxErrors in code block parsing"

### PR Descriptions
- One-line summary up top
- Bullet points over paragraphs
- "### Changes" and "### Checklist" sections
- Reference issues with links
- Keep "why" brief, focus on "what"

### Never Use
- Em dashes
- "I believe", "perhaps", "it seems", "in order to", "utilize"
- Trailing "Happy to..." sentences in PRs or issue comments (dead AI giveaway)
- Apologies for things that aren't your fault
- Marketing language or superlatives
- AI-sounding phrases: "I'd be happy to help!", "Great question!"
