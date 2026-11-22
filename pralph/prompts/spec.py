SPEC_INITIAL_PROMPT = """\
# Spec Authoring Mode

You are writing the specification for a spec-driven project. The spec lives in the
repository, is committed alongside the code, and is the single source of truth:
stories are derived from it and every implementation is checked against it.

## Request

{{user_prompt}}

## Files to create

- `{{spec_file}}` — WHAT and WHY: requirements. No implementation detail.
- `{{design_file}}` — HOW: the technical design that satisfies the requirements.
{{guardrails_instruction}}
{{existing_inputs}}
## Instructions

1. Inspect the repository (Glob, Grep, Read) so the spec fits the code that already exists
2. Use WebSearch to research best practices for this type of project
3. Write `{{spec_file}}` following the structure below
4. Write `{{design_file}}` following the structure below
5. Provide a brief summary of what you created in <changes_summary>...</changes_summary> tags

## Structure of spec.md

```markdown
# <Name> — Specification

## Overview
Problem, goals, and why this matters. Who it is for.

## User scenarios
The primary journeys, in priority order.

## Requirements

### REQ-001: <Short title>
<One capability, stated from the user's point of view or as "The system SHALL ...">

**Acceptance criteria**
- Given <context>, when <action>, then <observable outcome>
- <another testable condition>

### REQ-002: <Short title>
...

## Non-functional requirements

### NFR-001: <Short title>
<Performance, security, accessibility, compliance ... with a measurable threshold>

**Acceptance criteria**
- ...

## Out of scope
What this spec deliberately does not cover.

## Open questions
Anything unresolved. Mark the affected requirement with `[NEEDS CLARIFICATION: question]`.

## Changelog
- {{today}}: Initial specification.
```

### Requirement rules

- Every requirement heading has the exact form `### REQ-001: Short title` (non-functional:
  `### NFR-001: Short title`). pralph parses these headings — keep the format exact.
- IDs are sequential, unique, and permanent. They are referenced by stories, commits and tickets.
- One independently testable capability per requirement. Split anything that needs "and".
- Every requirement has acceptance criteria that can be judged true or false.
- Describe behaviour, not implementation. No frameworks, tables or endpoints in spec.md.
- Do not guess silently: mark uncertainty with `[NEEDS CLARIFICATION: ...]` and list it
  under "Open questions".

## Structure of design.md

```markdown
# <Name> — Design

## Architecture overview
## Components            (for each: responsibility, and the requirement IDs it satisfies)
## Data model            (if applicable)
## Interfaces / APIs     (if applicable)
## Security considerations
## Testing strategy
## Decisions             (decision, alternatives considered, rationale)
```

Reference requirement IDs wherever a design element exists because of a requirement.

## Research Notes

Save your web search findings to: {{research_notes_file}}
Use Write to create this file with a summary of what you searched for, key findings, \
URLs, and how they informed the spec. This prevents duplicate research in later iterations.

## Completion Signal

This is the first iteration — do NOT signal completion yet. There will be refinement iterations after this.
"""

SPEC_ITERATION_PROMPT = """\
# Spec Refinement Mode

You are working on the in-repo specification of a spec-driven project. The spec is the
single source of truth: stories are derived from it and implementations are checked
against it. This is iteration {{current}} of {{total}}.

## Request

{{user_prompt}}

If the request describes a change (a new feature, a changed rule, something to drop),
your job is to make the spec say so. Otherwise, improve the spec's completeness and quality.

## Files

- Requirements: `{{spec_file}}`
- Design: `{{design_file}}`
- Research notes: `{{research_notes_file}}`
{{other_specs}}
## Instructions

1. Read the research notes (if the file exists), then the spec and the design
2. Compare them against the request and the actual codebase. Look for: parts of the
   request the spec does not cover yet, contradictions, vague or untestable requirements,
   missing acceptance criteria, `[NEEDS CLARIFICATION]` markers you can now resolve,
   design sections that no longer match the requirements
3. If research is needed on a NEW topic (not already in the research notes), use WebSearch
   and append the findings to the research notes
4. Use Edit to make targeted changes. Keep spec.md about WHAT/WHY and design.md about HOW,
   and keep the two consistent
5. Record every requirement-level change in the `## Changelog` section of spec.md
   (today is {{today}}): which IDs were added, changed or removed, and why
6. Provide a brief summary of changes in <changes_summary>...</changes_summary> tags

Do NOT output the documents in your response. Edit the files directly.

## Requirement ID rules — CRITICAL

- Requirement headings keep the exact form `### REQ-001: Short title` / `### NFR-001: Short title`
- NEVER renumber or reuse an ID. Existing stories, commits and tickets reference them
- A new requirement gets the next unused number
- A removed requirement: delete its section and note the removal in the Changelog
- Only change the text of an existing requirement when its meaning changes or it was
  genuinely unclear. pralph detects edited requirements and schedules follow-up work for
  them — do not reword requirements for style

## Completion Signal

When the request is fully reflected in the spec and the spec is complete — every
requirement testable, design consistent with the requirements, no gaps you can close —
output `[PLANNING_COMPLETE]` at the end of your response.

If there is still something to improve, make your changes and do NOT output the signal.
"""

SPEC_EXTRACT_PROMPT = """\
# Story Extraction Mode (spec-driven)

You are breaking a specification down into implementable user stories. The specification
lives in the repository and is the source of truth: every story must trace back to the
requirement(s) it implements.

## Specifications

{{specs_overview}}

{{spec_content}}

## Requirements needing stories

{{pending_requirements}}

## Existing Stories (DO NOT DUPLICATE)

Total stories generated so far: {{total_stories}}

### Previously Generated Stories
Reference these IDs when specifying dependencies:
{{existing_stories}}

### Category Statistics
Use these to assign the next available ID for each category:
{{category_stats}}

## Your Task

1. Read the spec and design files listed above with the Read tool
2. For every requirement under "Requirements needing stories":
   - **new** requirement — generate the stories needed to implement it completely
   - **changed** requirement — the spec text was edited after its stories were written.
     Compare the current text with the existing stories and generate follow-up stories
     for the difference only. If the edit changes nothing that needs building, generate nothing
3. Then check the other requirements: where the existing stories do not cover all of a
   requirement's acceptance criteria, add the missing stories
4. Use the design to shape the work — foundation and infrastructure first. Infrastructure
   stories trace to the requirements they enable
5. For each story, provide:
   - **ID**: Use format CATEGORY-NNN (see category stats for next number)
   - A clear title
   - User story format: "As a [user], I want [feature] so that [benefit]"
   - Acceptance criteria (testable conditions, derived from the requirement's criteria)
   - Priority (1-5, where 1 is highest)
   - Category (uppercase: AUTH, API, DB, UI, INFRA, etc.)
   - Estimated complexity (small, medium, large)
   - **Dependencies**: Array of story IDs this depends on
   - **spec**: Name of the spec the story belongs to
   - **requirements**: Array of requirement IDs the story implements (at least one)

## Output Format

Return your findings as JSON:
```json
{
  "stories": [
    {
      "id": "AUTH-043",
      "title": "Short descriptive title",
      "content": "As a user, I want to...",
      "acceptance_criteria": ["Criterion 1", "Criterion 2"],
      "priority": 1,
      "category": "AUTH",
      "complexity": "medium",
      "dependencies": ["AUTH-001", "AUTH-002"],
      "spec": "{{example_spec}}",
      "requirements": ["REQ-001", "REQ-002"]
    }
  ],
  "notes": "Observations, including anything the spec seems to be missing"
}
```

## Spec-driven rules

- No story without a requirement. If you believe something is missing from the spec, say
  so in "notes" — the spec has to be amended first, do not invent scope here
- Only use requirement IDs that exist in the specs listed above
- A story may implement several requirements, and a requirement may need several stories
- Each story should be independently implementable (given its dependencies)

## ID Assignment Rules

1. Look at category_stats to find the next available ID for each category
2. If category "AUTH" has next_id: 43, use "AUTH-043" for your first AUTH story
3. If you generate multiple stories in the same category, increment: AUTH-043, AUTH-044, etc.
4. For new categories not in category_stats, start at 001

## Dependency Guidelines

- **Infrastructure/foundation stories** typically have NO dependencies (they are the foundation)
- **Feature stories** depend on their infrastructure
- **ONLY reference story IDs** that exist in "Previously Generated Stories" above or are
  other stories in THIS batch

## Completion Signal — CRITICAL RULES

When every requirement is fully covered by stories and you have NOTHING new to add, return
an empty stories array and output the signal on its own line:

```json
{"stories": [], "notes": "All requirements are covered."}
```

[GENERATION_COMPLETE]

**IMPORTANT:**
- NEVER output `[GENERATION_COMPLETE]` if you are returning new stories in this response
- NEVER mention, reference, or discuss `[GENERATION_COMPLETE]` in your text — do not explain
  whether you are or are not emitting it
- The signal must appear ALONE on its own line, not inside a sentence
- Either output the signal (when truly done) or don't — say nothing about it
"""

SPEC_IMPLEMENT_SECTION = """\

## Specification (source of truth)

This project is spec-driven. The specification is part of the repository and is the
source of truth for what the software does:

{{spec_files}}

{{story_requirements}}

### Rules for spec-driven work

1. Read the spec and the design before writing code. Build what the spec says — the
   acceptance criteria of the requirements above must hold when you are done
2. Code and spec must never disagree. If the spec turns out to be wrong, ambiguous or
   incomplete for this story, fix the spec (and the design, if affected) in the same
   commit, and add a line to the `## Changelog` section of spec.md saying what changed and why
3. Never renumber or reuse requirement IDs, and do not reword requirements you are not changing
4. Stay inside this story. New ideas belong in the spec's "Open questions", not in the code
5. The spec files are part of the repository. If any of them are untracked or have
   uncommitted changes when you commit, include them in your commit
{{tasks_rule}}
"""

SPEC_REVIEW_SECTION = """\
## Specification Conformance

This project is spec-driven: the specification in the repository is the source of truth.

{{spec_files}}

{{story_requirements}}

In addition to the criteria below:
- Verify the acceptance criteria of each requirement listed above against the code
- Reject (`critical`) if the implementation contradicts spec.md or design.md and the
  spec was not updated in the same change — code and spec must not diverge
- Reject (`major`) if requirement IDs were renumbered or unrelated requirements were reworded
"""
