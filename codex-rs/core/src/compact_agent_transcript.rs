//! Renders the full rollout as an aliased transcript for agent compaction and resolves the
//! agent's keep-list back to the original items.
//!
//! The transcript is built from every persisted response item, including items from earlier
//! context windows, so each compaction starts from the original trajectory rather than from a
//! previous summary. Only items that may be carried into a new window receive an alias.

use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Write as _;

use codex_history::ResponseItemEnvelope;
use codex_history::RolloutItem;
use codex_protocol::items::TurnItem;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::LocalShellAction;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::plaintext_agent_message_content;
use codex_utils_audio::estimate_audio_token_count;
use codex_utils_output_truncation::TruncationPolicy;
use codex_utils_output_truncation::truncate_function_output_payload;
use codex_utils_output_truncation::with_serialization_allowance;

use crate::compact::content_items_to_text;
use crate::compact::is_summary_message;
use crate::context_manager::estimate_item_token_count;

const COMPACTION_BOUNDARY: &str = "--- context compacted here ---";

pub(crate) struct TranscriptEntry {
    /// The item as it would be installed: tool outputs carry the live-history truncation.
    pub(crate) kept: ResponseItemEnvelope,
    pub(crate) kept_tokens: i64,
}

pub(crate) struct AgentTranscript {
    // Entry `i` has alias `#{i + 1}`.
    entries: Vec<TranscriptEntry>,
    rendered: String,
}

impl AgentTranscript {
    pub(crate) fn build(rollout_items: &[RolloutItem], policy: TruncationPolicy) -> Self {
        let mut entries = Vec::new();
        let mut rendered = String::new();
        let mut seen_ids = HashSet::new();
        for rollout_item in rollout_items {
            let envelope = match rollout_item {
                RolloutItem::ResponseItem(envelope) => envelope,
                RolloutItem::Compacted(_) => {
                    let _ = writeln!(rendered, "\n{COMPACTION_BOUNDARY}\n");
                    continue;
                }
                _ => continue,
            };
            if !is_keepable(&envelope.item) {
                continue;
            }
            // Retained items are re-persisted inside later checkpoints; show each item once.
            if let Some(id) = envelope.item.id()
                && !seen_ids.insert(id.clone())
            {
                continue;
            }
            let kept = kept_form(envelope, policy);
            let kept_tokens = estimate_item_token_count(&kept.item);
            let alias = entries.len() + 1;
            let _ = writeln!(
                rendered,
                "=== #{alias} {} · ~{kept_tokens} tokens ===\n{}\n",
                item_header(&envelope.item),
                item_body(&envelope.item).trim_end(),
            );
            entries.push(TranscriptEntry { kept, kept_tokens });
        }
        Self { entries, rendered }
    }

    pub(crate) fn rendered(&self) -> &str {
        &self.rendered
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Resolves `keep.txt` to kept items in log order, or returns every problem found.
    ///
    /// Tool calls and their outputs are kept together; a pair whose other half is not in the
    /// transcript is dropped because the Responses API rejects unpaired calls and outputs.
    pub(crate) fn resolve_keep_list(
        &self,
        keep: &str,
        token_budget: i64,
    ) -> Result<Vec<ResponseItemEnvelope>, Vec<String>> {
        let mut errors = Vec::new();
        let mut selected = BTreeSet::new();
        for line in keep.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match parse_keep_line(line) {
                Some((start, end)) if start <= end && start >= 1 && end <= self.len() => {
                    selected.extend(start - 1..end);
                }
                Some((start, end)) if start > end => {
                    errors.push(format!("`{line}`: range start is after its end"));
                }
                Some(_) => errors.push(format!(
                    "`{line}`: the transcript only has items #1-#{}",
                    self.len()
                )),
                None => errors.push(format!(
                    "`{line}`: expected `#N` or `#N-#M`, one entry per line"
                )),
            }
        }
        if !errors.is_empty() {
            return Err(errors);
        }

        let mut by_call_id: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, entry) in self.entries.iter().enumerate() {
            if let Some(call_id) = call_id(&entry.kept.item) {
                by_call_id.entry(call_id).or_default().push(index);
            }
        }
        let partners = selected
            .iter()
            .filter_map(|&index| call_id(&self.entries[index].kept.item))
            .flat_map(|call_id| by_call_id.get(call_id).into_iter().flatten().copied())
            .collect::<Vec<_>>();
        selected.extend(partners);
        selected.retain(|&index| {
            let Some(call_id) = call_id(&self.entries[index].kept.item) else {
                return true;
            };
            let group = &by_call_id[call_id];
            let has_call = group
                .iter()
                .any(|&i| is_tool_call(&self.entries[i].kept.item));
            let has_output = group
                .iter()
                .any(|&i| !is_tool_call(&self.entries[i].kept.item));
            has_call && has_output
        });

        let kept_tokens = selected
            .iter()
            .map(|&index| self.entries[index].kept_tokens)
            .sum::<i64>();
        if kept_tokens > token_budget {
            return Err(vec![format!(
                "the kept items total ~{kept_tokens} tokens (tool calls and outputs are counted \
                 together), over the budget of {token_budget}; keep fewer items"
            )]);
        }
        Ok(selected
            .into_iter()
            .map(|index| self.entries[index].kept.clone())
            .collect())
    }
}

/// Items that can be carried into a new window verbatim.
///
/// Initial context is rebuilt for every window, summaries are never carried forward, and
/// reasoning is tied to the model that produced it, so none of those are offered.
fn is_keepable(item: &ResponseItem) -> bool {
    match item {
        ResponseItem::Message { role, .. } => match role.as_str() {
            "user" => matches!(
                crate::event_mapping::parse_turn_item(item),
                Some(TurnItem::UserMessage(user)) if !is_summary_message(&user.message())
            ),
            "assistant" => true,
            _ => false,
        },
        ResponseItem::AgentMessage { .. }
        | ResponseItem::LocalShellCall { .. }
        | ResponseItem::FunctionCall { .. }
        | ResponseItem::FunctionCallOutput { .. }
        | ResponseItem::CustomToolCall { .. }
        | ResponseItem::CustomToolCallOutput { .. }
        | ResponseItem::ToolSearchCall { .. }
        | ResponseItem::ToolSearchOutput { .. }
        | ResponseItem::WebSearchCall { .. } => true,
        ResponseItem::AdditionalTools { .. }
        | ResponseItem::Reasoning { .. }
        | ResponseItem::ImageGenerationCall { .. }
        | ResponseItem::Compaction { .. }
        | ResponseItem::ConfigurationUpdate { .. }
        | ResponseItem::CompactionTrigger { .. }
        | ResponseItem::ContextCompaction { .. }
        | ResponseItem::Other => false,
    }
}

/// Applies the same tool-output truncation that live history applies when recording.
fn kept_form(envelope: &ResponseItemEnvelope, policy: TruncationPolicy) -> ResponseItemEnvelope {
    let mut kept = envelope.clone();
    if let ResponseItem::FunctionCallOutput { output, .. }
    | ResponseItem::CustomToolCallOutput { output, .. } = &mut kept.item
    {
        let policy = envelope
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.history_truncation_token_limit)
            .map(TruncationPolicy::Tokens)
            .unwrap_or_else(|| with_serialization_allowance(policy));
        truncate_function_output_payload(output, policy, estimate_audio_token_count);
    }
    kept
}

fn call_id(item: &ResponseItem) -> Option<&str> {
    match item {
        ResponseItem::FunctionCall { call_id, .. }
        | ResponseItem::CustomToolCall { call_id, .. }
        | ResponseItem::CustomToolCallOutput { call_id, .. } => Some(call_id.as_str()),
        ResponseItem::FunctionCallOutput { call_id, .. }
        | ResponseItem::LocalShellCall { call_id, .. }
        | ResponseItem::ToolSearchCall { call_id, .. }
        | ResponseItem::ToolSearchOutput { call_id, .. } => call_id.as_deref(),
        _ => None,
    }
}

fn is_tool_call(item: &ResponseItem) -> bool {
    matches!(
        item,
        ResponseItem::FunctionCall { .. }
            | ResponseItem::CustomToolCall { .. }
            | ResponseItem::LocalShellCall { .. }
            | ResponseItem::ToolSearchCall { .. }
    )
}

fn parse_keep_line(line: &str) -> Option<(usize, usize)> {
    let entry = line.split_whitespace().collect::<String>();
    let parse_alias = |alias: &str| alias.strip_prefix('#').unwrap_or(alias).parse().ok();
    match entry.split_once('-') {
        Some((start, end)) => Some((parse_alias(start)?, parse_alias(end)?)),
        None => {
            let alias = parse_alias(&entry)?;
            Some((alias, alias))
        }
    }
}

fn item_header(item: &ResponseItem) -> String {
    let with_call_id = |label: String| match call_id(item) {
        Some(call_id) => format!("{label} · call_id={call_id}"),
        None => label,
    };
    match item {
        ResponseItem::Message { role, .. } => role.clone(),
        ResponseItem::AgentMessage {
            author, recipient, ..
        } => format!("agent message {author} -> {recipient}"),
        ResponseItem::FunctionCall {
            name, namespace, ..
        } => with_call_id(match namespace {
            Some(namespace) => format!("tool call `{namespace}.{name}`"),
            None => format!("tool call `{name}`"),
        }),
        ResponseItem::CustomToolCall { name, .. } => with_call_id(format!("tool call `{name}`")),
        ResponseItem::LocalShellCall { .. } => with_call_id("shell call".to_string()),
        ResponseItem::ToolSearchCall { .. } => with_call_id("tool search".to_string()),
        ResponseItem::FunctionCallOutput { .. }
        | ResponseItem::CustomToolCallOutput { .. }
        | ResponseItem::ToolSearchOutput { .. } => with_call_id("tool output".to_string()),
        ResponseItem::WebSearchCall { .. } => "web search".to_string(),
        _ => "item".to_string(),
    }
}

fn item_body(item: &ResponseItem) -> String {
    match item {
        ResponseItem::Message { content, .. } => {
            let images = content
                .iter()
                .filter(|part| {
                    matches!(part, codex_protocol::models::ContentItem::InputImage { .. })
                })
                .count();
            let mut text = content_items_to_text(content).unwrap_or_default();
            if images > 0 {
                let _ = write!(text, "\n[{images} image(s)]");
            }
            text
        }
        ResponseItem::AgentMessage { content, .. } => plaintext_agent_message_content(content)
            .unwrap_or_else(|| "[encrypted content]".to_string()),
        ResponseItem::FunctionCall { arguments, .. } => arguments.clone(),
        ResponseItem::CustomToolCall { input, .. } => input.clone(),
        ResponseItem::LocalShellCall { action, .. } => match action {
            LocalShellAction::Exec(exec) => exec.command.join(" "),
        },
        ResponseItem::ToolSearchCall { arguments, .. } => arguments.to_string(),
        ResponseItem::FunctionCallOutput { output, .. }
        | ResponseItem::CustomToolCallOutput { output, .. } => output_text(output),
        ResponseItem::ToolSearchOutput { tools, .. } => format!("[{} tool(s) loaded]", tools.len()),
        ResponseItem::WebSearchCall { action, .. } => action
            .as_ref()
            .and_then(|action| serde_json::to_string(action).ok())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn output_text(output: &FunctionCallOutputPayload) -> String {
    if let Some(text) = output.text_content() {
        return text.to_string();
    }
    output
        .content_items()
        .unwrap_or_default()
        .iter()
        .map(|part| match part {
            FunctionCallOutputContentItem::InputText { text } => text.as_str(),
            FunctionCallOutputContentItem::InputImage { .. } => "[image]",
            FunctionCallOutputContentItem::InputAudio { .. } => "[audio]",
            FunctionCallOutputContentItem::EncryptedContent { .. } => "[encrypted content]",
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
#[path = "compact_agent_transcript_tests.rs"]
mod tests;
