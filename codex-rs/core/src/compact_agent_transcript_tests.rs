use super::*;
use crate::compact::SUMMARY_PREFIX;
use codex_history::CodexHarnessMetadata;
use codex_history::CompactedItem;
use codex_protocol::ResponseItemId;
use codex_protocol::models::ContentItem;
use pretty_assertions::assert_eq;

fn message(id: &str, role: &str, text: &str) -> RolloutItem {
    RolloutItem::ResponseItem(ResponseItemEnvelope::new(ResponseItem::Message {
        encrypted_content: None,
        status: None,
        id: Some(ResponseItemId::from_server(id.to_string())),
        role: role.to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }))
}

fn tool_call(id: &str, call_id: &str) -> RolloutItem {
    RolloutItem::ResponseItem(ResponseItemEnvelope::new(ResponseItem::FunctionCall {
        encrypted_content: None,
        status: None,
        id: Some(ResponseItemId::from_server(id.to_string())),
        name: "shell".to_string(),
        namespace: None,
        arguments: r#"{"command":"cargo test"}"#.to_string(),
        encrypted_function_args: None,
        call_id: call_id.to_string(),
        internal_chat_message_metadata_passthrough: None,
    }))
}

fn tool_output(id: &str, call_id: &str, text: &str) -> RolloutItem {
    RolloutItem::ResponseItem(ResponseItemEnvelope::new(
        ResponseItem::FunctionCallOutput {
            id: Some(ResponseItemId::from_server(id.to_string())),
            call_id: Some(call_id.to_string()),
            name: None,
            namespace: None,
            output: FunctionCallOutputPayload::from_text(text.to_string()),
            internal_chat_message_metadata_passthrough: None,
        },
    ))
}

fn reasoning(id: &str) -> RolloutItem {
    RolloutItem::ResponseItem(ResponseItemEnvelope::new(ResponseItem::Reasoning {
        id: Some(ResponseItemId::from_server(id.to_string())),
        summary: Vec::new(),
        content: None,
        encrypted_content: Some("opaque".to_string()),
        internal_chat_message_metadata_passthrough: None,
    }))
}

fn compacted() -> RolloutItem {
    RolloutItem::Compacted(CompactedItem {
        message: String::new(),
        replacement_history: None,
        guardian_history: None,
        retained_context: None,
        mcp_resource_origins: None,
        window_number: None,
        first_window_id: None,
        previous_window_id: None,
        window_id: None,
        compaction_response_id: None,
        latest_token_usage_record: None,
        resume_metadata: None,
    })
}

fn ids(items: &[ResponseItemEnvelope]) -> Vec<&str> {
    items
        .iter()
        .map(|item| item.item.id().expect("kept items have ids").as_str())
        .collect()
}

fn sample_transcript() -> AgentTranscript {
    AgentTranscript::build(
        &[
            message("msg_dev", "developer", "sandbox instructions"),
            message("msg_1", "user", "fix the failing test"),
            reasoning("rs_1"),
            tool_call("fc_1", "call_1"),
            tool_output("fco_1", "call_1", "test foo ... FAILED"),
            compacted(),
            message(
                "msg_summary",
                "user",
                &format!("{SUMMARY_PREFIX}\nan earlier summary"),
            ),
            // Retained from the first window and re-persisted after the checkpoint.
            message("msg_1", "user", "fix the failing test"),
            message("msg_2", "assistant", "the test now passes"),
        ],
        TruncationPolicy::Tokens(10_000),
    )
}

#[test]
fn transcript_aliases_keepable_items_once_in_log_order() {
    let transcript = sample_transcript();

    assert_eq!(transcript.len(), 4);
    let rendered = transcript.rendered();
    assert!(rendered.contains("=== #1 user · ~"), "{rendered}");
    assert!(rendered.contains("fix the failing test"), "{rendered}");
    assert!(
        rendered.contains("=== #2 tool call `shell` · call_id=call_1 · ~"),
        "{rendered}"
    );
    assert!(
        rendered.contains("=== #3 tool output · call_id=call_1 · ~"),
        "{rendered}"
    );
    assert!(rendered.contains(COMPACTION_BOUNDARY), "{rendered}");
    assert!(rendered.contains("=== #4 assistant · ~"), "{rendered}");
    assert!(!rendered.contains("sandbox instructions"), "{rendered}");
    assert!(!rendered.contains("an earlier summary"), "{rendered}");
    assert_eq!(rendered.matches("fix the failing test").count(), 1);
}

#[test]
fn keep_list_adds_tool_partners_and_sorts_by_log_order() {
    let transcript = sample_transcript();

    let kept = transcript
        .resolve_keep_list("#4\n#3\n", /*token_budget*/ 10_000)
        .expect("valid keep list");

    assert_eq!(ids(&kept), vec!["fc_1", "fco_1", "msg_2"]);
}

#[test]
fn keep_list_accepts_ranges_and_bare_numbers() {
    let transcript = sample_transcript();

    let kept = transcript
        .resolve_keep_list("  1 - 2 \n\n#4", /*token_budget*/ 10_000)
        .expect("valid keep list");

    assert_eq!(ids(&kept), vec!["msg_1", "fc_1", "fco_1", "msg_2"]);
}

#[test]
fn keep_list_reports_every_invalid_line() {
    let transcript = sample_transcript();

    let errors = transcript
        .resolve_keep_list("#2\n#9\nlast item\n#3-#1", /*token_budget*/ 10_000)
        .expect_err("invalid keep list");

    assert_eq!(
        errors,
        vec![
            "`#9`: the transcript only has items #1-#4".to_string(),
            "`last item`: expected `#N` or `#N-#M`, one entry per line".to_string(),
            "`#3-#1`: range start is after its end".to_string(),
        ]
    );
}

#[test]
fn keep_list_enforces_token_budget_including_partners() {
    let transcript = sample_transcript();

    let errors = transcript
        .resolve_keep_list("#3", /*token_budget*/ 1)
        .expect_err("over budget");

    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("over the budget of 1"), "{errors:?}");
}

#[test]
fn keep_list_drops_tool_calls_without_outputs() {
    let transcript = AgentTranscript::build(
        &[
            message("msg_1", "user", "run it"),
            tool_call("fc_1", "call_1"),
        ],
        TruncationPolicy::Tokens(10_000),
    );

    let kept = transcript
        .resolve_keep_list("#1-#2", /*token_budget*/ 10_000)
        .expect("valid keep list");

    assert_eq!(ids(&kept), vec!["msg_1"]);
}

#[test]
fn kept_tool_outputs_use_live_history_truncation() {
    let long_output = "line of build output\n".repeat(5_000);
    let RolloutItem::ResponseItem(mut output) = tool_output("fco_1", "call_1", &long_output) else {
        unreachable!();
    };
    output.metadata = Some(CodexHarnessMetadata {
        history_truncation_token_limit: Some(200),
        ..Default::default()
    });
    let transcript = AgentTranscript::build(
        &[
            tool_call("fc_1", "call_1"),
            RolloutItem::ResponseItem(output),
        ],
        TruncationPolicy::Tokens(10_000),
    );

    // The agent reads the full output, but the kept copy matches what live history holds.
    assert_eq!(
        transcript
            .rendered()
            .matches("line of build output")
            .count(),
        5_000
    );
    let kept = transcript
        .resolve_keep_list("#2", /*token_budget*/ 10_000)
        .expect("truncated output fits the budget");
    let ResponseItem::FunctionCallOutput { output, .. } = &kept[1].item else {
        panic!("expected the tool output second");
    };
    let kept_text = output.text_content().expect("text output");
    assert!(
        kept_text.len() < long_output.len() / 10,
        "{}",
        kept_text.len()
    );
}
