use anyhow::Result;
use codex_core::compact::SUMMARIZATION_PROMPT;
use codex_core::compact::SUMMARY_PREFIX;
use codex_features::Feature;
use codex_model_provider_info::ModelProviderInfo;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed_with_tokens;
use core_test_support::responses::ev_function_call;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::path::Path;
use std::path::PathBuf;
use wiremock::MockServer;

const USER_GOAL: &str = "AGENT_COMPACTION_USER_GOAL";
const DROPPED_REPLY: &str = "AGENT_COMPACTION_DROPPED_REPLY";
const AGENT_SUMMARY: &str = "AGENT_COMPACTION_SUMMARY";

fn local_compaction_provider(server: &MockServer) -> ModelProviderInfo {
    let mut provider = built_in_model_providers(/*openai_base_url*/ None)["openai"].clone();
    provider.name = "OpenAI-compatible test provider".to_string();
    provider.base_url = Some(format!("{}/v1", server.uri()));
    provider.supports_websockets = false;
    provider
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn manual_agent_compaction_installs_summary_and_kept_items() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let write_output =
        format!("printf '{AGENT_SUMMARY}\\n' > summary.md && printf '#1\\n' > keep.txt");
    let request_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("parent-reply", DROPPED_REPLY),
                ev_completed_with_tokens("parent-response", /*total_tokens*/ 100),
            ]),
            // The compaction agent writes both files, then confirms.
            sse(vec![
                ev_function_call(
                    "write-output",
                    "exec_command",
                    &json!({ "cmd": write_output }).to_string(),
                ),
                ev_completed_with_tokens("agent-tool", /*total_tokens*/ 200),
            ]),
            sse(vec![
                ev_assistant_message("agent-done", "Both files are written."),
                ev_completed_with_tokens("agent-done-response", /*total_tokens*/ 250),
            ]),
            sse(vec![
                ev_assistant_message("after-compaction", "continuing"),
                ev_completed_with_tokens("after-compaction-response", /*total_tokens*/ 120),
            ]),
        ],
    )
    .await;
    let test = build_agent_compaction_codex(&server, /*auto_compact_token_limit*/ None).await?;

    test.submit_turn(USER_GOAL).await?;
    test.codex.submit(Op::Compact).await?;
    assert_eq!(
        warnings_until_turn_complete(&test).await,
        Vec::<String>::new()
    );
    test.submit_turn("keep going").await?;

    let requests = request_log.requests();
    assert_eq!(requests.len(), 4);

    let agent_request = &requests[1];
    assert!(
        agent_request
            .instructions_text()
            .starts_with("You are the compaction agent"),
        "{}",
        agent_request.instructions_text()
    );
    assert!(
        agent_request
            .message_input_texts("user")
            .iter()
            .any(|text| text.starts_with("The transcript has 2 items")),
        "{:?}",
        agent_request.message_input_texts("user")
    );

    let after = &requests[3];
    let user_texts = after.message_input_texts("user");
    assert!(
        user_texts.iter().any(|text| text == USER_GOAL),
        "{user_texts:?}"
    );
    assert!(
        user_texts
            .iter()
            .any(|text| text == &format!("{SUMMARY_PREFIX}\n{AGENT_SUMMARY}")),
        "{user_texts:?}"
    );
    assert!(!after.body_contains_text(DROPPED_REPLY));

    // The agent's working files stay on disk for inspection.
    let dir = only_compaction_dir(test.home.path());
    let transcript = std::fs::read_to_string(dir.join("transcript.md"))?;
    assert!(transcript.contains("=== #1 user"), "{transcript}");
    assert!(transcript.contains(USER_GOAL), "{transcript}");
    assert!(transcript.contains(DROPPED_REPLY), "{transcript}");
    assert_eq!(std::fs::read_to_string(dir.join("keep.txt"))?, "#1\n");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pre_turn_auto_compaction_uses_agent() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let write_output = format!("printf '{AGENT_SUMMARY}\\n' > summary.md && : > keep.txt");
    let request_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("parent-reply", DROPPED_REPLY),
                // Past the auto-compact limit, so the next turn compacts first.
                ev_completed_with_tokens("parent-response", /*total_tokens*/ 50_000),
            ]),
            sse(vec![
                ev_function_call(
                    "write-output",
                    "exec_command",
                    &json!({ "cmd": write_output }).to_string(),
                ),
                ev_completed_with_tokens("agent-tool", /*total_tokens*/ 200),
            ]),
            sse(vec![
                ev_assistant_message("agent-done", "Both files are written."),
                ev_completed_with_tokens("agent-done-response", /*total_tokens*/ 250),
            ]),
            sse(vec![
                ev_assistant_message("after-compaction", "continuing"),
                ev_completed_with_tokens("after-compaction-response", /*total_tokens*/ 120),
            ]),
        ],
    )
    .await;
    let test = build_agent_compaction_codex(&server, Some(10_000)).await?;

    test.submit_turn(USER_GOAL).await?;
    test.submit_turn("next request").await?;

    let requests = request_log.requests();
    assert_eq!(requests.len(), 4);
    assert!(
        requests[1]
            .instructions_text()
            .starts_with("You are the compaction agent")
    );
    let user_texts = requests[3].message_input_texts("user");
    assert!(
        user_texts
            .iter()
            .any(|text| text == &format!("{SUMMARY_PREFIX}\n{AGENT_SUMMARY}")),
        "{user_texts:?}"
    );
    // Nothing was kept, so only the summary and the incoming message remain.
    assert!(
        !user_texts.iter().any(|text| text == USER_GOAL),
        "{user_texts:?}"
    );
    assert!(
        user_texts.iter().any(|text| text == "next request"),
        "{user_texts:?}"
    );
    assert!(!requests[3].body_contains_text(DROPPED_REPLY));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_agent_output_is_retried_then_falls_back_to_standard_compaction() -> Result<()> {
    skip_if_no_network!(Ok(()));
    const LOCAL_SUMMARY: &str = "AGENT_COMPACTION_LOCAL_FALLBACK";

    let server = start_mock_server().await;
    let request_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("parent-reply", DROPPED_REPLY),
                ev_completed_with_tokens("parent-response", /*total_tokens*/ 100),
            ]),
            // The agent claims success twice without writing anything.
            sse(vec![
                ev_assistant_message("agent-first", "Done."),
                ev_completed_with_tokens("agent-first-response", /*total_tokens*/ 200),
            ]),
            sse(vec![
                ev_assistant_message("agent-retry", "Done again."),
                ev_completed_with_tokens("agent-retry-response", /*total_tokens*/ 220),
            ]),
            sse(vec![
                ev_assistant_message("local-summary", LOCAL_SUMMARY),
                ev_completed_with_tokens("local-summary-response", /*total_tokens*/ 150),
            ]),
            sse(vec![
                ev_assistant_message("after-compaction", "continuing"),
                ev_completed_with_tokens("after-compaction-response", /*total_tokens*/ 120),
            ]),
        ],
    )
    .await;
    let test = build_agent_compaction_codex(&server, /*auto_compact_token_limit*/ None).await?;

    test.submit_turn(USER_GOAL).await?;
    test.codex.submit(Op::Compact).await?;
    let warnings = warnings_until_turn_complete(&test).await;
    test.submit_turn("keep going").await?;

    assert_eq!(
        warnings,
        vec![
            "Agent compaction failed (invalid agent output: `summary.md` was not written; \
             `keep.txt` was not written (an empty file keeps nothing)); using standard \
             compaction."
                .to_string()
        ]
    );
    let requests = request_log.requests();
    assert_eq!(requests.len(), 5);
    let retry_texts = requests[2].message_input_texts("user");
    assert!(
        retry_texts.iter().any(|text| text
            .starts_with("Your output could not be used:\n- `summary.md` was not written\n")),
        "{retry_texts:?}"
    );
    assert!(requests[3].body_contains_text(SUMMARIZATION_PROMPT));
    assert!(
        requests[4]
            .message_input_texts("user")
            .iter()
            .any(|text| text == &format!("{SUMMARY_PREFIX}\n{LOCAL_SUMMARY}"))
    );
    Ok(())
}

async fn build_agent_compaction_codex(
    server: &MockServer,
    auto_compact_token_limit: Option<i64>,
) -> Result<TestCodex> {
    let model_provider = local_compaction_provider(server);
    let mut builder = test_codex().with_config(move |config| {
        config.model_provider = model_provider;
        config.model_auto_compact_token_limit = auto_compact_token_limit;
        config
            .features
            .enable(Feature::AgentCompaction)
            .expect("enable agent compaction");
    });
    builder.build_with_auto_env(server).await
}

async fn warnings_until_turn_complete(test: &TestCodex) -> Vec<String> {
    let mut warnings = Vec::new();
    wait_for_event(&test.codex, |event| {
        if let EventMsg::Warning(warning) = event {
            warnings.push(warning.message.clone());
        }
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    warnings
}

fn only_compaction_dir(home: &Path) -> PathBuf {
    let thread_dirs = std::fs::read_dir(home.join("compactions"))
        .expect("compactions directory")
        .collect::<Result<Vec<_>, _>>()
        .expect("thread directories");
    assert_eq!(thread_dirs.len(), 1);
    let runs = std::fs::read_dir(thread_dirs[0].path())
        .expect("thread compaction directory")
        .collect::<Result<Vec<_>, _>>()
        .expect("compaction runs");
    assert_eq!(runs.len(), 1);
    runs[0].path()
}
