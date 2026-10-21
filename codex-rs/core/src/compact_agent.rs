//! Agent compaction: a sub-agent reads the full rollout as a transcript file and writes the next
//! window's handoff summary plus a list of original items to keep verbatim.
//!
//! The agent always starts from the complete persisted trajectory, so no window is ever a
//! summary of a summary. Preparation runs before the compaction lifecycle starts: if the agent
//! fails, nothing has been emitted yet and the caller falls back to the standard strategy.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use codex_analytics::CompactionImplementation;
use codex_analytics::CompactionPhase;
use codex_analytics::CompactionReason;
use codex_analytics::CompactionTrigger;
use codex_features::Feature;
use codex_history::ResponseItemEnvelope;
use codex_protocol::config_types::WebSearchMode;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::items::ContextCompactionItem;
use codex_protocol::items::TurnItem;
use codex_protocol::models::BaseInstructionsProvenance;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::EnvironmentConfigState;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::SandboxPolicy;
use codex_protocol::protocol::SubAgentSource;
use codex_protocol::protocol::TurnEnvironmentRequest;
use codex_protocol::protocol::WarningEvent;
use codex_protocol::turn_input::TurnInputMode;
use codex_protocol::turn_input::TurnInputRequest;
use codex_protocol::turn_input::TurnInputSubmission;
use codex_protocol::turn_input::TurnStartOptions;
use codex_protocol::user_input::UserInput;
use codex_utils_absolute_path::AbsolutePathBuf;
use codex_utils_output_truncation::approx_token_count;
use codex_utils_path_uri::PathUri;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::codex_delegate::DelegateEnvironments;
use crate::codex_delegate::run_codex_thread_interactive_in_environments;
use crate::compact::CompactedHistoryMetadata;
use crate::compact::CompactionAnalyticsAttempt;
use crate::compact::CompactionAnalyticsDetails;
use crate::compact::SUMMARY_PREFIX;
use crate::compact::build_compaction_replacement_history;
use crate::compact_agent_transcript::AgentTranscript;
use crate::config::Config;
use crate::config::Constrained;
use crate::context::CompactionSummary;
use crate::context::ContextualUserFragment;
use crate::context::UserGoalUpdate;
use crate::context::world_state::WorldState;
use crate::hook_runtime::PostCompactHookOutcome;
use crate::hook_runtime::PreCompactHookOutcome;
use crate::hook_runtime::run_post_compact_hooks;
use crate::hook_runtime::run_pre_compact_hooks;
use crate::session::GitEnrichmentPolicy;
use crate::session::SessionIo;
use crate::session::session::Session;
use crate::session::step_context::StepContext;
use crate::session::turn_context::TurnContext;

use codex_prompts::AGENT_COMPACTION_PROMPT;

const TRANSCRIPT_FILE: &str = "transcript.md";
const SUMMARY_FILE: &str = "summary.md";
const KEEP_FILE: &str = "keep.txt";
/// Upper bound for verbatim items; also capped at a quarter of the auto-compact limit.
const KEPT_ITEMS_MAX_TOKENS: i64 = 64_000;
const SUMMARY_MAX_TOKENS: usize = 12_000;
/// The first attempt plus one retry with validation feedback.
const MAX_ATTEMPTS: usize = 2;
const AGENT_TIMEOUT: Duration = Duration::from_secs(15 * 60);

struct AgentCompactionOutput {
    summary: String,
    kept: Vec<ResponseItemEnvelope>,
    input_goal_ids: std::collections::HashSet<codex_protocol::ResponseItemId>,
}

/// Runs agent compaction and installs its result.
///
/// Returns `Ok(false)` when the agent could not produce a usable result; the caller should then
/// run the standard compaction strategy. `step_context` selects the summarizing model and
/// `replacement_step_context` owns the replacement window, as for the other strategies.
pub(crate) async fn run_agent_compact_task(
    sess: &Arc<Session>,
    step_context: &Arc<StepContext>,
    replacement_step_context: &Arc<StepContext>,
    world_state: Arc<WorldState>,
    trigger: CompactionTrigger,
    reason: CompactionReason,
    phase: CompactionPhase,
) -> CodexResult<bool> {
    let turn_context = &step_context.turn;
    // Box the agent's session startup to keep it off the turn stack.
    let output =
        match tokio::time::timeout(AGENT_TIMEOUT, Box::pin(prepare(sess, step_context))).await {
            Ok(Ok(output)) => output,
            Ok(Err(err)) => {
                fall_back(sess, turn_context, &format!("{err:#}")).await;
                return Ok(false);
            }
            Err(_) => {
                fall_back(sess, turn_context, "the compaction agent timed out").await;
                return Ok(false);
            }
        };

    let attempt = CompactionAnalyticsAttempt::begin(
        sess.as_ref(),
        turn_context.as_ref(),
        trigger,
        reason,
        CompactionImplementation::Agent,
        phase,
    )
    .await;
    let retained_count = output.kept.len();
    let analytics_details = CompactionAnalyticsDetails {
        compaction_summary_tokens: i64::try_from(approx_token_count(&output.summary)).ok(),
        ..Default::default()
    };
    if let PreCompactHookOutcome::Stopped = run_pre_compact_hooks(sess, turn_context, trigger).await
    {
        let error = CodexErr::TurnAborted;
        attempt
            .track(
                sess.as_ref(),
                codex_analytics::CompactionStatus::Interrupted,
                Some(&error),
                analytics_details,
            )
            .await;
        return Err(error);
    }

    let compaction_item = TurnItem::ContextCompaction(ContextCompactionItem::new());
    sess.emit_turn_item_started(turn_context, &compaction_item)
        .await;
    install(sess, replacement_step_context, &world_state, output).await;
    sess.emit_turn_item_completed(turn_context, compaction_item)
        .await;
    tracing::info!(retained_count, "installed agent compaction");

    let post_compact_outcome = run_post_compact_hooks(sess, turn_context, trigger).await;
    attempt
        .track(
            sess.as_ref(),
            codex_analytics::CompactionStatus::Completed,
            /*codex_error*/ None,
            analytics_details,
        )
        .await;
    if let PostCompactHookOutcome::Stopped = post_compact_outcome {
        return Err(CodexErr::TurnAborted);
    }
    Ok(true)
}

async fn fall_back(sess: &Session, turn_context: &TurnContext, reason: &str) {
    warn!(reason, "agent compaction failed; using standard compaction");
    sess.send_event(
        turn_context,
        EventMsg::Warning(WarningEvent {
            message: format!("Agent compaction failed ({reason}); using standard compaction."),
        }),
    )
    .await;
}

async fn install(
    sess: &Session,
    replacement_step_context: &StepContext,
    world_state: &WorldState,
    output: AgentCompactionOutput,
) {
    let turn_context = &replacement_step_context.turn;
    let summary_text = format!("{SUMMARY_PREFIX}\n{}", output.summary.trim());
    let mut summary_item = ResponseItemEnvelope::new(ContextualUserFragment::into(
        CompactionSummary::new(summary_text.clone()),
    ));
    summary_item.set_turn_id_if_missing(&turn_context.sub_id);
    let mut compacted_history = output.kept;
    compacted_history.push(summary_item);

    let (window_number, window_ids) = sess.advance_auto_compact_window().await;
    let (new_history, world_state_baseline) = build_compaction_replacement_history(
        sess,
        replacement_step_context,
        world_state,
        compacted_history,
    )
    .await;
    sess.replace_compacted_history(
        new_history,
        replacement_step_context.to_turn_context_item(),
        world_state_baseline,
        CompactedHistoryMetadata {
            input_goal_ids: output.input_goal_ids,
            message: summary_text,
            window_number,
            window_ids,
            compaction_response_id: None,
            compaction_model_hash: turn_context.model_info().comp_hash.clone(),
            reviewer_compaction_hash: None,
        },
    )
    .await;
    sess.recompute_token_usage(turn_context).await;
}

async fn prepare(
    sess: &Arc<Session>,
    step_context: &Arc<StepContext>,
) -> anyhow::Result<AgentCompactionOutput> {
    let turn_context = &step_context.turn;
    let live_thread = sess
        .live_thread()
        .context("the session is not persisted, so there is no full transcript")?;
    let input_goal_ids = UserGoalUpdate::message_ids(sess.clone_history().await.raw_items());
    sess.flush_rollout()
        .await
        .context("failed to flush the rollout")?;
    let history = live_thread
        .load_history(/*include_archived*/ true)
        .await
        .context("failed to load the rollout")?;
    let transcript = AgentTranscript::build(
        &history.items,
        turn_context.model_info().truncation_policy.into(),
    );
    anyhow::ensure!(
        !transcript.is_empty(),
        "the rollout has no items to compact"
    );

    let dir = turn_context
        .config
        .codex_home
        .join("compactions")
        .join(sess.thread_id.to_string())
        .join(uuid::Uuid::now_v7().to_string());
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("failed to create {}", dir.display()))?;
    tokio::fs::write(dir.join(TRANSCRIPT_FILE), transcript.rendered())
        .await
        .context("failed to write the transcript")?;

    let kept_budget = turn_context
        .model_info()
        .auto_compact_token_limit()
        .map_or(KEPT_ITEMS_MAX_TOKENS, |limit| limit / 4)
        .min(KEPT_ITEMS_MAX_TOKENS);
    let config = agent_config(turn_context, &dir)?;
    // Dropping this future (for example on interrupt) cancels and shuts down the agent.
    let cancel = CancellationToken::new();
    let _cancel_on_drop = cancel.clone().drop_guard();
    let (_agent_session, io) = Box::pin(run_codex_thread_interactive_in_environments(
        config,
        Arc::clone(&sess.services.auth_manager),
        Arc::clone(&sess.services.models_manager),
        Arc::clone(sess),
        Arc::clone(turn_context),
        DelegateEnvironments::Explicit(vec![local_environment(&dir)]),
        cancel,
        SubAgentSource::Compact,
        codex_extension_api::SessionIsolation::Isolated,
        /*initial_history*/ None,
        GitEnrichmentPolicy::Skip,
        codex_sandboxing::WindowsSandboxProxySettingsMode::Reconcile,
    ))
    .await
    .context("failed to start the compaction agent")?;

    let mut message = format!(
        "The transcript has {} items and is in `{TRANSCRIPT_FILE}`. The kept items' token \
         budget is {kept_budget}. Write `{SUMMARY_FILE}` and `{KEEP_FILE}`.",
        transcript.len(),
    );
    for attempt in 1..=MAX_ATTEMPTS {
        run_agent_turn(&io, turn_context, message).await?;
        match read_output(&dir, &transcript, kept_budget).await {
            Ok((summary, kept)) => {
                return Ok(AgentCompactionOutput {
                    summary,
                    kept,
                    input_goal_ids,
                });
            }
            Err(problems) if attempt < MAX_ATTEMPTS => {
                message = format!(
                    "Your output could not be used:\n- {}\nFix the files and reply when done.",
                    problems.join("\n- ")
                );
            }
            Err(problems) => anyhow::bail!("invalid agent output: {}", problems.join("; ")),
        }
    }
    unreachable!("the last attempt returns or bails")
}

fn agent_config(turn_context: &TurnContext, dir: &AbsolutePathBuf) -> anyhow::Result<Config> {
    let mut config = turn_context.config.as_ref().clone();
    config.cwd = dir.clone();
    // The agent may only write inside its own compaction directory.
    config
        .set_legacy_sandbox_policy(SandboxPolicy::new_workspace_write_policy())
        .map_err(|err| anyhow::anyhow!("cannot sandbox the compaction agent: {err}"))?;
    config.permissions.approval_policy = Constrained::allow_only(AskForApproval::Never);
    config
        .web_search_mode
        .set(WebSearchMode::Disabled)
        .map_err(|err| anyhow::anyhow!("cannot disable web search: {err}"))?;
    for feature in [
        Feature::Collab,
        Feature::MultiAgentV2,
        // The agent's own window falls back to standard compaction.
        Feature::AgentCompaction,
    ] {
        let _ = config.features.disable(feature);
    }
    config.base_instructions = Some(AGENT_COMPACTION_PROMPT.to_string());
    config.base_instructions_provenance = Some(BaseInstructionsProvenance::Custom);
    config.developer_instructions = None;
    config.model = Some(turn_context.model_info().slug.clone());
    config.ephemeral = true;
    Ok(config)
}

fn local_environment(dir: &AbsolutePathBuf) -> TurnEnvironmentRequest {
    let dir = PathUri::from_abs_path(dir);
    TurnEnvironmentRequest {
        environment_id: codex_exec_server::LOCAL_ENVIRONMENT_ID.to_string(),
        cwd: dir.clone(),
        workspace_roots: vec![dir],
        config: EnvironmentConfigState::FromThread,
    }
}

/// Starts one agent turn and waits for it to finish.
async fn run_agent_turn(
    io: &SessionIo,
    parent_ctx: &TurnContext,
    message: String,
) -> anyhow::Result<()> {
    let submission = io
        .submit_turn_input(
            TurnInputRequest::user_input(vec![UserInput::Text {
                text: message,
                text_elements: Vec::new(),
            }])
            .on_start(TurnStartOptions {
                parent_turn_id: Some(parent_ctx.sub_id.clone()),
                root_turn_id: parent_ctx.turn_metadata_state.root_turn_id(),
                ..Default::default()
            }),
            TurnInputMode::StartIfIdle,
        )
        .await
        .context("failed to submit the compaction agent's turn")?;
    anyhow::ensure!(
        matches!(submission, TurnInputSubmission::Started { .. }),
        "the compaction agent's turn did not start: {submission:?}"
    );
    while let Ok(event) = io.rx_event.recv().await {
        match event.msg {
            EventMsg::TurnComplete(_) => return Ok(()),
            EventMsg::TurnAborted(_) => anyhow::bail!("the compaction agent's turn was aborted"),
            _ => {}
        }
    }
    anyhow::bail!("the compaction agent exited before finishing its turn")
}

/// Reads and validates the agent's files, returning every problem for the retry message.
async fn read_output(
    dir: &AbsolutePathBuf,
    transcript: &AgentTranscript,
    kept_budget: i64,
) -> Result<(String, Vec<ResponseItemEnvelope>), Vec<String>> {
    let mut problems = Vec::new();
    let summary = match tokio::fs::read_to_string(dir.join(SUMMARY_FILE)).await {
        Ok(summary) if summary.trim().is_empty() => {
            problems.push(format!("`{SUMMARY_FILE}` is empty"));
            None
        }
        Ok(summary) if approx_token_count(&summary) > SUMMARY_MAX_TOKENS => {
            problems.push(format!(
                "`{SUMMARY_FILE}` is ~{} tokens; keep it under {SUMMARY_MAX_TOKENS}",
                approx_token_count(&summary)
            ));
            None
        }
        Ok(summary) => Some(summary),
        Err(_) => {
            problems.push(format!("`{SUMMARY_FILE}` was not written"));
            None
        }
    };
    let kept = match tokio::fs::read_to_string(dir.join(KEEP_FILE)).await {
        Ok(keep) => match transcript.resolve_keep_list(&keep, kept_budget) {
            Ok(kept) => Some(kept),
            Err(keep_problems) => {
                problems.extend(
                    keep_problems
                        .into_iter()
                        .map(|problem| format!("`{KEEP_FILE}`: {problem}")),
                );
                None
            }
        },
        Err(_) => {
            problems.push(format!(
                "`{KEEP_FILE}` was not written (an empty file keeps nothing)"
            ));
            None
        }
    };
    match (summary, kept) {
        (Some(summary), Some(kept)) if problems.is_empty() => Ok((summary, kept)),
        _ => Err(problems),
    }
}
