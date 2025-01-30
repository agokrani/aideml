import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aide.agent import Agent, ActionAgent
from aide.journal import Journal, Node
from aide.utils.metric import MetricValue, WorstMetricValue
from aide.utils.execution_result import ExecutionResult
from aide.actions import SubmitReview


@pytest.fixture
def mock_config(tmp_path):
    """
    Create a Config mock where `workspace_dir` points to a real,
    temporary directory. This prevents FileNotFoundError in agent.step.
    """
    # Create /tmp/workspace within the tmp_path
    # (though tmp_path is actually a unique test directory each run)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    config = MagicMock()
    config.agent.steps = 5
    config.agent.time_limit = 3600
    config.agent.data_preview = True
    config.agent.obfuscate = False
    config.agent.expose_prediction = True
    config.agent.k_fold_validation = 5
    config.agent.search.num_drafts = 2
    config.agent.search.debug_prob = 0.3
    config.agent.search.max_debug_depth = 3
    config.exec.timeout = 300
    config.exec.use_modal = False

    # Use the real, created directory
    config.workspace_dir = workspace_dir
    return config


@pytest.fixture
def mock_journal():
    return Journal()


@pytest.fixture
def agent(mock_config, mock_journal):
    return Agent(
        task_desc="Train a classifier on the given dataset",
        cfg=mock_config,
        journal=mock_journal,
    )


def test_agent_initialization(agent, mock_config):
    assert agent.task_desc == "Train a classifier on the given dataset"
    assert agent.cfg == mock_config
    assert agent.current_step == 0
    assert agent._initial_research is None
    assert agent.data_preview is None


def test_search_policy_initial_drafting(agent):
    # When not enough drafts, should return None to trigger new draft
    assert agent.search_policy() is None


def test_search_policy_debugging(agent):
    # First add enough draft nodes to satisfy the initial drafting check
    for _ in range(agent.acfg.search.num_drafts):
        draft_node = Node(plan="draft", code="code")
        draft_node.is_buggy = False
        draft_node.metric = MetricValue(
            0.5, maximize=True
        )  # Add metric to make it a good node
        agent.journal.append(draft_node)

    # Create a buggy leaf node
    node = Node(plan="test plan", code="test code")
    node.is_buggy = True
    agent.journal.append(node)

    # Mock random to force debug path
    with patch("random.random", return_value=0.1):  # Less than debug_prob
        selected_node = agent.search_policy()
        assert selected_node == node


def test_search_policy_improvement(agent):
    # Create a good node
    node = Node(plan="test plan", code="test code")
    node.is_buggy = False
    node.metric = MetricValue(0.8, maximize=True)
    agent.journal.append(node)

    # Add enough drafts
    for _ in range(agent.acfg.search.num_drafts):
        draft_node = Node(plan="draft", code="code")
        draft_node.is_buggy = False
        draft_node.metric = MetricValue(
            0.5, maximize=True
        )  # Add metric to make it a good node
        agent.journal.append(draft_node)

    # Mock random to avoid debug path
    with patch("random.random", return_value=0.9):  # Greater than debug_prob
        selected_node = agent.search_policy()
        assert selected_node == node


@pytest.mark.asyncio
@patch("aide.agent.query")
@patch("pathlib.Path.exists", return_value=True)  # Force submission file to appear
async def test_parse_exec_result_success(
    mock_exists,  # mocks Path(...).exists()
    mock_query,  # mocks aide.agent.query()
    agent,
):
    node = Node(plan="test plan", code="test code")
    exec_result = ExecutionResult(
        term_out=["Training completed. Validation accuracy: 0.85"],
        exec_time=0.1,
        exc_type=None,
        exc_info=None,
        exc_stack=None,
    )

    # Create a proper SubmitReview object
    submit_review = MagicMock(spec=SubmitReview)
    submit_review.summary = "Execution successful"
    submit_review.is_bug = False
    submit_review.metric = 0.85
    submit_review.has_csv_submission = True
    submit_review.lower_is_better = False
    submit_review.missing_libraries = []

    # Set up query mock to return our SubmitReview
    mock_query.return_value = submit_review

    # Mock callback_manager for async execution
    mock_callback_manager = MagicMock()
    mock_callback_manager.execute_callback = AsyncMock(
        side_effect=[
            True,  # has_submission check
        ]
    )

    result_node = await agent.parse_exec_result(
        node=node,
        exec_result=exec_result,
        callback_manager=mock_callback_manager,
        use_modal=False,
    )

    assert not result_node.is_buggy
    assert result_node.metric.value == 0.85
    assert result_node.analysis == "Execution successful"


@pytest.mark.asyncio
async def test_parse_exec_result_failure(agent):
    node = Node(plan="test plan", code="test code")
    exec_result = ExecutionResult(
        term_out=["ProcessLookupError: [Errno 8] Exec format error"],
        exec_time=0.1,
        exc_type="ProcessLookupError",
        exc_info={"msg": "ProcessLookupError: [Errno 8] Exec format error"},
        exc_stack=[],
    )

    with patch("aide.agent.query") as mock_query:
        # Create a proper SubmitReview object for failure case
        submit_review = SubmitReview(
            summary="Execution failed",
            is_bug=True,
            metric=None,
            has_csv_submission=False,
            lower_is_better=False,
            missing_libraries=[],
        )
        mock_query.return_value = submit_review

        result_node = await agent.parse_exec_result(node, exec_result)

        assert result_node.is_buggy
        assert isinstance(result_node.metric, WorstMetricValue)
        assert result_node.analysis == "Execution failed"


@pytest.mark.asyncio
async def test_step_draft(agent):
    # Mock necessary components
    mock_callback = AsyncMock()
    mock_callback_manager = MagicMock()
    mock_callback_manager.execute_callback = AsyncMock()

    with patch.object(agent, "_draft") as mock_draft, patch.object(
        agent, "parse_exec_result"
    ) as mock_parse:

        # Setup mock returns
        draft_node = Node(plan="draft plan", code="draft code")
        mock_draft.return_value = draft_node
        mock_parse.return_value = draft_node

        # Execute step
        await agent.step(mock_callback, mock_callback_manager)

        # Verify draft was called
        mock_draft.assert_called_once()
        assert agent.current_step == 1
        assert len(agent.journal.nodes) == 1


@pytest.mark.asyncio
async def test_step_debug(agent):
    """
    Ensure the agent debugs a buggy leaf node rather than returning None,
    by meeting the 'draft_nodes' requirement, forcing debug path via random(),
    returning a real SubmitReview from query, and a real ExecutionResult from callback_manager.
    """
    # Create submission and best_submission directories
    submission_dir = agent.cfg.workspace_dir / "submission"
    best_submission_dir = agent.cfg.workspace_dir / "best_submission"
    best_submission_dir.mkdir(parents=True, exist_ok=True)

    # 1) Add two "draft" nodes so that search_policy won't say "not enough drafts"
    draft_node_1 = Node(plan="draft1", code="draft1 code", parent=None, children=set())
    draft_node_1.is_buggy = False
    draft_node_1.metric = MetricValue(0.7, maximize=True)  # Add metric
    agent.journal.append(draft_node_1)

    draft_node_2 = Node(plan="draft2", code="draft2 code", parent=None, children=set())
    draft_node_2.is_buggy = False
    draft_node_2.metric = MetricValue(0.6, maximize=True)  # Add metric
    agent.journal.append(draft_node_2)

    # 2) Add a buggy leaf node (no children => leaf, parent=None => debug_depth=0).
    buggy_node = Node(
        plan="buggy plan",
        code="buggy code",
        parent=None,
        children=set(),
        _term_out=["Error: example error"],
    )
    buggy_node.is_buggy = True
    buggy_node.metric = WorstMetricValue()  # Add worst metric for buggy node
    agent.journal.append(buggy_node)

    # 3) Mock callback_manager to return a *real* ExecutionResult and create submission file
    mock_callback_manager = MagicMock()

    def execute_callback_side_effect(*args, **kwargs):
        if args[0] == "exec":
            # Create submission directory and file during "execution"
            submission_dir.mkdir(parents=True, exist_ok=True)
            with open(submission_dir / "submission.csv", "w") as f:
                f.write("id,prediction\n1,0.5\n")
            with open(submission_dir / "solution.py", "w") as f:
                f.write("print('Hello World')")

            return ExecutionResult(
                term_out=["Execution successful"],
                exec_time=0.1,
                exc_type=None,
                exc_info=None,
                exc_stack=None,
            )
        return AsyncMock().return_value

    mock_callback_manager.execute_callback = AsyncMock(
        side_effect=execute_callback_side_effect
    )

    # 4) Create a real SubmitReview for the execution evaluation
    submit_review = SubmitReview(
        summary="Fixed bug",
        is_bug=False,
        metric=0.85,
        has_csv_submission=True,
        lower_is_better=False,
        missing_libraries=[],
    )

    # 5) Mock query to return different responses for code generation vs execution evaluation
    def mock_query_side_effect(*args, **kwargs):
        # Check if functions argument is SubmitReview
        if "functions" in kwargs and kwargs["functions"] == SubmitReview:
            return submit_review
        # Otherwise return debug code
        return "```python\nprint('Hello World')\n```"

    # 6) Apply all patches
    with patch("aide.agent.query") as mock_query:
        mock_query.side_effect = mock_query_side_effect

        # Force debug path
        with patch("random.random", return_value=0.1):
            await agent.step(exec_callback=None, callback_manager=mock_callback_manager)

    # Verify we have a new node appended (draft_node_1, draft_node_2, buggy_node, and now debug_node)
    assert (
        len(agent.journal.nodes) == 4
    ), f"Expected 4 nodes, got {len(agent.journal.nodes)}"

    new_node = agent.journal.nodes[-1]
    assert (
        new_node is not buggy_node
    ), "Expected a newly created debug node, not the same old node."
    assert agent.current_step == 1, "Agent did not increment its step count."

    # The new debug node should not be buggy since we provided a successful execution
    assert not new_node.is_buggy, "Expected the debugged node to no longer be buggy."
    assert isinstance(
        new_node.metric, MetricValue
    ), "New node metric should be a real metric, not WorstMetricValue."
    assert (
        new_node.metric.value == 0.85
    ), "Agent did not set the correct metric on the debugged node."


@pytest.mark.asyncio
@patch("aide.agent.query")
@patch(
    "pathlib.Path.exists", return_value=True
)  # <-- Force the submission file to appear
async def test_parse_exec_result_missing_libraries(
    mock_exists,  # mocks Path(...).exists()
    mock_query,  # mocks aide.agent.query()
    agent,
):
    """
    If parse_exec_result sees missing libraries, it should install them once, re-execute,
    and update the node to is_buggy=False if a submission exists.
    """
    node = Node(plan="plan", code="code")
    # Initial execution result showing missing module
    exec_result = ExecutionResult(
        term_out=["ModuleNotFoundError: No module named x"],
        exec_time=0.1,
        exc_type=None,  # Or set to something like 'ModuleNotFoundError' if relevant
        exc_info=None,
        exc_stack=[],
    )

    # First "SubmitReview": says we are missing libs
    submit_review_missing = MagicMock(spec=SubmitReview)
    submit_review_missing.summary = "Missing library x"
    submit_review_missing.is_bug = True
    submit_review_missing.metric = None
    submit_review_missing.has_csv_submission = False
    submit_review_missing.lower_is_better = False
    submit_review_missing.missing_libraries = ["x"]

    # Second "SubmitReview": success
    submit_review_success = MagicMock(spec=SubmitReview)
    submit_review_success.summary = "It worked!"
    submit_review_success.is_bug = False
    submit_review_success.metric = 0.85
    submit_review_success.has_csv_submission = True
    submit_review_success.lower_is_better = False
    submit_review_success.missing_libraries = []

    # Mock LLM responses (we call query twice)
    mock_query.side_effect = [submit_review_missing, submit_review_success]

    # Mock callback_manager for re-execution
    # Mock callback_manager for re-execution
    mock_callback_manager = MagicMock()
    mock_callback_manager.execute_callback = AsyncMock(
        side_effect=[
            exec_result,  # first exec call
            ExecutionResult(  # second exec call after installation
                term_out=["Training completed successfully"],
                exec_time=0.1,
                exc_type=None,
                exc_info=None,
                exc_stack=None,
            ),
        ]
    )

    # Parse result: should install library "x" and run again
    updated_node = await agent.parse_exec_result(
        node=node,
        exec_result=exec_result,
        callback_manager=mock_callback_manager,
        use_modal=False,
    )

    # Verify we tried installing the missing library
    mock_callback_manager.execute_callback.assert_any_call("install_dependecies", ["x"])

    # Now the agent sees "submission.csv" (mock_exists -> True),
    # so the node should not be buggy, and metric should be set
    assert updated_node.is_buggy is False, "Node should be marked non-buggy"
    assert updated_node.metric.value == 0.85, "Metric should be 0.85"
    assert (
        updated_node.analysis == "It worked!"
    ), "Analysis should match the second SubmitReview"


@pytest.mark.asyncio
async def test_action_agent_predict_next_action():
    cfg = MagicMock()
    cfg.agent.copilot.model = "gpt-4"
    cfg.agent.copilot.temp = 0.7
    cfg.agent.convert_system_to_user = False

    agent = ActionAgent("Train a classifier", cfg)

    with patch("aide.agent.query") as mock_query:
        mock_query.return_value.get_tool_message.return_value = "draft"

        user_messages = [{"role": "user", "content": "What should I do next?"}]
        # Because ActionAgent.predict_next_action() is an async method in your code,
        # we await it here.
        response = await agent.predict_next_action(user_messages)

        # The result is an ActionAgentLLMResponse or similar.
        # We get the "draft" message from get_tool_message()
        tool_message = response.get_tool_message()

        assert tool_message == "draft"
        assert len(agent.message_history) == 2  # user msg + the LLM "draft" message

        # Check that query was called
        mock_query.assert_called_once()
        _, kwargs = mock_query.call_args
        assert "functions" in kwargs
        assert len(kwargs["functions"]) == 4


@pytest.mark.asyncio
async def test_cache_best_node(agent):
    """Test cache_best_node is called via callback_manager when a node becomes best during agent.step()"""

    # Create a callback manager and track all execute_callback calls
    mock_callback_manager = MagicMock()
    callback_results = []

    async def mock_execute_callback(*args, **kwargs):
        callback_results.append((args, kwargs))
        if args[0] == "exec":
            return ExecutionResult(
                term_out=["Success"],
                exec_time=0.1,
                exc_type=None,
                exc_info=None,
                exc_stack=None,
            )
        return True

    mock_callback_manager.execute_callback = AsyncMock(
        side_effect=mock_execute_callback
    )

    # Setup first node as current best
    first_node = Node(plan="first plan", code="first code")
    first_node.is_buggy = False
    first_node.metric = MetricValue(0.7, maximize=True)
    agent.journal.append(first_node)

    # Set up mocks
    with patch("aide.agent.query") as mock_query, patch(
        "pathlib.Path.exists", return_value=True
    ), patch.object(agent, "_draft") as mock_draft:

        # Setup query to return successful review
        mock_query.return_value = SubmitReview(
            summary="Success",
            is_bug=False,
            metric=0.8,  # Better than current best
            has_csv_submission=True,
            lower_is_better=False,
            missing_libraries=[],
        )

        # Return new node from draft
        new_node = Node(plan="new plan", code="new code")
        mock_draft.return_value = new_node

        # Execute step which should trigger cache_best_node
        await agent.step(callback_manager=mock_callback_manager)

        # Verify cache_best_node was called
        assert any(
            args[0] == "cache_best_node" for args, _ in callback_results
        ), "cache_best_node callback was not called"

        # Get the cache_best_node callback call and verify its argument
        cache_call = next(
            call for call in callback_results if call[0][0] == "cache_best_node"
        )
        cached_node = cache_call[0][1]
        assert cached_node.metric.value == 0.8, "Wrong node was cached"
        assert cached_node.id == new_node.id, "Wrong node was cached"

        assert agent.cfg.workspace_dir.joinpath(
            "best_solution"
        ).exists(), "best_solution directory was not created"
        assert agent.cfg.workspace_dir.joinpath(
            "best_submission"
        ).exists(), "best_submission directory was not created"
