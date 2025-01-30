# Mock igraph before importing journal
import sys
import pytest
from unittest.mock import MagicMock
from aide.journal import (
    Journal,
    Node,
    InteractiveSession,
    get_path_to_node,
    get_longest_path,
    filter_on_path,
    filter_for_best_path,
    filter_journal,
    cache_best_node,
)
from aide.utils.metric import MetricValue, WorstMetricValue
from aide.utils.execution_result import ExecutionResult

sys.modules["igraph"] = MagicMock()


@pytest.fixture
def journal():
    return Journal()


@pytest.fixture
def sample_nodes():
    nodes = []
    # Create a simple tree structure:
    #       n1
    #      /  \
    #     n2   n3
    #    /  \
    #   n4   n5

    n1 = Node(plan="root plan", code="root code", parent=None)
    n2 = Node(plan="child1 plan", code="child1 code", parent=n1)
    n3 = Node(plan="child2 plan", code="child2 code", parent=n1)
    n4 = Node(plan="grandchild1 plan", code="grandchild1 code", parent=n2)
    n5 = Node(plan="grandchild2 plan", code="grandchild2 code", parent=n2)

    nodes.extend([n1, n2, n3, n4, n5])
    return nodes


def test_journal_initialization(journal):
    assert len(journal.nodes) == 0
    assert journal.get_best_node() is None


def test_journal_append(journal, sample_nodes):
    for node in sample_nodes:
        journal.append(node)

    assert len(journal.nodes) == len(sample_nodes)
    assert journal.nodes == sample_nodes


def test_get_best_node_no_nodes(journal):
    assert journal.get_best_node() is None


def test_get_best_node_with_metrics(journal):
    # Create nodes with different metrics
    n1 = Node(plan="plan1", code="code1")
    n1.metric = MetricValue(0.8, maximize=True)
    n1.is_buggy = False

    n2 = Node(plan="plan2", code="code2")
    n2.metric = MetricValue(0.9, maximize=True)
    n2.is_buggy = False

    n3 = Node(plan="plan3", code="code3")
    n3.metric = MetricValue(0.7, maximize=True)
    n3.is_buggy = False

    for node in [n1, n2, n3]:
        journal.append(node)

    best_node = journal.get_best_node()
    assert best_node == n2  # Should return node with highest metric


def test_get_best_node_with_buggy_nodes(journal):
    # Create mix of buggy and non-buggy nodes
    n1 = Node(plan="plan1", code="code1")
    n1.metric = MetricValue(0.8, maximize=True)
    n1.is_buggy = False

    n2 = Node(plan="plan2", code="code2")
    n2.metric = MetricValue(0.9, maximize=True)
    n2.is_buggy = True  # Buggy node should be ignored

    n3 = Node(plan="plan3", code="code3")
    n3.metric = MetricValue(0.7, maximize=True)
    n3.is_buggy = False

    for node in [n1, n2, n3]:
        journal.append(node)

    best_node = journal.get_best_node()
    assert best_node == n1  # Should return best non-buggy node


def test_get_best_node_minimize_metric(journal):
    # Test when lower metric is better
    n1 = Node(plan="plan1", code="code1")
    n1.metric = MetricValue(0.3, maximize=False)
    n1.is_buggy = False

    n2 = Node(plan="plan2", code="code2")
    n2.metric = MetricValue(0.1, maximize=False)
    n2.is_buggy = False

    n3 = Node(plan="plan3", code="code3")
    n3.metric = MetricValue(0.2, maximize=False)
    n3.is_buggy = False

    for node in [n1, n2, n3]:
        journal.append(node)

    best_node = journal.get_best_node()
    assert best_node == n2  # Should return node with lowest metric


def test_buggy_nodes_property(journal, sample_nodes):
    # Make some nodes buggy
    sample_nodes[1].is_buggy = True
    sample_nodes[3].is_buggy = True

    for node in sample_nodes:
        journal.append(node)

    buggy_nodes = journal.buggy_nodes
    assert len(buggy_nodes) == 2
    assert sample_nodes[1] in buggy_nodes
    assert sample_nodes[3] in buggy_nodes


def test_good_nodes_property(journal, sample_nodes):
    # Make some nodes buggy
    sample_nodes[1].is_buggy = True
    sample_nodes[3].is_buggy = True

    for node in sample_nodes:
        journal.append(node)

    good_nodes = journal.good_nodes
    assert len(good_nodes) == 3
    assert sample_nodes[0] in good_nodes
    assert sample_nodes[2] in good_nodes
    assert sample_nodes[4] in good_nodes


def test_draft_nodes_property(journal, sample_nodes):
    # Draft nodes are nodes without parents
    for node in sample_nodes:
        journal.append(node)

    draft_nodes = journal.draft_nodes
    # Only n1 is a root node (no parent)
    assert len(draft_nodes) == 1
    assert sample_nodes[0] in draft_nodes  # n1


def test_generate_summary(journal, sample_nodes):
    # Set up nodes with different metrics and states
    sample_nodes[0].metric = MetricValue(0.8, maximize=True)
    sample_nodes[1].metric = MetricValue(0.85, maximize=True)
    sample_nodes[1].is_buggy = True
    sample_nodes[2].metric = MetricValue(0.9, maximize=True)
    sample_nodes[3].metric = WorstMetricValue()
    sample_nodes[4].metric = MetricValue(0.95, maximize=True)

    for node in sample_nodes:
        journal.append(node)

    summary = journal.generate_summary()
    assert isinstance(summary, str)
    assert "Design:" in summary
    assert "Results:" in summary
    assert "Validation Metric:" in summary
    assert "0.8" in summary  # Check for one of the metric values


def test_node_relationships(sample_nodes):
    n1, n2, n3, n4, n5 = sample_nodes

    # Test parent-child relationships
    assert n2.parent == n1
    assert n3.parent == n1
    assert n4.parent == n2
    assert n5.parent == n2

    # Test is_leaf property
    assert not n1.is_leaf
    assert not n2.is_leaf
    assert n3.is_leaf
    assert n4.is_leaf
    assert n5.is_leaf


def test_node_debug_depth(sample_nodes):
    n1, n2, n3, n4, n5 = sample_nodes

    # Make some nodes buggy to test debug depth
    n1.is_buggy = True  # Root node is buggy
    n2.is_buggy = True  # Child1 is buggy
    n3.is_buggy = False  # Child2 is not buggy
    n4.is_buggy = False  # Grandchild1 is not buggy
    n5.is_buggy = False  # Grandchild2 is not buggy

    # Test debug depth
    # n1 is a root node (draft), so debug_depth = 0
    assert (
        n1.debug_depth == 0
    ), "Root node should have debug_depth 0 as it's a draft node"

    # n2 is a debug node (parent n1 is buggy), so debug_depth = 1
    assert n2.debug_depth == 1, "n2 should have debug_depth 1 as its parent is buggy"

    # n3 is an debug node (parent n1 is buggy), so debug_depth = 1
    assert n3.debug_depth == 1, "n3 should have debug_depth 1 as it's parent is buggy"

    # n4 is a debug node (parent n2 is buggy), so debug_depth = 2 (consecutive debug steps)
    assert n4.debug_depth == 2, "n4 should have debug_depth 2 as it's in a debug chain"

    # n5 is a debug node (parent n2 is buggy), so debug_depth = 2 (consecutive debug steps)
    assert n5.debug_depth == 2, "n5 should have debug_depth 2 as it's in a debug chain"


def test_node_absorb_exec_result():
    node = Node(plan="test plan", code="test code")
    exec_result = ExecutionResult(
        term_out="test output", exec_time=1.0, exc_type=None, exc_info={}, exc_stack=[]
    )
    node.absorb_exec_result(exec_result)

    assert node._term_out == "test output"
    assert node.exec_time == 1.0
    assert node.exc_type is None
    assert node.exc_info == {}
    assert node.exc_stack == []


def test_node_term_out_property():
    node = Node(plan="test plan", code="test code")
    node._term_out = ["line1\n", "line2\n", "line3\n"]
    assert node.term_out == "line1\nline2\nline3\n"


def test_interactive_session():
    session = InteractiveSession()
    node1 = Node(plan="test plan 1", code="test code 1")
    node2 = Node(plan="test plan 2", code="test code 2")

    session.append(node1)
    session.append(node2)

    assert len(session.nodes) == 2
    assert node1.step == 0
    assert node2.step == 1


def test_interactive_session_generate_nb_trace():
    session = InteractiveSession()
    node = Node(plan="test plan", code="test code")
    node._term_out = ["test output"]
    session.append(node)

    trace = session.generate_nb_trace(include_prompt=True, comment_headers=True)
    assert "## In [1]:" in trace
    assert "test code" in trace
    assert "## Out [1]:" in trace
    assert "test output" in trace
    assert "## In [2]:" in trace  # For prompt


def test_get_path_to_node(journal, sample_nodes):
    for node in sample_nodes:
        journal.append(node)

    # Test path to leaf node n4
    path = get_path_to_node(journal, sample_nodes[3].id)  # n4
    assert len(path) == 3
    assert path[0] == sample_nodes[0].id  # n1
    assert path[1] == sample_nodes[1].id  # n2
    assert path[2] == sample_nodes[3].id  # n4


def test_get_longest_path(journal, sample_nodes):
    for node in sample_nodes:
        journal.append(node)

    longest_path = get_longest_path(journal)
    assert len(longest_path) == 3  # n1 -> n2 -> n4 or n5


def test_filter_on_path(journal, sample_nodes):
    for node in sample_nodes:
        journal.append(node)

    path = [
        sample_nodes[0].id,
        sample_nodes[1].id,
        sample_nodes[3].id,
    ]  # n1 -> n2 -> n4
    filtered_journal = filter_on_path(journal, path)

    assert len(filtered_journal.nodes) == 3
    assert filtered_journal.nodes[0].id == sample_nodes[0].id
    assert filtered_journal.nodes[1].id == sample_nodes[1].id
    assert filtered_journal.nodes[2].id == sample_nodes[3].id

    # Check that term_out and exc_stack are omitted
    for node in filtered_journal.nodes:
        assert node._term_out == "<OMITTED>"
        assert node.exc_stack == "<OMITTED>"


def test_filter_for_best_path(journal, sample_nodes):
    for node in sample_nodes:
        journal.append(node)

    # Make n4 the best node
    sample_nodes[3].metric = MetricValue(0.9, maximize=True)
    sample_nodes[3].is_buggy = False

    filtered_journal = filter_for_best_path(journal, sample_nodes[3].id)
    assert len(filtered_journal.nodes) == 3  # n1 -> n2 -> n4


def test_filter_journal_with_best_node(journal, sample_nodes):
    for node in sample_nodes:
        journal.append(node)
        node.is_buggy = False
        node.metric = MetricValue(0.5, maximize=True)

    # Make n4 the best node
    sample_nodes[3].metric = MetricValue(0.9, maximize=True)

    filtered_journal = filter_journal(journal)
    assert len(filtered_journal.nodes) == 3  # Should contain path to best node


def test_filter_journal_without_good_nodes(journal, sample_nodes):
    for node in sample_nodes:
        journal.append(node)
        node.is_buggy = True
        node.metric = WorstMetricValue()

    filtered_journal = filter_journal(journal)
    assert len(filtered_journal.nodes) == 3  # Should contain longest path


@pytest.fixture
def mock_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "submission").mkdir()
    (workspace / "submission" / "submission.csv").write_text("test,data\n1,2\n")
    return workspace


def test_cache_best_node(mock_workspace):
    node = Node(plan="test plan", code="test code")
    node.id = "test_id"

    cache_best_node(node, mock_workspace)

    # Check best_solution directory
    best_solution_dir = mock_workspace / "best_solution"
    assert best_solution_dir.exists()
    assert (best_solution_dir / "solution.py").read_text() == "test code"
    assert (best_solution_dir / "node_id.txt").read_text() == "test_id"

    # Check best_submission directory
    best_submission_dir = mock_workspace / "best_submission"
    assert best_submission_dir.exists()
    assert (best_submission_dir / "submission.csv").exists()


def test_node_stage_name():
    # Test draft node
    root = Node(plan="root", code="code")
    assert root.stage_name == "draft"

    # Test debug node
    parent = Node(plan="parent", code="code")
    parent.is_buggy = True
    debug_node = Node(plan="debug", code="code", parent=parent)
    assert debug_node.stage_name == "debug"

    # Test improve node
    good_parent = Node(plan="good parent", code="code")
    good_parent.is_buggy = False
    improve_node = Node(plan="improve", code="code", parent=good_parent)
    assert improve_node.stage_name == "improve"


def test_node_generate_summary():
    node = Node(plan="test plan", code="test code")
    node.analysis = "test analysis"
    node.metric = MetricValue(0.8, maximize=True)

    summary = node.generate_summary(include_code=True)
    assert "Design: test plan" in summary
    assert "Code: test code" in summary
    assert "Results: test analysis" in summary
    assert "Validation Metric: 0.8" in summary

    summary_no_code = node.generate_summary(include_code=False)
    assert "Code: test code" not in summary_no_code


def test_journal_get_metric_history(journal, sample_nodes):
    for node in sample_nodes:
        node.metric = MetricValue(0.5, maximize=True)
        journal.append(node)

    metric_history = journal.get_metric_history()
    assert len(metric_history) == len(sample_nodes)
    assert all(isinstance(m, MetricValue) for m in metric_history)


def test_journal_get_node_by_id(journal, sample_nodes):
    for node in sample_nodes:
        journal.append(node)

    # Test getting existing node
    node = journal.get(sample_nodes[2].id)
    assert node == sample_nodes[2]

    # Test getting non-existent node
    with pytest.raises(ValueError):
        journal.get("non-existent-id")
