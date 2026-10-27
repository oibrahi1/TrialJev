"""The zero-shot probe must present the same two options in either order."""
import pytest

import prefinetune_probe as pp  # noqa: E402


@pytest.mark.parametrize("order", ["excluded_first", "eligible_first"])
def test_plain_prompt_orders_options(order):
    prompt = pp.build_prompt("PATIENT", "TRIAL", None, "plain", order)
    a, b = prompt.index("A. "), prompt.index("B. ")
    first, second = pp.ORDERS[order]
    assert prompt[a:b].lower().startswith(f"a. {first}")
    assert prompt[b:].lower().startswith(f"b. {second}")
    assert prompt.endswith("Answer:")


def test_orders_are_exact_swaps():
    assert pp.ORDERS["excluded_first"] == tuple(reversed(pp.ORDERS["eligible_first"]))
