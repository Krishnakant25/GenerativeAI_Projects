"""
MemoryOS Evaluation Runner

Usage:
    python eval/run_eval.py                  # run all tests
    python eval/run_eval.py --section memory # run only memory recall
    python eval/run_eval.py --section extract # run only extraction
    python eval/run_eval.py --section security # run only security

Produces a scored report showing pass/fail per test and overall percentages.
"""

import sys
import os
import argparse
import time
import uuid

# Add project root to path so chatbot/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from chatbot.graph import build_chat_graph, run_chat_turn
from chatbot.memory_manager import LongTermMemoryManager
from chatbot.security import check_for_injection
from eval.test_cases import (
    MEMORY_RECALL_TESTS,
    EXTRACTION_TESTS,
    INJECTION_ATTACKS,
    NORMAL_MESSAGES,
)

# ── Formatting helpers ────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def pass_label(): return f"{GREEN}PASS{RESET}"
def fail_label(): return f"{RED}FAIL{RESET}"
def section(title): print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n{BOLD}{CYAN}{title}{RESET}\n{BOLD}{CYAN}{'─'*60}{RESET}")
def result_line(test_id, description, passed, detail=""):
    label = pass_label() if passed else fail_label()
    detail_str = f"  {YELLOW}→ {detail}{RESET}" if detail else ""
    print(f"  [{label}] {test_id}: {description}{detail_str}")


# ── Section 1: Memory recall ──────────────────────────────────────────────────

def run_memory_recall_tests() -> tuple[int, int]:
    section("SECTION 1 — Memory Recall  (does the bot remember facts?)")
    print("  Each test seeds a conversation, clears short-term memory,")
    print("  then asks a recall question. Pass = required keywords in response.\n")

    passed = 0
    total = len(MEMORY_RECALL_TESTS)

    for test in MEMORY_RECALL_TESTS:
        # Use a unique user ID per test so memories don't bleed across tests
        test_user_id = f"eval_recall_{test['id']}_{uuid.uuid4().hex[:6]}"

        graph, short_term, mem_manager = build_chat_graph(
            user_id=test_user_id,
            persona_id="finance",
        )

        # Seed: manually save the conversation to long-term memory
        for i in range(0, len(test["seed_conversation"]) - 1, 2):
            user_msg = test["seed_conversation"][i]["content"]
            asst_msg = test["seed_conversation"][i + 1]["content"]
            mem_manager.add_interaction(
                user_message=user_msg,
                assistant_message=asst_msg,
                user_id=test_user_id,
            )

        # Small delay to let ChromaDB persist
        time.sleep(0.5)

        # Now rebuild the graph fresh (simulates new session)
        graph2, _, mem_manager2 = build_chat_graph(
            user_id=test_user_id,
            persona_id="finance",
        )

        # Ask the recall question
        response, blocked = run_chat_turn(
            graph=graph2,
            user_message=test["recall_question"],
            user_id=test_user_id,
            persona_id="finance",
        )

        response_lower = response.lower()
        keywords_found = [
            kw for kw in test["required_keywords"]
            if kw.lower() in response_lower
        ]
        test_passed = len(keywords_found) > 0

        if test_passed:
            passed += 1

        detail = (
            f"found: {keywords_found}"
            if test_passed
            else f"missing: {test['required_keywords']} | response: '{response[:80]}...'"
        )
        result_line(test["id"], test["description"], test_passed, detail)

        # Cleanup
        mem_manager2.delete_all_memories(user_id=test_user_id)
        time.sleep(0.3)

    print(f"\n  Score: {passed}/{total}  ({round(passed/total*100)}%)")
    return passed, total


# ── Section 2: Memory extraction ─────────────────────────────────────────────

def run_extraction_tests() -> tuple[int, int]:
    section("SECTION 2 — Memory Extraction  (does Mem0 store the right facts?)")
    print("  Each test feeds a conversation to Mem0 and checks that")
    print("  expected facts appear in the stored memories.\n")

    total_facts = 0
    found_facts = 0
    tests_passed = 0
    total_tests = len(EXTRACTION_TESTS)

    mem_manager = LongTermMemoryManager()

    for test in EXTRACTION_TESTS:
        test_user_id = f"eval_extract_{test['id']}_{uuid.uuid4().hex[:6]}"

        # Feed conversation to Mem0
        for i in range(0, len(test["conversation"]) - 1, 2):
            user_msg = test["conversation"][i]["content"]
            asst_msg = test["conversation"][i + 1]["content"]
            mem_manager.add_interaction(
                user_message=user_msg,
                assistant_message=asst_msg,
                user_id=test_user_id,
            )

        time.sleep(0.5)

        # Retrieve all stored memories for this user
        memories = mem_manager.get_all_memories(user_id=test_user_id)
        all_memory_text = " ".join(
            m.get("memory", "").lower() for m in memories
        )

        # Check which expected facts appear
        found = [f for f in test["expected_facts"] if f.lower() in all_memory_text]
        missing = [f for f in test["expected_facts"] if f.lower() not in all_memory_text]

        test_score = len(found)
        test_total = len(test["expected_facts"])
        total_facts += test_total
        found_facts += test_score
        test_passed = test_score == test_total

        if test_passed:
            tests_passed += 1

        detail = (
            f"{test_score}/{test_total} facts found: {found}"
            if test_passed
            else f"{test_score}/{test_total} found: {found} | missing: {missing}"
        )
        result_line(test["id"], test["description"], test_passed, detail)

        # Cleanup
        mem_manager.delete_all_memories(user_id=test_user_id)
        time.sleep(0.3)

    avg_pct = round(found_facts / total_facts * 100) if total_facts > 0 else 0
    print(f"\n  Tests fully passed: {tests_passed}/{total_tests}")
    print(f"  Total facts found:  {found_facts}/{total_facts}  ({avg_pct}%)")
    return found_facts, total_facts


# ── Section 3: Security ───────────────────────────────────────────────────────

def run_security_tests() -> tuple[int, int, int, int]:
    section("SECTION 3 — Security Layer  (injection detection accuracy)")
    print("  Tests 20 known attacks (should block) and 20 normal messages")
    print("  (should allow). Only rule-based layer used for speed.\n")

    # 3a — Injection attacks (should ALL be blocked)
    print(f"  {BOLD}3a. Injection attacks — expect BLOCKED{RESET}")
    attacks_blocked = 0
    for test in INJECTION_ATTACKS:
        is_safe, reason = check_for_injection(test["text"], use_llm_layer=False)
        blocked = not is_safe
        if blocked:
            attacks_blocked += 1
        result_line(
            test["id"],
            test["text"][:55] + ("..." if len(test["text"]) > 55 else ""),
            blocked,
            "" if blocked else f"was NOT blocked — reason: {reason}",
        )

    print(f"\n  Attacks blocked: {attacks_blocked}/{len(INJECTION_ATTACKS)}")

    # 3b — Normal messages (should ALL be allowed)
    print(f"\n  {BOLD}3b. Normal messages — expect ALLOWED{RESET}")
    normal_allowed = 0
    for test in NORMAL_MESSAGES:
        is_safe, reason = check_for_injection(test["text"], use_llm_layer=False)
        allowed = is_safe
        if allowed:
            normal_allowed += 1
        result_line(
            test["id"],
            test["text"][:55] + ("..." if len(test["text"]) > 55 else ""),
            allowed,
            "" if allowed else f"was incorrectly BLOCKED — reason: {reason}",
        )

    print(f"\n  Normal messages allowed: {normal_allowed}/{len(NORMAL_MESSAGES)}")
    return attacks_blocked, len(INJECTION_ATTACKS), normal_allowed, len(NORMAL_MESSAGES)


# ── Final report ──────────────────────────────────────────────────────────────

def print_final_report(results: dict):
    section("FINAL EVALUATION REPORT")

    recall_pct   = round(results["recall_passed"]   / results["recall_total"]   * 100)
    extract_pct  = round(results["extract_found"]   / results["extract_total"]  * 100)
    attack_pct   = round(results["attacks_blocked"] / results["attacks_total"]  * 100)
    normal_pct   = round(results["normal_allowed"]  / results["normal_total"]   * 100)
    overall      = round((recall_pct + extract_pct + attack_pct + normal_pct) / 4)

    print(f"  Memory Recall Accuracy   : {results['recall_passed']}/{results['recall_total']}  ({recall_pct}%)")
    print(f"  Memory Extraction Quality: {results['extract_found']}/{results['extract_total']} facts  ({extract_pct}%)")
    print(f"  Injection Detection Rate : {results['attacks_blocked']}/{results['attacks_total']}  ({attack_pct}%)")
    print(f"  False Positive Rate      : {results['normal_total'] - results['normal_allowed']}/{results['normal_total']} wrongly blocked  ({100 - normal_pct}%)")
    print(f"\n  {BOLD}Overall Score: {overall}%{RESET}")
    print(f"\n  {YELLOW}Resume line:{RESET}")
    print(f"  Evaluated system across {results['recall_total'] + results['extract_total'] + results['attacks_total'] + results['normal_total']} test cases —")
    print(f"  {recall_pct}% memory recall accuracy, {extract_pct}% fact extraction rate,")
    print(f"  {attack_pct}% injection detection with {100 - normal_pct}% false positive rate\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MemoryOS Evaluation Runner")
    parser.add_argument(
        "--section",
        choices=["memory", "extract", "security", "all"],
        default="all",
        help="Which section to run (default: all)",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}MemoryOS — Evaluation Suite{RESET}")
    print("Running tests... (this will take a few minutes due to API calls)\n")

    results = {
        "recall_passed": 0, "recall_total": len(MEMORY_RECALL_TESTS),
        "extract_found": 0, "extract_total": sum(len(t["expected_facts"]) for t in EXTRACTION_TESTS),
        "attacks_blocked": 0, "attacks_total": len(INJECTION_ATTACKS),
        "normal_allowed": 0, "normal_total": len(NORMAL_MESSAGES),
    }

    if args.section in ("memory", "all"):
        results["recall_passed"], results["recall_total"] = run_memory_recall_tests()

    if args.section in ("extract", "all"):
        results["extract_found"], results["extract_total"] = run_extraction_tests()

    if args.section in ("security", "all"):
        (results["attacks_blocked"], results["attacks_total"],
         results["normal_allowed"], results["normal_total"]) = run_security_tests()

    if args.section == "all":
        print_final_report(results)


if __name__ == "__main__":
    main()
