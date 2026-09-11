#!/usr/bin/env python3
"""Generate Midge's reproducible NL-to-action training and evaluation corpus."""

from __future__ import annotations

import argparse
import json
import random
import sys
import types
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Corpus generation only exercises pure validation functions. Keep it usable in
# data-only/HPC staging environments before runtime dependencies are installed.
try:
    import requests  # noqa: F401
except ImportError:
    sys.modules["requests"] = types.ModuleType("requests")
try:
    import mido  # noqa: F401
except ImportError:
    mido_stub = types.ModuleType("mido")
    mido_stub.Message = object
    sys.modules["mido"] = mido_stub

from llm_parser import validate_action  # noqa: E402

SEED = 5636
SPLIT_SIZES = {"train": 480, "validation": 80, "test": 80}
ALIASES = [
    "kick drum", "snare", "closed hi hat", "open hi hat", "ride",
    "crash", "tom 1", "tom 2", "tom 3", "rim", "clap", "cowbell",
]
GROUPS = {
    "toms": ["tom 1", "tom 2", "tom 3"],
    "cymbals": ["closed hi hat", "open hi hat", "ride", "crash"],
    "drums": ["kick drum", "snare", "tom 1", "tom 2", "tom 3"],
    "accents": ["rim", "clap", "cowbell"],
}
SOURCES = ["kboard", "simmons"]


def base_rule(rule_id: str, alias: str = "snare") -> dict:
    return {
        "id": rule_id, "name": rule_id.replace("-", " "), "enabled": True,
        "priority": 100, "when": {"event": "note_on", "alias": alias},
        "condition": {},
        "derive": {"source": "input_note", "select": "first", "transpose": 0},
        "output": {"type": "note", "channel": 4, "velocity": "input",
                   "suppress_original": True},
        "state_updates": [], "continue": False,
    }


CONTEXTS = {
    "default": {
        "aliases": {name: {"source": "simmons", "note": 36 + i, "channel": 10}
                    for i, name in enumerate(ALIASES)},
        "groups": GROUPS,
        "rules": {}, "rule_order": [],
    },
    "rules_loaded": {
        "aliases": {name: {"source": "simmons", "note": 36 + i, "channel": 10}
                    for i, name in enumerate(ALIASES)},
        "groups": GROUPS,
        "rules": {
            "snare-chord-accent": base_rule("snare-chord-accent", "snare"),
            "kick-pulse": base_rule("kick-pulse", "kick drum"),
            "tom-echo": base_rule("tom-echo", "tom 1"),
        },
        "rule_order": ["snare-chord-accent", "kick-pulse", "tom-echo"],
    },
}


def canonical(action: dict, context_ref: str = "default") -> dict:
    raw = deepcopy(action)
    validated = validate_action(deepcopy(raw), deepcopy(CONTEXTS[context_ref]), SOURCES)
    # update_rule is the sole action whose accepted model/wire representation
    # (a patch in "changes") differs from the validator's merged runtime form.
    return raw if raw.get("action") == "update_rule" else validated


def executable(category, subcategory, utterance, action, group, template,
               tags=(), context_ref="default", provenance="synthetic"):
    return {
        "category": category, "subcategory": subcategory, "utterance": utterance,
        "context_ref": context_ref, "target": canonical(action, context_ref),
        "expected_behavior": "execute", "reason": None, "sft_eligible": True,
        "tags": list(tags), "provenance": provenance,
        "template_id": template, "semantic_group": group,
    }


def negative(category, subcategory, utterance, behavior, reason, group, template,
             tags=(), provenance="synthetic"):
    return {
        "category": category, "subcategory": subcategory, "utterance": utterance,
        "context_ref": "default", "target": None,
        "expected_behavior": behavior, "reason": reason, "sft_eligible": False,
        "tags": list(tags), "provenance": provenance,
        "template_id": template, "semantic_group": group,
    }


def rule_action(i, alias, group, every_n, select, transpose, channel,
                *, velocity=None, active_only=False, state_guard=False,
                fixed=None, output_type="note"):
    condition = {}
    if every_n > 1:
        condition["every_n"] = every_n
    if velocity:
        condition[velocity[0]] = velocity[1]
    if active_only:
        condition["active_context_group"] = group
    if state_guard:
        condition["state_key"] = "fill_mode"
        condition["equals"] = True
    derive = ({"source": "fixed", "notes": fixed, "select": "first", "transpose": transpose}
              if fixed else
              {"source": "active_chord", "selector_group": group,
               "select": select, "transpose": transpose})
    descriptor = "fixed note" if fixed else f"{group} {select.replace('chord_', '')}"
    rule_name = f"{alias} {descriptor} rule"
    rule = {
        "id": "-".join(rule_name.split()), "name": rule_name,
        "enabled": True, "priority": 100,
        "when": {"event": "note_on", "alias": alias}, "condition": condition,
        "derive": derive,
        "output": {"type": output_type, "channel": channel, "velocity": "input",
                   "suppress_original": True},
        "state_updates": [], "continue": False,
    }
    return {"action": "install_rule", "rule": rule}


def build_positive() -> list[dict]:
    rows = []
    route_templates = [
        "Route {name} to MIDI channel {ch}", "Send the {name} out on channel {ch}",
        "Put {name} on ch {ch}", "I need {name} coming through channel {ch}",
        "channel {ch} for the {name} please",
    ]
    for i in range(110):
        kind = i % 3
        ch = 1 + (i * 7) % 16
        if kind == 0:
            name = ALIASES[(i // 3) % len(ALIASES)]
            action = {"action": "route_alias", "alias": name, "channel": ch}
            subtype = "route_alias"
        elif kind == 1:
            name = list(GROUPS)[(i // 3) % len(GROUPS)]
            action = {"action": "route_group", "group": name, "channel": ch}
            subtype = "route_group"
        else:
            source = SOURCES[(i // 3) % 2]
            name = "keyboard" if source == "kboard" else "Simmons kit"
            action = {"action": "route_source", "source": source, "channel": ch}
            subtype = "route_source"
        t = route_templates[(i // 11) % len(route_templates)]
        rows.append(executable("atomic", subtype, t.format(name=name, ch=ch), action,
                               f"route-{subtype}-{name}-{ch}", f"route-{subtype}-{i//22}",
                               ["atomic", "routing"]))

    simple_specs = []
    for alias in ALIASES:
        simple_specs += [
            ("mute", f"Mute {alias}", {"action": "mute_alias", "alias": alias}),
            ("unmute", f"Bring {alias} back in", {"action": "unmute_alias", "alias": alias}),
            ("clear_route", f"Clear the route for {alias}", {"action": "clear_alias_route", "alias": alias}),
        ]
    for group in GROUPS:
        simple_specs.append(("clear_group", f"Remove the {group} group route",
                             {"action": "clear_group_route", "group": group}))
    for source, spoken in [("kboard", "keyboard"), ("simmons", "drum kit")]:
        simple_specs.append(("passthrough", f"Let the {spoken} use its original channel",
                             {"action": "source_passthrough", "source": source}))
    for alias in ALIASES:
        for mode in ["note", "chord"]:
            simple_specs.append((f"capture_{mode}",
                f"Capture the next {mode} from the keyboard for {alias} on channel {(len(simple_specs)%16)+1}",
                {"action": f"capture_{mode}_trigger", "trigger_alias": alias,
                 "capture_source": "kboard", "channel": (len(simple_specs)%16)+1}))
        simple_specs.append(("clear_capture", f"Clear the captured trigger for {alias}",
                             {"action": "clear_trigger_assignment", "trigger_alias": alias}))
    alternates = ["", " please", " for the live set", " until I change it"]
    for i in range(90):
        subtype, text, action = simple_specs[i % len(simple_specs)]
        suffix = alternates[i // len(simple_specs)]
        rows.append(executable("atomic", subtype, text + suffix, action,
                               f"simple-{subtype}-{json.dumps(action, sort_keys=True)}",
                               f"simple-{subtype}-{i%len(simple_specs)}", ["atomic"]))

    selects = ["lowest", "highest", "root", "chord_third", "chord_fifth", "cycle", "random"]
    transposes = [-24, -12, 0, 12, 24]
    rule_templates = [
        "On {count} of {alias}, play the {sel} note of the current {group} chord {octave} on channel {ch}",
        "When {alias} fires, take the {sel} note from the active {group} harmony, {octave}, and send it to ch {ch} on {count}",
        "Use {alias} to trigger the {sel} {group} chord tone {octave} on MIDI {ch}, but only on {count}",
        "For {count} from {alias}, output the {sel} note in the current {group} context {octave} through channel {ch}",
    ]
    octave_text = {-24: "two octaves down", -12: "one octave down", 0: "untransposed",
                   12: "one octave up", 24: "two octaves up"}
    def ordinal(n):
        if 10 <= n % 100 <= 20: suffix = "th"
        else: suffix = {1:"st",2:"nd",3:"rd"}.get(n%10,"th")
        return f"{n}{suffix}"

    for i in range(208):
        alias = ALIASES[i % len(ALIASES)]
        group = list(GROUPS)[(i // 3) % len(GROUPS)]
        n = 1 + (i * 3) % 7
        sel = selects[(i // 5) % len(selects)]
        tr = transposes[(i // 7) % len(transposes)]
        ch = 1 + (i * 5) % 16
        velocity = (("min_velocity", 64) if (i // 6) % 2 == 0 else ("max_velocity", 72)) if i % 6 == 0 else None
        action = rule_action(i, alias, group, n, sel, tr, ch,
                             velocity=velocity,
                             state_guard=i % 5 == 0)
        count_text = "every hit" if n == 1 else f"every {ordinal(n)} hit"
        text = rule_templates[(i // 48) % len(rule_templates)].format(
            count=count_text, alias=alias, sel=sel.replace("chord_", "chord "), group=group,
            octave=octave_text[tr], ch=ch)
        if velocity:
            text += (f" when I hit it at velocity {velocity[1]} or harder" if velocity[0] == "min_velocity"
                     else f" only when the velocity is {velocity[1]} or softer")
        if i % 5 == 0:
            text += " but only while fill mode is on"
        rows.append(executable("compositional" if i >= 100 else "rules",
                               "conditional_chord_rule", text, action,
                               f"rule-{alias}-{group}-{n}-{sel}-{tr}-{ch}-{i%30}",
                               f"rule-pattern-{i//20}",
                               ["install_rule", "multi_operation"] +
                               (["context_constraint"] if i % 5 == 0 else []) +
                               (["exception"] if i % 6 == 0 else []) +
                               (["velocity"] if i % 6 == 0 else [])))

    fixed_templates = [
        "On every {n} {alias} hit, play MIDI note {note} on channel {ch}",
        "Have {alias} trigger note {note} through ch {ch}, but only on every {n} hit",
        "each {n} {alias} should send note {note} on MIDI channel {ch}",
    ]
    for i in range(50):
        alias = ALIASES[i % len(ALIASES)]; n = 2 + i % 6; note = 36 + (i * 7) % 60; ch = 1 + i % 16
        action = rule_action(300 + i, alias, "toms", n, "first", 0, ch, fixed=[note])
        text = fixed_templates[i % 3].format(n=ordinal(n), alias=alias, note=note, ch=ch)
        rows.append(executable("rules", "fixed_note_rule", text, action,
                               f"fixed-{alias}-{n}-{note}-{ch}", f"fixed-template-{i%3}",
                               ["install_rule", "fixed_note"] + (["noisy"] if i%3==2 else [])))

    update_phrases = {
        "disable_rule": "Disable {rule}", "enable_rule": "Turn {rule} back on",
        "delete_rule": "Delete the {rule} rule",
    }
    rules = ["snare-chord-accent", "kick-pulse", "tom-echo"]
    for i in range(27):
        action_name = list(update_phrases)[i % 3]; rule_id = rules[(i // 3) % 3]
        text = update_phrases[action_name].format(rule=rule_id.replace("-", " "))
        text += ["", " please", " for now", "—the existing one"][i // 12]
        rows.append(executable("contextual", action_name, text,
                               {"action": action_name, "rule_id": rule_id},
                               f"lifecycle-{action_name}-{rule_id}", f"lifecycle-{action_name}-{i//9}",
                               ["context", "ordering_dependency"], "rules_loaded"))
    for i in range(25):
        rule_id = rules[i % 3]; tr = transposes[i % 5]
        text = f"Change {rule_id.replace('-', ' ')} so it transposes {abs(tr)} semitones {'down' if tr < 0 else 'up' if tr > 0 else 'with no shift'}"
        rows.append(executable("contextual", "update_rule", text,
            {"action": "update_rule", "rule_id": rule_id, "changes": {"derive": {"transpose": tr}}},
            f"update-{rule_id}-{tr}", f"update-transpose-{i//5}",
            ["context", "reference", "ordering_dependency"], "rules_loaded"))

    # Noisy but executable paraphrases with varied ASR-like errors and repairs.
    noisy_templates = [
        "uh route {alias} to channel {ch}", "send teh {alias} out chanel {ch}",
        "{alias} on midi {ch} pls", "I mean {alias}, channel {ch}, thanks",
        "route the {alias}—sorry—through {ch}", "can ch {ch} take {alias}",
        "hey midge {alias} goes out on {ch}", "put {alias} thru channel {ch}",
        "need the {alias} coming outta midi {ch}", "rouet {alias} to chan {ch}",
    ]
    for i in range(50):
        alias = ALIASES[i % len(ALIASES)]; ch = 1 + (i * 11) % 16
        text = noisy_templates[i % len(noisy_templates)].format(alias=alias,ch=ch)
        rows.append(executable("robustness", "noisy_routing", text,
                               {"action": "route_alias", "alias": alias, "channel": ch},
                               f"noisy-route-{alias}-{ch}", f"noise-{i//10}",
                               ["noisy", "typo", "routing"]))
    return rows


def build_negative() -> list[dict]:
    rows = []
    ambiguous = [
        "Route it to channel 4", "Mute that one", "Put the drums over there",
        "Use the other tom", "Make it louder", "Change the rule", "Send that to five",
    ]
    for i in range(35):
        text = ambiguous[i % len(ambiguous)] + (["", " please", " now", " instead", " like before"][i // 7])
        rows.append(negative("negative", "ambiguous", text, "clarify",
            "The request lacks a uniquely resolvable alias, group, source, rule, or parameter.",
            f"ambiguous-{i%len(ambiguous)}", f"ambiguous-{i%len(ambiguous)}", ["ambiguity", "reference"]))
    ood = [
        "What is the weather tomorrow?", "Write me a bass line", "Order more MIDI cables",
        "Open Spotify and play Kraftwerk", "Explain quantum entanglement", "Turn off the kitchen lights",
        "Email the set list to the band", "Who won the game?", "Render this song as an MP3",
        "Tell me a joke about drummers",
    ]
    for i in range(25):
        text = ood[i % len(ood)] + (["", " please", " right now"][i // 10])
        rows.append(negative("negative", "out_of_domain", text, "reject",
            "The request is outside Midge's supported MIDI action contract.",
            f"ood-{i%len(ood)}", f"ood-{i%len(ood)}", ["out_of_domain"]))
    sequential = [
        "Mute the snare and route the kick to channel 8",
        "Clear the toms route, then put the cymbals on channel 2",
        "Route the keyboard to 3 and the drum kit to 10",
        "Delete kick pulse, then enable tom echo",
        "Capture a chord for the snare and mute the ride",
    ]
    for i in range(20):
        text = sequential[i % len(sequential)] + (["", " please", " in that order", " before the set"][i // 5])
        rows.append(negative("negative", "multiple_actions", text, "clarify",
            "Midge's current parser accepts exactly one action object; the request must be split while preserving order.",
            f"sequential-{i%len(sequential)}", f"sequential-{i%len(sequential)}", ["ordering_dependency", "unsupported_multi_action"]))
    return rows


def split_rows(rows: list[dict]) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    # Keep every semantic group in one split while optimizing distribution by
    # both broad category and fine-grained subcategory.
    groups = defaultdict(list)
    for row in rows:
        groups[row["semantic_group"]].append(row)
    group_items=list(groups.values())
    total_sub=Counter(r["subcategory"] for r in rows)
    total_cat=Counter(r["category"] for r in rows)
    ratios={s:SPLIT_SIZES[s]/len(rows) for s in SPLIT_SIZES}
    best=None
    for trial in range(200):
        trial_rng=random.Random(SEED+trial)
        trial_rng.shuffle(group_items)
        ordered=sorted(group_items,key=lambda g:(len(g),-total_sub[g[0]["subcategory"]]),reverse=True)
        splits={name:[] for name in SPLIT_SIZES}; sub_counts=defaultdict(Counter); cat_counts=defaultdict(Counter)
        remaining=[]
        lifecycle_index={"snare-chord-accent":"train","kick-pulse":"validation","tom-echo":"test"}
        for group in ordered:
            key=group[0]["semantic_group"]
            forced=next((split for rule_id,split in lifecycle_index.items() if key.startswith("lifecycle-") and key.endswith(rule_id)),None)
            forced = forced or {"ambiguous-0":"validation", "ambiguous-1":"test",
                                "sequential-0":"validation", "sequential-1":"test"}.get(key)
            if forced:
                splits[forced].extend(group); sub_counts[forced][group[0]["subcategory"]]+=len(group); cat_counts[forced][group[0]["category"]]+=len(group)
            else: remaining.append(group)
        failed=False
        for group in remaining:
            sub=group[0]["subcategory"]; cat=group[0]["category"]; size=len(group)
            choices=[s for s in SPLIT_SIZES if len(splits[s])+size<=SPLIT_SIZES[s]]
            if not choices: failed=True; break
            def cost(s):
                sub_target=total_sub[sub]*ratios[s]; cat_target=total_cat[cat]*ratios[s]
                sub_before=sub_counts[s][sub]; cat_before=cat_counts[s][cat]
                sub_after=sub_counts[s][sub]+size; cat_after=cat_counts[s][cat]+size
                distribution=((sub_after-sub_target)**2-(sub_before-sub_target)**2)
                distribution+=.25*((cat_after-cat_target)**2-(cat_before-cat_target)**2)
                fill=(len(splits[s])+size)/SPLIT_SIZES[s]
                return (distribution,fill,trial_rng.random())
            chosen=min(choices,key=cost)
            splits[chosen].extend(group); sub_counts[chosen][sub]+=size; cat_counts[chosen][cat]+=size
        if failed or {s:len(v) for s,v in splits.items()}!=SPLIT_SIZES: continue
        penalty=0
        for s in SPLIT_SIZES:
            for sub,n in total_sub.items(): penalty+=(sub_counts[s][sub]-n*ratios[s])**2/max(1,n)
            for cat,n in total_cat.items(): penalty+=.25*(cat_counts[s][cat]-n*ratios[s])**2/max(1,n)
        if best is None or penalty<best[0]: best=(penalty,splits)
    if best is None:
        raise RuntimeError(f"Unexpected split sizes: { {s: len(v) for s,v in splits.items()} }")
    splits=best[1]
    for split, items in splits.items():
        rng.shuffle(items)
        for i, row in enumerate(items, 1):
            row["split"] = split
            row["id"] = f"midge-{split[:3]}-{i:04d}"
    return splits


def ensure_unique_utterances(rows: list[dict]) -> None:
    """Resolve accidental template collisions deterministically before splitting."""
    seen = {}
    for row in rows:
        key = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in row["utterance"].lower()).split())
        occurrence = seen.get(key, 0)
        if occurrence:
            suffixes = [" in this patch", " for the live set", " until I change it", " for this song"]
            row["utterance"] += suffixes[(occurrence - 1) % len(suffixes)]
        seen[key] = occurrence + 1


def gold_rows() -> list[dict]:
    specs=[]
    atomic_specs=[
        ("Let channel sixteen carry the kick.",{"action":"route_alias","alias":"kick drum","channel":16},"routing"),
        ("Move the whole cymbal section over to MIDI 11.",{"action":"route_group","group":"cymbals","channel":11},"routing"),
        ("I want the keyboard passing its own channel through again.",{"action":"source_passthrough","source":"kboard"},"passthrough"),
        ("Silence the open hat without touching the closed one.",{"action":"mute_alias","alias":"open hi hat"},"mute"),
        ("The ride can come back now.",{"action":"unmute_alias","alias":"ride"},"unmute"),
        ("Forget where the tom group was routed.",{"action":"clear_group_route","group":"toms"},"clear_group"),
        ("Make the next chord I play on the K-Board belong to the snare, outputting on 7.",{"action":"capture_chord_trigger","trigger_alias":"snare","capture_source":"kboard","channel":7},"capture_chord"),
        ("Remove the note or chord that the cowbell had captured.",{"action":"clear_trigger_assignment","trigger_alias":"cowbell"},"clear_capture"),
    ]
    for i,(text,action,sub) in enumerate(atomic_specs):
        specs.append(executable("gold_atomic",sub,text,action,f"gold-atomic-{i}","gold-handwritten",["gold","paraphrase"],provenance="handwritten"))

    comp_specs=[
        ("Every fourth snare should pick the top note from the current tom chord, lift it an octave, and send it on four.","snare","toms",4,"highest",12,4,None,False),
        ("When the kick lands hard—velocity 96 or more—use the root of the active tom harmony two octaves below on channel 6.","kick drum","toms",1,"root",-24,6,("min_velocity",96),False),
        ("While a cymbal chord is the live context, rotate through its notes on successive rim hits and output them on MIDI 2.","rim","cymbals",1,"cycle",0,2,None,True),
        ("Let every third clap choose a random accent-chord pitch, down twelve semitones, through channel 9.","clap","accents",3,"random",-12,9,None,False),
        ("On the second, fourth, sixth and so on kick, take the lowest drum-context note and put it on channel 12.","kick drum","drums",2,"lowest",0,12,None,False),
        ("A softly played tom 1, no harder than velocity 50, should voice the third of the current tom chord on channel 8.","tom 1","toms",1,"chord_third",0,8,("max_velocity",50),False),
        ("For every fifth ride strike, send the fifth of the live cymbal harmony up two octaves on MIDI channel 15.","ride","cymbals",5,"chord_fifth",24,15,None,False),
        ("Each crash can mirror the first pitch of the active cymbal chord one octave lower, but only when that cymbal context is active; use channel 3.","crash","cymbals",1,"first",-12,3,None,True),
        ("Turn alternate cowbell hits into the highest note in the accents harmony on channel 10.","cowbell","accents",2,"highest",0,10,None,False),
        ("Every seventh closed hat, borrow the lowest note from the current drum chord, add twelve semitones, and emit channel 5.","closed hi hat","drums",7,"lowest",12,5,None,False),
        ("When tom 3 hits at velocity 80 or above, cycle the tom harmony two octaves downward through MIDI 13.","tom 3","toms",1,"cycle",-24,13,("min_velocity",80),False),
        ("Use every third open hat to play the middle chord tone—the chord third—from the cymbal context on channel one.","open hi hat","cymbals",3,"chord_third",0,1,None,False),
    ]
    for i,(text,alias,group,n,select,tr,ch,velocity,active) in enumerate(comp_specs):
        specs.append(executable("gold_compositional","conditional_chord_rule",text,
            rule_action(900+i,alias,group,n,select,tr,ch,velocity=velocity,active_only=active),
            f"gold-rule-{i}","gold-handwritten",["gold","multi_operation","compositional"]+(["exception"] if velocity else [])+(["context_constraint"] if active else []),provenance="handwritten"))

    contextual_specs=[
        ("Take snare chord accent down another octave.",{"action":"update_rule","rule_id":"snare-chord-accent","changes":{"derive":{"transpose":-12}}},"update_rule"),
        ("The kick pulse is too high; make its output channel 2 instead.",{"action":"update_rule","rule_id":"kick-pulse","changes":{"output":{"channel":2}}},"update_rule"),
        ("Make tom echo respond only every third hit.",{"action":"update_rule","rule_id":"tom-echo","changes":{"condition":{"every_n":3}}},"update_rule"),
        ("Stop suppressing the original snare when snare chord accent fires.",{"action":"update_rule","rule_id":"snare-chord-accent","changes":{"output":{"suppress_original":False}}},"update_rule"),
        ("Pause kick pulse; don't erase it.",{"action":"disable_rule","rule_id":"kick-pulse"},"disable_rule"),
        ("Reactivate the tom echo rule.",{"action":"enable_rule","rule_id":"tom-echo"},"enable_rule"),
        ("Remove snare chord accent permanently.",{"action":"delete_rule","rule_id":"snare-chord-accent"},"delete_rule"),
        ("Set kick pulse to transpose without any pitch shift.",{"action":"update_rule","rule_id":"kick-pulse","changes":{"derive":{"transpose":0}}},"update_rule"),
    ]
    for i,(text,action,sub) in enumerate(contextual_specs):
        specs.append(executable("gold_contextual",sub,text,action,f"gold-context-{i}","gold-handwritten",["gold","context","reference","ordering_dependency"],"rules_loaded",provenance="handwritten"))

    noisy_specs=[
        ("Could you route the—sorry—the snare to channle 14?",{"action":"route_alias","alias":"snare","channel":14},"noisy_routing"),
        ("i need teh cowbell out midi 3",{"action":"route_alias","alias":"cowbell","channel":3},"noisy_routing"),
        ("hey mute, uh, not the open hat, mute closed hi hat",{"action":"mute_alias","alias":"closed hi hat"},"noisy_correction"),
        ("tom two thru 9—nine is the channel",{"action":"route_alias","alias":"tom 2","channel":9},"noisy_routing"),
        ("midge pls let k board pass thru normal",{"action":"source_passthrough","source":"kboard"},"noisy_passthrough"),
        ("unmoot the rim",{"action":"unmute_alias","alias":"rim"},"noisy_unmute"),
        ("next keybord note goes with crash, output five",{"action":"capture_note_trigger","trigger_alias":"crash","capture_source":"kboard","channel":5},"noisy_capture"),
        ("clear whatever chord the clap was trigering",{"action":"clear_trigger_assignment","trigger_alias":"clap"},"noisy_clear"),
    ]
    for i,(text,action,sub) in enumerate(noisy_specs):
        specs.append(executable("gold_robustness",sub,text,action,f"gold-noise-{i}","gold-handwritten",["gold","noisy"],provenance="handwritten"))

    negs = [
        ("Route it to channel four, but leave that one where it is.","clarify","ambiguous"),
        ("Make the current rule more musical.","clarify","ambiguous"),
        ("Send the other tom where the snare was before.","clarify","ambiguous"),
        ("Mute the one that's too loud.","clarify","ambiguous"),
        ("Route either tom 1 or tom 2 to channel 6.","clarify","ambiguous"),
        ("Change kick pulse to channel 3 and then disable it.","clarify","multiple_actions"),
        ("Clear the tom route before you move cymbals to 4.","clarify","multiple_actions"),
        ("Capture the next chord for snare and mute its original note.","clarify","multiple_actions"),
        ("Set the tempo to 120 BPM.","reject","unsupported_midi_capability"),
        ("Put a long reverb on the snare.","reject","unsupported_audio_capability"),
        ("Save all of this as a preset called encore.","reject","unsupported_midge_capability"),
        ("Send program change 18 to the Proteus.","reject","unsupported_midi_capability"),
    ]
    for i,(text,behavior,sub) in enumerate(negs):
        reason = ("The request is outside Midge's MIDI action contract." if behavior=="reject" else
                  "The request cannot resolve to exactly one safe supported action from the supplied context.")
        specs.append(negative("gold_negative", sub, text, behavior, reason,
            f"gold-negative-{i}", "gold-handwritten", ["gold",sub], provenance="handwritten"))
    assert len(specs) == 48
    for i,row in enumerate(specs,1): row.update(id=f"midge-gold-{i:03d}", split="gold")
    return specs


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True)+"\n" for r in rows), encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"data")
    args=parser.parse_args()
    rows=build_positive()+build_negative()
    ensure_unique_utterances(rows)
    assert len(rows)==640, len(rows)
    splits=split_rows(rows)
    for split,items in splits.items(): write_jsonl(args.output/"splits"/f"{split}.jsonl",items)
    write_jsonl(args.output/"gold_benchmark.jsonl",gold_rows())
    (args.output/"contexts.json").write_text(json.dumps({"sources":SOURCES,"contexts":CONTEXTS},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"Generated {sum(map(len,splits.values()))} split examples and 48 gold cases in {args.output}")


if __name__ == "__main__": main()
